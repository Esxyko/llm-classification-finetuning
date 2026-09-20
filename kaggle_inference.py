"""Standalone offline Kaggle inference script for LMSYS preference submission.

Required Kaggle inputs:
- Attach a dataset that contains ``test.csv``.
- Attach a dataset that contains ``sample_submission.csv``.
- Attach a dataset that contains the trained ``head.pt`` checkpoint.
- Attach a dataset that contains a local copy of the Qwen backbone/tokenizer files.

Important: the trained baseline checkpoint stores only the classifier head + metadata.
It does not include Qwen model/tokenizer weights.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

# Required file resolution behavior
TEST_CSV_PATH = os.getenv("KAGGLE_TEST_CSV")
SAMPLE_SUBMISSION_PATH = os.getenv("KAGGLE_SAMPLE_SUBMISSION")
CHECKPOINT_PATH = os.getenv("BASELINE_CHECKPOINT_PATH")
BACKBONE_PATH = os.getenv("BASELINE_BACKBONE_PATH")
OUTPUT_PATH = os.getenv("KAGGLE_OUTPUT_PATH", "/kaggle/working/submission.csv")

# Optional runtime overrides
DEVICE_OVERRIDE = os.getenv("KAGGLE_DEVICE")
INFERENCE_BATCH_SIZE = int(os.getenv("KAGGLE_INFERENCE_BATCH_SIZE", "256"))

# Baseline compatibility + architecture constants (must match repository semantics)
MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_LENGTH = 1024
SERIALIZER_VERSION = "aligned-prompt-newline-response-v1"
CHECKPOINT_SCHEMA_VERSION = 1
CLASS_COUNT = 3
CLASSIFIER_HIDDEN_SIZE = 512
DROPOUT = 0.1
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
ALLOW_TF32 = True

TEST_COLUMNS = ("id", "prompt", "response_a", "response_b")
SUBMISSION_COLUMNS = ("id", "winner_model_a", "winner_model_b", "winner_tie")
PROBABILITY_COLUMNS = SUBMISSION_COLUMNS[1:]


class InferenceError(RuntimeError):
    """Raised when inference setup, loading, or output validation fails."""


@dataclass(frozen=True, slots=True)
class PairwiseText:
    """Canonical pairwise serialized text for one row."""

    row_id: int
    response_a: str
    response_b: str


class ConversationSerializer:
    """Serialize aligned prompt/response JSON arrays exactly like baseline/data.py."""

    def serialize_pair(
        self,
        row_id: int,
        prompt_value: str,
        response_a_value: str,
        response_b_value: str,
    ) -> PairwiseText:
        prompts = self._load_array(prompt_value, row_id, "prompt")
        response_a = self._load_array(response_a_value, row_id, "response_a")
        response_b = self._load_array(response_b_value, row_id, "response_b")

        if not prompts:
            raise InferenceError(f"Row {row_id} contains an empty prompt conversation.")
        if len(prompts) != len(response_a) or len(prompts) != len(response_b):
            raise InferenceError(
                f"Row {row_id} has misaligned prompt and response turn counts."
            )

        for turn, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise InferenceError(f"Row {row_id} prompt turn {turn} must be a string.")

        text_a = self._serialize_branch(row_id, prompts, response_a, "response_a")
        text_b = self._serialize_branch(row_id, prompts, response_b, "response_b")
        return PairwiseText(row_id=row_id, response_a=text_a, response_b=text_b)

    @staticmethod
    def _load_array(value: str, row_id: int, column: str) -> list[Any]:
        if not isinstance(value, str):
            raise InferenceError(f"Row {row_id} {column} must be JSON text.")
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise InferenceError(
                f"Row {row_id} {column} is not a valid JSON array: {error}"
            ) from error
        if not isinstance(parsed, list):
            raise InferenceError(f"Row {row_id} {column} must be a JSON array.")
        return parsed

    @staticmethod
    def _serialize_branch(
        row_id: int,
        prompts: list[Any],
        responses: list[Any],
        column: str,
    ) -> str:
        turns: list[str] = []
        for turn, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
            if response is not None and not isinstance(response, str):
                raise InferenceError(
                    f"Row {row_id} {column} turn {turn} must be a string or null."
                )
            turns.append(f"{prompt}\n{response or ''}")
        return "\n\n".join(turns)


class PairwiseClassificationHead(nn.Module):
    """Exact baseline classifier head architecture."""

    def __init__(
        self,
        backbone_hidden_size: int,
        hidden_size: int = CLASSIFIER_HIDDEN_SIZE,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(backbone_hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, CLASS_COUNT),
        )

    def forward(self, h_a: Tensor, h_b: Tensor) -> Tensor:
        if h_a.shape != h_b.shape or h_a.ndim != 2:
            raise InferenceError("h_a and h_b must have the same [batch, hidden] shape.")
        features = torch.cat((h_a, h_b, h_a - h_b, h_a * h_b), dim=-1)
        return self.network(features)


@dataclass(frozen=True, slots=True)
class LoadedHeadCheckpoint:
    """Loaded and validated checkpoint payload."""

    state_dict: dict[str, Tensor]
    backbone_hidden_size: int


class KaggleInference:
    """End-to-end offline inference pipeline for Kaggle submission generation."""

    def __init__(self) -> None:
        self.device = self._resolve_device()
        self._configure_torch_backend()
        self.serializer = ConversationSerializer()

    def run(self) -> None:
        test_path = self._resolve_input_csv(
            explicit_path=TEST_CSV_PATH,
            env_name="KAGGLE_TEST_CSV",
            filename="test.csv",
        )
        sample_path = self._resolve_input_csv(
            explicit_path=SAMPLE_SUBMISSION_PATH,
            env_name="KAGGLE_SAMPLE_SUBMISSION",
            filename="sample_submission.csv",
        )
        checkpoint_path = self._resolve_required_path(
            CHECKPOINT_PATH,
            env_name="BASELINE_CHECKPOINT_PATH",
            description="baseline head checkpoint (head.pt)",
        )
        backbone_path = self._resolve_required_path(
            BACKBONE_PATH,
            env_name="BASELINE_BACKBONE_PATH",
            description="offline Qwen backbone/tokenizer directory",
            expect_directory=True,
        )

        test_df = self._read_csv_with_exact_columns(test_path, TEST_COLUMNS)
        sample_df = self._read_csv_with_exact_columns(sample_path, SUBMISSION_COLUMNS)
        self._validate_test_and_sample_alignment(test_df, sample_df)

        checkpoint = self._load_checkpoint(checkpoint_path)
        tokenizer = self._load_tokenizer(backbone_path)
        backbone = self._load_backbone(backbone_path)
        head = self._build_head(checkpoint)

        probabilities = self._predict_probabilities(
            test_df=test_df,
            tokenizer=tokenizer,
            backbone=backbone,
            head=head,
        )

        submission = pd.DataFrame(
            {
                "id": test_df["id"].to_numpy(copy=True),
                "winner_model_a": probabilities[:, 0],
                "winner_model_b": probabilities[:, 1],
                "winner_tie": probabilities[:, 2],
            }
        )
        self._validate_submission(submission, test_df, sample_df)
        output_path = Path(OUTPUT_PATH).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)

        print(
            f"Success: wrote {len(submission):,} rows to {output_path} "
            f"using device={self.device} batch_size={INFERENCE_BATCH_SIZE}."
        )

    def _resolve_device(self) -> torch.device:
        if DEVICE_OVERRIDE:
            return torch.device(DEVICE_OVERRIDE)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _configure_torch_backend(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
            torch.backends.cudnn.allow_tf32 = ALLOW_TF32

    @staticmethod
    def _resolve_input_csv(
        *,
        explicit_path: str | None,
        env_name: str,
        filename: str,
    ) -> Path:
        if explicit_path:
            path = Path(explicit_path).expanduser().resolve()
            if not path.is_file():
                raise InferenceError(
                    f"{env_name} points to a missing file: {path}."
                )
            return path

        kaggle_input_root = Path("/kaggle/input")
        candidates = sorted(
            path for path in kaggle_input_root.rglob(filename) if path.is_file()
        )
        if len(candidates) == 1:
            return candidates[0].resolve()

        if len(candidates) == 0:
            raise InferenceError(
                f"Could not find {filename} under /kaggle/input. "
                f"Set {env_name} to the exact file path."
            )
        candidate_list = "\n".join(f"- {path}" for path in candidates)
        raise InferenceError(
            f"Found multiple {filename} candidates under /kaggle/input:\n"
            f"{candidate_list}\n"
            f"Set {env_name} to the exact file path to disambiguate."
        )

    @staticmethod
    def _resolve_required_path(
        value: str | None,
        *,
        env_name: str,
        description: str,
        expect_directory: bool = False,
    ) -> Path:
        if not value:
            raise InferenceError(
                f"Missing required environment variable {env_name} for {description}."
            )
        path = Path(value).expanduser().resolve()
        exists = path.is_dir() if expect_directory else path.is_file()
        if not exists:
            expected_type = "directory" if expect_directory else "file"
            raise InferenceError(
                f"{env_name} must point to an existing {expected_type}: {path}."
            )
        return path

    @staticmethod
    def _read_csv_with_exact_columns(path: Path, expected_columns: tuple[str, ...]) -> pd.DataFrame:
        try:
            data = pd.read_csv(path)
        except Exception as error:
            raise InferenceError(f"Could not read {path}: {error}") from error
        if tuple(data.columns) != expected_columns:
            raise InferenceError(
                f"{path.name} must contain exactly these columns in order: "
                f"{','.join(expected_columns)}"
            )
        if data.empty:
            raise InferenceError(f"{path.name} contains no rows.")
        return data

    @staticmethod
    def _validate_test_and_sample_alignment(
        test_df: pd.DataFrame,
        sample_df: pd.DataFrame,
    ) -> None:
        if test_df.loc[:, list(TEST_COLUMNS)].isna().any().any():
            raise InferenceError("test.csv contains missing values.")
        if not test_df["id"].equals(sample_df["id"]):
            raise InferenceError(
                "test.csv and sample_submission.csv IDs must match exactly in the same order."
            )

    @staticmethod
    def _torch_dtype_for_backbone(device: torch.device) -> torch.dtype:
        if PRECISION == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32

    def _load_tokenizer(self, backbone_path: Path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                backbone_path,
                local_files_only=True,
            )
        except Exception as error:
            raise InferenceError(
                f"Could not load local tokenizer from {backbone_path}: {error}"
            ) from error

        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise InferenceError(
                    "Tokenizer has neither a pad token nor an EOS token."
                )
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_backbone(self, backbone_path: Path) -> nn.Module:
        dtype = self._torch_dtype_for_backbone(self.device)
        try:
            backbone = AutoModel.from_pretrained(
                backbone_path,
                torch_dtype=dtype,
                attn_implementation=ATTENTION_IMPLEMENTATION,
                local_files_only=True,
            )
        except Exception as error:
            raise InferenceError(
                f"Could not load local backbone from {backbone_path}: {error}"
            ) from error

        backbone.config.use_cache = False
        backbone.requires_grad_(False)
        backbone.to(self.device)
        backbone.eval()
        return backbone

    @staticmethod
    def _load_checkpoint(path: Path) -> LoadedHeadCheckpoint:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise InferenceError(f"Could not load checkpoint {path}: {error}") from error

        if not isinstance(checkpoint, dict):
            raise InferenceError("Checkpoint root must be a mapping.")
        metadata = checkpoint.get("metadata")
        state_dict = checkpoint.get("state_dict")
        if not isinstance(metadata, dict) or not isinstance(state_dict, dict):
            raise InferenceError("Checkpoint must contain metadata and state_dict mappings.")

        expected = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "serializer_version": SERIALIZER_VERSION,
            "precision": PRECISION,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "allow_tf32": ALLOW_TF32,
            "classifier_hidden_size": CLASSIFIER_HIDDEN_SIZE,
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise InferenceError(
                "Checkpoint metadata is incompatible for fields: "
                f"{', '.join(mismatches)}."
            )

        class_count = metadata.get("class_count")
        if class_count != CLASS_COUNT:
            raise InferenceError(
                f"Checkpoint class_count must be {CLASS_COUNT}, got {class_count!r}."
            )

        backbone_hidden_size = metadata.get("backbone_hidden_size")
        if (
            isinstance(backbone_hidden_size, bool)
            or not isinstance(backbone_hidden_size, int)
            or backbone_hidden_size < 1
        ):
            raise InferenceError("Checkpoint backbone_hidden_size must be a positive integer.")

        if not state_dict or not all(
            isinstance(name, str) and isinstance(value, Tensor)
            for name, value in state_dict.items()
        ):
            raise InferenceError("Checkpoint state_dict must contain tensor parameters.")
        if not all(torch.isfinite(value).all() for value in state_dict.values()):
            raise InferenceError("Checkpoint state_dict contains non-finite parameters.")

        return LoadedHeadCheckpoint(
            state_dict=state_dict,
            backbone_hidden_size=backbone_hidden_size,
        )

    def _build_head(self, checkpoint: LoadedHeadCheckpoint) -> PairwiseClassificationHead:
        head = PairwiseClassificationHead(
            backbone_hidden_size=checkpoint.backbone_hidden_size,
            hidden_size=CLASSIFIER_HIDDEN_SIZE,
            dropout=DROPOUT,
        ).to(device=self.device, dtype=torch.float32)
        try:
            head.load_state_dict(checkpoint.state_dict, strict=True)
        except RuntimeError as error:
            raise InferenceError(f"Checkpoint head parameters are invalid: {error}") from error
        head.eval()
        return head

    def _predict_probabilities(
        self,
        *,
        test_df: pd.DataFrame,
        tokenizer,
        backbone: nn.Module,
        head: PairwiseClassificationHead,
    ) -> np.ndarray:
        if INFERENCE_BATCH_SIZE < 1:
            raise InferenceError("KAGGLE_INFERENCE_BATCH_SIZE must be >= 1.")

        records = [
            self.serializer.serialize_pair(
                row_id=int(row.id),
                prompt_value=row.prompt,
                response_a_value=row.response_a,
                response_b_value=row.response_b,
            )
            for row in test_df.itertuples(index=False)
        ]

        all_probabilities: list[np.ndarray] = []
        hidden_size = int(backbone.config.hidden_size)

        with torch.inference_mode():
            for start in range(0, len(records), INFERENCE_BATCH_SIZE):
                batch = records[start : start + INFERENCE_BATCH_SIZE]
                flat_texts = [
                    text
                    for record in batch
                    for text in (record.response_a, record.response_b)
                ]
                tokenized = tokenizer(
                    flat_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                sequence_length = int(tokenized["input_ids"].shape[1])
                pair_count = len(batch)
                input_ids = tokenized["input_ids"].reshape(pair_count, 2, sequence_length)
                attention_mask = tokenized["attention_mask"].reshape(
                    pair_count, 2, sequence_length
                )

                input_ids = input_ids.to(self.device, non_blocking=True)
                attention_mask = attention_mask.to(self.device, non_blocking=True)

                outputs = backbone(
                    input_ids=input_ids.reshape(pair_count * 2, sequence_length),
                    attention_mask=attention_mask.reshape(pair_count * 2, sequence_length),
                    use_cache=False,
                    return_dict=True,
                )
                hidden_states = outputs.last_hidden_state.reshape(
                    pair_count,
                    2,
                    sequence_length,
                    hidden_size,
                )

                positions = torch.arange(sequence_length, device=attention_mask.device)
                positions = positions.view(1, 1, sequence_length).expand_as(attention_mask)
                last_positions = positions.masked_fill(attention_mask.eq(0), -1).amax(dim=-1)
                if last_positions.lt(0).any():
                    raise InferenceError(
                        "Every response branch must contain at least one token."
                    )

                gather_indices = last_positions[..., None, None].expand(-1, -1, 1, hidden_size)
                pooled = hidden_states.gather(dim=2, index=gather_indices).squeeze(dim=2)

                logits = head(
                    pooled[:, 0].to(dtype=torch.float32),
                    pooled[:, 1].to(dtype=torch.float32),
                )
                probabilities = torch.softmax(logits, dim=-1).to(device="cpu").numpy()
                all_probabilities.append(probabilities)

        if not all_probabilities:
            raise InferenceError("Inference produced no predictions.")
        return np.concatenate(all_probabilities, axis=0)

    @staticmethod
    def _validate_submission(
        submission: pd.DataFrame,
        test_df: pd.DataFrame,
        sample_df: pd.DataFrame,
    ) -> None:
        if tuple(submission.columns) != SUBMISSION_COLUMNS:
            raise InferenceError(
                f"Submission columns must be exactly: {','.join(SUBMISSION_COLUMNS)}"
            )
        if tuple(submission.columns) != tuple(sample_df.columns):
            raise InferenceError(
                "Submission columns/order must match sample_submission.csv exactly."
            )
        if len(submission) != len(test_df):
            raise InferenceError("Submission row count must equal test row count.")
        if not submission["id"].equals(test_df["id"]):
            raise InferenceError("Submission IDs must exactly match test.csv IDs in order.")

        values = submission.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        if np.isnan(values).any():
            raise InferenceError("Submission probabilities contain NaN values.")
        if not np.isfinite(values).all():
            raise InferenceError("Submission probabilities contain non-finite values.")
        if (values < 0.0).any() or (values > 1.0).any():
            raise InferenceError("Submission probabilities must be within [0, 1].")
        if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
            raise InferenceError(
                "Submission probability rows must sum to 1 within absolute tolerance 1e-6."
            )


def main() -> None:
    KaggleInference().run()


if __name__ == "__main__":
    main()
