"""Validate and serialize structured mono-input comparison records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerBase

from ...errors import ModelExecutionError
from .model import INPUT_INSTRUCTION


@dataclass(frozen=True, slots=True)
class MonoInputText:
    """One processed row serialized as a single Qwen input."""

    row_id: int
    is_swapped: bool
    text: str


@dataclass(frozen=True, slots=True)
class MonoInputModelData:
    """Validated augmented references and their row-aligned model inputs."""

    reference: pd.DataFrame
    texts: tuple[MonoInputText, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class MonoInputTestData:
    """Validated test IDs and both orientations of each mono-input text."""

    ids: tuple[int, ...]
    texts: tuple[MonoInputText, ...]
    fingerprint: str


class StructuredInputSerializer:
    """Serialize aligned turns as an instruction followed by deterministic JSON."""

    FORMAT_VERSION = "instruction-json-balanced-final-turn-v2"

    @property
    def version(self) -> str:
        """Return a cache key that changes when the editable instruction changes."""
        instruction_hash = hashlib.sha256(INPUT_INSTRUCTION.encode("utf-8")).hexdigest()
        return f"{self.FORMAT_VERSION}:{instruction_hash}"

    def serialize(
        self,
        row_id: int,
        is_swapped: bool,
        prompt_value: str,
        response_a_value: str,
        response_b_value: str,
    ) -> MonoInputText:
        """Return one instruction-plus-JSON input for an augmented row."""
        prompts = self._load_array(prompt_value, row_id, "prompt")
        response_a = self._load_array(response_a_value, row_id, "response_a")
        response_b = self._load_array(response_b_value, row_id, "response_b")
        if not prompts:
            raise ModelExecutionError(
                f"Row {row_id} contains an empty prompt conversation."
            )
        if len(prompts) != len(response_a) or len(prompts) != len(response_b):
            raise ModelExecutionError(
                f"Row {row_id} has misaligned prompt and response turn counts."
            )

        turns: list[dict[str, str]] = []
        for turn, (prompt, answer_a, answer_b) in enumerate(
            zip(prompts, response_a, response_b, strict=True)
        ):
            if not isinstance(prompt, str):
                raise ModelExecutionError(
                    f"Row {row_id} prompt turn {turn} must be a string."
                )
            for column, answer in (
                ("response_a", answer_a),
                ("response_b", answer_b),
            ):
                if answer is not None and not isinstance(answer, str):
                    raise ModelExecutionError(
                        f"Row {row_id} {column} turn {turn} must be a string or null."
                    )
            turns.append(
                {
                    "prompt": prompt,
                    "response_a": answer_a or "",
                    "response_b": answer_b or "",
                }
            )

        return MonoInputText(
            row_id=row_id,
            is_swapped=is_swapped,
            text=self._format(turns),
        )

    def fit_to_length(
        self,
        record: MonoInputText,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> str:
        """Keep complete JSON while giving the final responses equal token caps."""
        if self._token_count(record.text, tokenizer) <= max_length:
            return record.text

        turns = json.loads(record.text[len(INPUT_INSTRUCTION) + 1 :])
        final_turn = turns[-1]
        answer_a = final_turn["response_a"]
        answer_b = final_turn["response_b"]
        answer_a_ids = tokenizer.encode(answer_a, add_special_tokens=False)
        answer_b_ids = tokenizer.encode(answer_b, add_special_tokens=False)
        minimum_cap = int(bool(answer_a_ids or answer_b_ids))
        final_turn["response_a"] = self._answer_prefix(
            answer_a, answer_a_ids, minimum_cap, tokenizer
        )
        final_turn["response_b"] = self._answer_prefix(
            answer_b, answer_b_ids, minimum_cap, tokenizer
        )

        # Earlier complete turns are lower priority than the final prompt/pair.
        while len(turns) > 1 and self._token_count(
            self._format(turns), tokenizer
        ) > max_length:
            turns.pop(0)
        if self._token_count(self._format(turns), tokenizer) > max_length:
            raise ModelExecutionError(
                f"Row {record.row_id} final prompt and response prefixes exceed "
                f"the configured max_length of {max_length} tokens."
            )

        lower = minimum_cap
        upper = max(len(answer_a_ids), len(answer_b_ids))
        best = self._format(turns)
        while lower <= upper:
            cap = (lower + upper) // 2
            final_turn["response_a"] = self._answer_prefix(
                answer_a, answer_a_ids, cap, tokenizer
            )
            final_turn["response_b"] = self._answer_prefix(
                answer_b, answer_b_ids, cap, tokenizer
            )
            candidate = self._format(turns)
            if self._token_count(candidate, tokenizer) <= max_length:
                best = candidate
                lower = cap + 1
            else:
                upper = cap - 1
        return best

    @staticmethod
    def _answer_prefix(
        answer: str,
        token_ids: list[int],
        cap: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> str:
        if cap >= len(token_ids):
            return answer
        return tokenizer.decode(
            token_ids[:cap],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    @staticmethod
    def _token_count(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
        return len(tokenizer.encode(text, add_special_tokens=True))

    @staticmethod
    def _format(turns: list[dict[str, str]]) -> str:
        payload = json.dumps(turns, ensure_ascii=False, separators=(",", ":"))
        return f"{INPUT_INSTRUCTION}\n{payload}"

    @staticmethod
    def _load_array(value: str, row_id: int, column: str) -> list[Any]:
        if not isinstance(value, str):
            raise ModelExecutionError(f"Row {row_id} {column} must be JSON text.")
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ModelExecutionError(
                f"Row {row_id} {column} is not a valid JSON array: {error}"
            ) from error
        if not isinstance(parsed, list):
            raise ModelExecutionError(f"Row {row_id} {column} must be a JSON array.")
        return parsed


class MonoInputModelDataRepository:
    """Read, validate, and serialize every augmented training row."""

    REQUIRED_COLUMNS = (
        "id",
        "prompt",
        "response_a",
        "response_b",
        "label",
        "fold",
        "is_swapped",
    )
    SWAPPED_LABELS: ClassVar[dict[int, int]] = {0: 1, 1: 0, 2: 2}

    def __init__(self, processed_path: Path, n_splits: int) -> None:
        self._processed_path = processed_path.resolve()
        self._n_splits = n_splits
        self._serializer = StructuredInputSerializer()

    @property
    def serializer_version(self) -> str:
        return self._serializer.version

    def load(self) -> MonoInputModelData:
        if not self._processed_path.is_file():
            raise ModelExecutionError(
                f"Processed fold data not found: {self._processed_path}. "
                "Run 'uv run init' first."
            )
        try:
            data = pd.read_parquet(
                self._processed_path,
                columns=list(self.REQUIRED_COLUMNS),
            )
        except Exception as error:
            raise ModelExecutionError(
                f"Could not read processed fold data {self._processed_path}: {error}"
            ) from error

        self._validate_reference(data)
        data = data.reset_index(drop=True)
        texts = tuple(
            self._serializer.serialize(
                row_id=int(row.id),
                is_swapped=bool(row.is_swapped),
                prompt_value=row.prompt,
                response_a_value=row.response_a,
                response_b_value=row.response_b,
            )
            for row in data.itertuples(index=False)
        )
        return MonoInputModelData(
            reference=data.loc[:, ["id", "label", "fold", "is_swapped"]].copy(),
            texts=texts,
            fingerprint=_sha256(self._processed_path),
        )

    def _validate_reference(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ModelExecutionError("Processed fold data contains no rows.")
        if data.loc[:, list(self.REQUIRED_COLUMNS)].isna().any().any():
            raise ModelExecutionError(
                "Processed fold data contains missing required values."
            )
        if not data["label"].isin((0, 1, 2)).all():
            raise ModelExecutionError("Processed labels must be 0, 1, or 2.")
        if not data["is_swapped"].isin((True, False)).all():
            raise ModelExecutionError("Processed is_swapped values must be boolean.")
        expected_folds = set(range(self._n_splits))
        actual_folds = {int(value) for value in data["fold"].unique()}
        if actual_folds != expected_folds:
            raise ModelExecutionError(
                "Processed fold data must contain exactly the configured folds "
                f"0 through {self._n_splits - 1}."
            )

        pairs = data.groupby("id", sort=False).agg(
            rows=("id", "size"),
            folds=("fold", "nunique"),
            originals=("is_swapped", lambda values: int((~values).sum())),
            swapped=("is_swapped", lambda values: int(values.sum())),
        )
        if (
            pairs["rows"].ne(2)
            | pairs["folds"].ne(1)
            | pairs["originals"].ne(1)
            | pairs["swapped"].ne(1)
        ).any():
            raise ModelExecutionError(
                "Each ID must have one original and one swapped row in one fold."
            )
        originals = data.loc[~data["is_swapped"], ["id", "label"]].set_index("id")
        swapped = data.loc[data["is_swapped"], ["id", "label"]].set_index("id")
        swapped = swapped.loc[originals.index]
        translated = swapped["label"].map(self.SWAPPED_LABELS)
        if not translated.astype("int64").eq(originals["label"].astype("int64")).all():
            raise ModelExecutionError(
                "Swapped labels do not match their canonical labels."
            )


class MonoInputTestDataRepository:
    """Read and serialize both orientations of competition test records."""

    TEST_COLUMNS = ("id", "prompt", "response_a", "response_b")
    SUBMISSION_COLUMNS = (
        "id",
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    )

    def __init__(self, raw_dir: Path) -> None:
        self._test_path = raw_dir.resolve() / "test.csv"
        self._submission_path = raw_dir.resolve() / "sample_submission.csv"
        self._serializer = StructuredInputSerializer()

    @property
    def serializer_version(self) -> str:
        return self._serializer.version

    def load(self) -> MonoInputTestData:
        test = self._read_csv(self._test_path, "competition test data")
        submission = self._read_csv(self._submission_path, "sample submission")
        self._validate_columns(test, self.TEST_COLUMNS, self._test_path)
        self._validate_columns(
            submission,
            self.SUBMISSION_COLUMNS,
            self._submission_path,
        )
        if test.loc[:, list(self.TEST_COLUMNS)].isna().any().any():
            raise ModelExecutionError("Competition test data contains missing values.")
        test_ids = self._validated_ids(test, self._test_path)
        submission_ids = self._validated_ids(submission, self._submission_path)
        if not np.array_equal(test_ids, submission_ids):
            raise ModelExecutionError(
                "test.csv and sample_submission.csv IDs must match in the same order."
            )
        originals = tuple(
            self._serializer.serialize(
                row_id=int(row.id),
                is_swapped=False,
                prompt_value=row.prompt,
                response_a_value=row.response_a,
                response_b_value=row.response_b,
            )
            for row in test.itertuples(index=False)
        )
        swapped = tuple(
            self._serializer.serialize(
                row_id=int(row.id),
                is_swapped=True,
                prompt_value=row.prompt,
                response_a_value=row.response_b,
                response_b_value=row.response_a,
            )
            for row in test.itertuples(index=False)
        )
        return MonoInputTestData(
            ids=tuple(int(row_id) for row_id in test_ids),
            texts=originals + swapped,
            fingerprint=_sha256(self._test_path),
        )

    @staticmethod
    def _read_csv(path: Path, description: str) -> pd.DataFrame:
        if not path.is_file():
            raise ModelExecutionError(
                f"{description.capitalize()} not found: {path}. Run 'uv run data' first."
            )
        try:
            return pd.read_csv(path)
        except Exception as error:
            raise ModelExecutionError(
                f"Could not read {description} {path}: {error}"
            ) from error

    @staticmethod
    def _validate_columns(
        data: pd.DataFrame,
        expected: tuple[str, ...],
        path: Path,
    ) -> None:
        if tuple(data.columns) != expected:
            columns = ", ".join(expected)
            raise ModelExecutionError(
                f"{path.name} must contain exactly these columns in order: {columns}."
            )
        if data.empty:
            raise ModelExecutionError(f"{path.name} contains no rows.")

    @staticmethod
    def _validated_ids(data: pd.DataFrame, path: Path) -> np.ndarray:
        try:
            numeric_ids = pd.to_numeric(data["id"], errors="raise").to_numpy(
                dtype=np.float64
            )
        except (TypeError, ValueError) as error:
            raise ModelExecutionError(
                f"{path.name} IDs must be numeric integers."
            ) from error
        if (
            not np.isfinite(numeric_ids).all()
            or not np.equal(numeric_ids, np.floor(numeric_ids)).all()
        ):
            raise ModelExecutionError(f"{path.name} IDs must be finite integers.")
        integer_ids = numeric_ids.astype(np.int64)
        if pd.Series(integer_ids).duplicated().any():
            raise ModelExecutionError(f"{path.name} IDs must be unique.")
        return integer_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ModelExecutionError(f"Could not fingerprint {path}: {error}") from error
    return digest.hexdigest()


__all__ = (
    "MonoInputModelData",
    "MonoInputModelDataRepository",
    "MonoInputTestData",
    "MonoInputTestDataRepository",
    "MonoInputText",
    "StructuredInputSerializer",
)
