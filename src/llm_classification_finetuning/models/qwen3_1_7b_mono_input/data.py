"""Validate and serialize structured mono-input comparison records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import PreTrainedTokenizerBase

from ...errors import ModelExecutionError
from ..common import TestInputLoader, TrainingInputLoader

INPUT_INSTRUCTION = (
    "Compare response_a and response_b for prompts in the following JSON "
    "array and determine which response is better overall, or whether they are "
    "tied."
)


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
        while (
            len(turns) > 1
            and self._token_count(self._format(turns), tokenizer) > max_length
        ):
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

    REQUIRED_COLUMNS = TrainingInputLoader.REQUIRED_COLUMNS
    SWAPPED_LABELS = TrainingInputLoader.SWAPPED_LABELS

    def __init__(self, processed_path: Path, n_splits: int) -> None:
        self._loader = TrainingInputLoader(processed_path, n_splits)
        self._serializer = StructuredInputSerializer()

    @property
    def serializer_version(self) -> str:
        return self._serializer.version

    def load(self) -> MonoInputModelData:
        inputs = self._loader.load()
        data = inputs.data.reset_index(drop=True)
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
            fingerprint=inputs.fingerprint,
        )


class MonoInputTestDataRepository:
    """Read and serialize both orientations of competition test records."""

    TEST_COLUMNS = TestInputLoader.TEST_COLUMNS
    SUBMISSION_COLUMNS = TestInputLoader.SUBMISSION_COLUMNS

    def __init__(self, raw_dir: Path) -> None:
        self._loader = TestInputLoader(raw_dir)
        self._serializer = StructuredInputSerializer()

    @property
    def serializer_version(self) -> str:
        return self._serializer.version

    def load(self) -> MonoInputTestData:
        inputs = self._loader.load()
        test = inputs.data
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
            ids=inputs.ids,
            texts=originals + swapped,
            fingerprint=inputs.fingerprint,
        )


__all__ = (
    "MonoInputModelData",
    "MonoInputModelDataRepository",
    "MonoInputTestData",
    "MonoInputTestDataRepository",
    "MonoInputText",
    "StructuredInputSerializer",
)
