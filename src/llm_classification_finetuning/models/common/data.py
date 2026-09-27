"""Load and serialize processed pairwise training records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ...errors import ModelExecutionError
from .inputs import TestInputLoader, TrainingInputLoader


@dataclass(frozen=True, slots=True)
class PairwiseText:
    """One canonical source record serialized into two model inputs."""

    row_id: int
    response_a: str
    response_b: str


@dataclass(frozen=True, slots=True)
class ModelData:
    """Validated augmented references and canonical model inputs."""

    reference: pd.DataFrame
    canonical_texts: tuple[PairwiseText, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ModelTestData:
    """Validated competition test records and submission IDs."""

    ids: tuple[int, ...]
    canonical_texts: tuple[PairwiseText, ...]
    fingerprint: str


class ConversationSerializer:
    """Serialize aligned prompt/response JSON arrays as chronological turns."""

    FORMAT_VERSION = "aligned-prompt-newline-response-v1"

    def serialize_pair(
        self,
        row_id: int,
        prompt_value: str,
        response_a_value: str,
        response_b_value: str,
    ) -> PairwiseText:
        """Build the A and B branch text for one canonical row."""
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

        for turn, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise ModelExecutionError(
                    f"Row {row_id} prompt turn {turn} must be a string."
                )

        text_a = self._serialize_branch(row_id, prompts, response_a, "response_a")
        text_b = self._serialize_branch(row_id, prompts, response_b, "response_b")
        return PairwiseText(row_id=row_id, response_a=text_a, response_b=text_b)

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
                raise ModelExecutionError(
                    f"Row {row_id} {column} turn {turn} must be a string or null."
                )
            turns.append(f"{prompt}\n{response or ''}")
        return "\n\n".join(turns)


class ModelDataRepository:
    """Read and validate the processed fold artifact for model training."""

    REQUIRED_COLUMNS = TrainingInputLoader.REQUIRED_COLUMNS
    SWAPPED_LABELS = TrainingInputLoader.SWAPPED_LABELS

    def __init__(self, processed_path: Path, n_splits: int) -> None:
        self._loader = TrainingInputLoader(processed_path, n_splits)
        self._serializer = ConversationSerializer()

    @property
    def serializer_version(self) -> str:
        """Return the cache-relevant conversation format identifier."""
        return self._serializer.FORMAT_VERSION

    def load(self) -> ModelData:
        """Return validated references, canonical texts, and a file fingerprint."""
        inputs = self._loader.load()
        data = inputs.data
        canonical = data.loc[~data["is_swapped"]]
        canonical_texts = tuple(
            self._serializer.serialize_pair(
                row_id=int(row.id),
                prompt_value=row.prompt,
                response_a_value=row.response_a,
                response_b_value=row.response_b,
            )
            for row in canonical.itertuples(index=False)
        )
        fingerprint = inputs.fingerprint
        return ModelData(
            reference=data.loc[:, ["id", "label", "fold", "is_swapped"]]
            .copy()
            .reset_index(drop=True),
            canonical_texts=canonical_texts,
            fingerprint=fingerprint,
        )


class ModelTestDataRepository:
    """Read and validate competition test and submission-template records."""

    TEST_COLUMNS = TestInputLoader.TEST_COLUMNS
    SUBMISSION_COLUMNS = TestInputLoader.SUBMISSION_COLUMNS

    def __init__(self, raw_dir: Path) -> None:
        self._loader = TestInputLoader(raw_dir)
        self._serializer = ConversationSerializer()

    @property
    def serializer_version(self) -> str:
        """Return the cache-relevant conversation format identifier."""
        return self._serializer.FORMAT_VERSION

    def load(self) -> ModelTestData:
        """Return validated test texts in sample-submission order."""
        inputs = self._loader.load()
        test = inputs.data

        canonical_texts = tuple(
            self._serializer.serialize_pair(
                row_id=int(row.id),
                prompt_value=row.prompt,
                response_a_value=row.response_a,
                response_b_value=row.response_b,
            )
            for row in test.itertuples(index=False)
        )
        return ModelTestData(
            ids=inputs.ids,
            canonical_texts=canonical_texts,
            fingerprint=inputs.fingerprint,
        )
