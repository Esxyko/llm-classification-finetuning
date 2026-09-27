"""Load and serialize processed pairwise training records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..errors import BaselineError


@dataclass(frozen=True, slots=True)
class PairwiseText:
    """One canonical source record serialized into two model inputs."""

    row_id: int
    response_a: str
    response_b: str


@dataclass(frozen=True, slots=True)
class BaselineData:
    """Validated augmented references and canonical model inputs."""

    reference: pd.DataFrame
    canonical_texts: tuple[PairwiseText, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class BaselineTestData:
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
            raise BaselineError(f"Row {row_id} contains an empty prompt conversation.")
        if len(prompts) != len(response_a) or len(prompts) != len(response_b):
            raise BaselineError(
                f"Row {row_id} has misaligned prompt and response turn counts."
            )

        for turn, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise BaselineError(
                    f"Row {row_id} prompt turn {turn} must be a string."
                )

        text_a = self._serialize_branch(row_id, prompts, response_a, "response_a")
        text_b = self._serialize_branch(row_id, prompts, response_b, "response_b")
        return PairwiseText(row_id=row_id, response_a=text_a, response_b=text_b)

    @staticmethod
    def _load_array(value: str, row_id: int, column: str) -> list[Any]:
        if not isinstance(value, str):
            raise BaselineError(f"Row {row_id} {column} must be JSON text.")
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise BaselineError(
                f"Row {row_id} {column} is not a valid JSON array: {error}"
            ) from error
        if not isinstance(parsed, list):
            raise BaselineError(f"Row {row_id} {column} must be a JSON array.")
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
                raise BaselineError(
                    f"Row {row_id} {column} turn {turn} must be a string or null."
                )
            turns.append(f"{prompt}\n{response or ''}")
        return "\n\n".join(turns)


class BaselineDataRepository:
    """Read and validate the processed fold artifact for baseline training."""

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
        self._serializer = ConversationSerializer()

    @property
    def serializer_version(self) -> str:
        """Return the cache-relevant conversation format identifier."""
        return self._serializer.FORMAT_VERSION

    def load(self) -> BaselineData:
        """Return validated references, canonical texts, and a file fingerprint."""
        if not self._processed_path.is_file():
            raise BaselineError(
                f"Processed fold data not found: {self._processed_path}. "
                "Run 'uv run init' first."
            )
        try:
            data = pd.read_parquet(
                self._processed_path,
                columns=list(self.REQUIRED_COLUMNS),
            )
        except Exception as error:
            raise BaselineError(
                f"Could not read processed fold data {self._processed_path}: {error}"
            ) from error

        self._validate_reference(data)
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
        fingerprint = _sha256(self._processed_path)
        return BaselineData(
            reference=data.loc[:, ["id", "label", "fold", "is_swapped"]]
            .copy()
            .reset_index(drop=True),
            canonical_texts=canonical_texts,
            fingerprint=fingerprint,
        )

    def _validate_reference(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise BaselineError("Processed fold data contains no rows.")
        if data.loc[:, list(self.REQUIRED_COLUMNS)].isna().any().any():
            raise BaselineError("Processed fold data contains missing required values.")
        if not data["label"].isin((0, 1, 2)).all():
            raise BaselineError("Processed labels must be 0, 1, or 2.")
        if not data["is_swapped"].isin((True, False)).all():
            raise BaselineError("Processed is_swapped values must be boolean.")

        expected_folds = set(range(self._n_splits))
        actual_folds = {int(value) for value in data["fold"].unique()}
        if actual_folds != expected_folds:
            raise BaselineError(
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
            raise BaselineError(
                "Each ID must have one original and one swapped row in one fold."
            )

        originals = data.loc[~data["is_swapped"], ["id", "label"]].set_index("id")
        swapped = data.loc[data["is_swapped"], ["id", "label"]].set_index("id")
        swapped = swapped.loc[originals.index]
        translated = swapped["label"].map(self.SWAPPED_LABELS)
        if not translated.astype("int64").eq(originals["label"].astype("int64")).all():
            raise BaselineError("Swapped labels do not match their canonical labels.")


class BaselineTestDataRepository:
    """Read and validate competition test and submission-template records."""

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
        self._serializer = ConversationSerializer()

    @property
    def serializer_version(self) -> str:
        """Return the cache-relevant conversation format identifier."""
        return self._serializer.FORMAT_VERSION

    def load(self) -> BaselineTestData:
        """Return validated test texts in sample-submission order."""
        test = self._read_csv(self._test_path, "competition test data")
        submission = self._read_csv(
            self._submission_path,
            "sample submission",
        )
        self._validate_columns(test, self.TEST_COLUMNS, self._test_path)
        self._validate_columns(
            submission,
            self.SUBMISSION_COLUMNS,
            self._submission_path,
        )
        if test.loc[:, list(self.TEST_COLUMNS)].isna().any().any():
            raise BaselineError("Competition test data contains missing values.")

        test_ids = self._validated_ids(test, self._test_path)
        submission_ids = self._validated_ids(submission, self._submission_path)
        if not np.array_equal(test_ids, submission_ids):
            raise BaselineError(
                "test.csv and sample_submission.csv IDs must match in the same order."
            )

        canonical_texts = tuple(
            self._serializer.serialize_pair(
                row_id=int(row.id),
                prompt_value=row.prompt,
                response_a_value=row.response_a,
                response_b_value=row.response_b,
            )
            for row in test.itertuples(index=False)
        )
        return BaselineTestData(
            ids=tuple(int(row_id) for row_id in test_ids),
            canonical_texts=canonical_texts,
            fingerprint=_sha256(self._test_path),
        )

    @staticmethod
    def _read_csv(path: Path, description: str) -> pd.DataFrame:
        if not path.is_file():
            raise BaselineError(
                f"{description.capitalize()} not found: {path}. Run 'uv run data' first."
            )
        try:
            return pd.read_csv(path)
        except Exception as error:
            raise BaselineError(
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
            raise BaselineError(
                f"{path.name} must contain exactly these columns in order: {columns}."
            )
        if data.empty:
            raise BaselineError(f"{path.name} contains no rows.")

    @staticmethod
    def _validated_ids(data: pd.DataFrame, path: Path) -> np.ndarray:
        try:
            numeric_ids = pd.to_numeric(data["id"], errors="raise").to_numpy(
                dtype=np.float64
            )
        except (TypeError, ValueError) as error:
            raise BaselineError(f"{path.name} IDs must be numeric integers.") from error
        if (
            not np.isfinite(numeric_ids).all()
            or not np.equal(numeric_ids, np.floor(numeric_ids)).all()
        ):
            raise BaselineError(f"{path.name} IDs must be finite integers.")
        integer_ids = numeric_ids.astype(np.int64)
        if pd.Series(integer_ids).duplicated().any():
            raise BaselineError(f"{path.name} IDs must be unique.")
        return integer_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BaselineError(f"Could not fingerprint {path}: {error}") from error
    return digest.hexdigest()
