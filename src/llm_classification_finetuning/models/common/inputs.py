"""Load and validate source tables shared by model input formats."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import MAX_VALIDATION_ROW_DEVIATION
from ...errors import ModelExecutionError


@dataclass(frozen=True, slots=True)
class TrainingInputs:
	data: pd.DataFrame
	fingerprint: str


@dataclass(frozen=True, slots=True)
class TestInputs:
	data: pd.DataFrame
	ids: tuple[int, ...]
	fingerprint: str


class TrainingInputLoader:
	"""Validate holdout membership and orientation pairs before serialization."""

	REQUIRED_COLUMNS = (
		"id",
		"prompt",
		"response_a",
		"response_b",
		"label",
		"split",
		"group_id",
		"is_swapped",
	)
	SWAPPED_LABELS = {0: 1, 1: 0, 2: 2}

	def __init__(
		self, processed_path: Path, *, validation_fraction: float | None = None
	) -> None:
		self._processed_path = processed_path.resolve()
		self._validation_fraction = validation_fraction

	def load(self) -> TrainingInputs:
		if not self._processed_path.is_file():
			raise ModelExecutionError(
				f"Processed split data not found: {self._processed_path}. "
				"Run 'uv run preprocess' with the selected --config first."
			)
		try:
			from pyarrow.parquet import read_schema

			columns = read_schema(self._processed_path).names
			if "fold" in columns and "split" not in columns:
				raise ModelExecutionError(
					"Legacy fold data is no longer supported. Set data.processed_path "
					"to data/processed/train_validation.parquet and run 'uv run preprocess'."
				)
			data = pd.read_parquet(
				self._processed_path, columns=list(self.REQUIRED_COLUMNS)
			)
		except ModelExecutionError:
			raise
		except Exception as error:
			raise ModelExecutionError(
				f"Could not read processed split data {self._processed_path}: {error}"
			) from error
		self.validate(data, validation_fraction=self._validation_fraction)
		return TrainingInputs(data=data, fingerprint=_sha256(self._processed_path))

	@staticmethod
	def validate(
		data: pd.DataFrame, *, validation_fraction: float | None = None
	) -> None:
		"""Require complete, nonoverlapping train/validation orientation pairs."""
		if data.empty:
			raise ModelExecutionError("Processed split data contains no rows.")
		if data.loc[:, list(TrainingInputLoader.REQUIRED_COLUMNS)].isna().any().any():
			raise ModelExecutionError(
				"Processed split data contains missing required values."
			)
		if not data["label"].isin((0, 1, 2)).all():
			raise ModelExecutionError("Processed labels must be 0, 1, or 2.")
		if not data["is_swapped"].isin((True, False)).all():
			raise ModelExecutionError("Processed is_swapped values must be boolean.")
		if set(data["split"].unique()) != {"train", "validation"}:
			raise ModelExecutionError(
				"Processed split data must contain nonempty 'train' and 'validation' partitions."
			)
		if data.groupby("group_id", sort=False)["split"].nunique().ne(1).any():
			raise ModelExecutionError(
				"A prompt group must not cross training and validation sets."
			)

		pairs = data.groupby("id", sort=False).agg(
			rows=("id", "size"),
			splits=("split", "nunique"),
			groups=("group_id", "nunique"),
			originals=("is_swapped", lambda values: int((~values).sum())),
			swapped=("is_swapped", lambda values: int(values.sum())),
		)
		if (
			pairs["rows"].ne(2)
			| pairs["splits"].ne(1)
			| pairs["groups"].ne(1)
			| pairs["originals"].ne(1)
			| pairs["swapped"].ne(1)
		).any():
			raise ModelExecutionError(
				"Each ID must have one original and one swapped row in one split and prompt group."
			)
		originals = data.loc[~data["is_swapped"], ["id", "label"]].set_index("id")
		if validation_fraction is not None:
			original_rows = data.loc[~data["is_swapped"]]
			share = float(original_rows["split"].eq("validation").mean())
			if abs(share / validation_fraction - 1) >= MAX_VALIDATION_ROW_DEVIATION:
				raise ModelExecutionError(
					"The saved validation row fraction does not match validation.validation_fraction. "
					"Run 'uv run preprocess' with the selected --config."
				)
		swapped = data.loc[data["is_swapped"], ["id", "label"]].set_index("id")
		swapped = swapped.loc[originals.index]
		translated = swapped["label"].map(TrainingInputLoader.SWAPPED_LABELS)
		if not translated.astype("int64").eq(originals["label"].astype("int64")).all():
			raise ModelExecutionError(
				"Swapped labels do not match their canonical labels."
			)


class TestInputLoader:
	"""Validate test rows against the exact submission template order."""

	TEST_COLUMNS = ("id", "prompt", "response_a", "response_b")
	SUBMISSION_COLUMNS = ("id", "winner_model_a", "winner_model_b", "winner_tie")

	def __init__(self, raw_dir: Path) -> None:
		self._test_path = raw_dir.resolve() / "test.csv"
		self._submission_path = raw_dir.resolve() / "sample_submission.csv"

	def load(self) -> TestInputs:
		test = self._read_csv(self._test_path, "competition test data")
		submission = self._read_csv(self._submission_path, "sample submission")
		self._validate_columns(test, self.TEST_COLUMNS, self._test_path)
		self._validate_columns(
			submission, self.SUBMISSION_COLUMNS, self._submission_path
		)
		if test.loc[:, list(self.TEST_COLUMNS)].isna().any().any():
			raise ModelExecutionError("Competition test data contains missing values.")
		test_ids = self._validated_ids(test, self._test_path)
		submission_ids = self._validated_ids(submission, self._submission_path)
		if not np.array_equal(test_ids, submission_ids):
			raise ModelExecutionError(
				"test.csv and sample_submission.csv IDs must match in the same order."
			)
		return TestInputs(
			data=test,
			ids=tuple(int(row_id) for row_id in test_ids),
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
		data: pd.DataFrame, expected: tuple[str, ...], path: Path
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
