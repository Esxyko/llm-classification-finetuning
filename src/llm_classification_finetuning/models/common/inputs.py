"""Load and validate source tables shared by model input formats."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...config import MAX_VALIDATION_ROW_DEVIATION
from ...errors import ModelExecutionError
from .cache_storage import CacheCorruptionError, CacheStorage


@dataclass(frozen=True, slots=True)
class TrainingInputs:
	data: pd.DataFrame
	fingerprint: str


class TrainingInputLoader:
	"""Validate original rows and holdout membership before serialization."""

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

	def __init__(
		self, processed_path: Path, *, validation_fraction: float | None = None
	) -> None:
		self._processed_path = processed_path.resolve()
		self._validation_fraction = validation_fraction

	def load(self) -> TrainingInputs:
		storage = CacheStorage(self._processed_path)
		if not storage.exists():
			raise ModelExecutionError(
				f"Processed split data not found: {self._processed_path}. "
				"Run 'uv run preprocess' with the selected --config first."
			)
		try:
			from pyarrow.parquet import read_schema

			with storage.open("r") as session:
				columns = read_schema(session.path).names
				if "fold" in columns and "split" not in columns:
					raise ModelExecutionError(
						"Legacy fold data is no longer supported. Set data.processed_path "
						"to data/processed/train_validation.parquet and run 'uv run preprocess'."
					)
				data = pd.read_parquet(
					session.path, columns=list(self.REQUIRED_COLUMNS)
				)
				fingerprint = _sha256(session.path)
		except CacheCorruptionError as error:
			raise ModelExecutionError(
				f"Corrupt processed split data {self._processed_path}: {error}. "
				"Run 'uv run preprocess' with the selected --config to rebuild it."
			) from error
		except ModelExecutionError:
			raise
		except Exception as error:
			raise ModelExecutionError(
				f"Could not read processed split data {self._processed_path}: {error}"
			) from error
		self.validate(data, validation_fraction=self._validation_fraction)
		return TrainingInputs(data=data, fingerprint=fingerprint)

	@staticmethod
	def validate(
		data: pd.DataFrame, *, validation_fraction: float | None = None
	) -> None:
		"""Require one original row per ID and nonoverlapping prompt groups."""
		if data.empty:
			raise ModelExecutionError("Processed split data contains no rows.")
		if data.loc[:, list(TrainingInputLoader.REQUIRED_COLUMNS)].isna().any().any():
			raise ModelExecutionError(
				"Processed split data contains missing required values."
			)
		if not data["label"].isin((0, 1, 2)).all():
			raise ModelExecutionError("Processed labels must be 0, 1, or 2.")
		if not pd.api.types.is_bool_dtype(data["is_swapped"]):
			raise ModelExecutionError("Processed is_swapped values must be boolean.")
		if data["is_swapped"].any():
			raise ModelExecutionError(
				"Processed split data must contain original rows only. "
				"Remove rows with is_swapped=True before loading it."
			)
		if data["id"].duplicated().any():
			raise ModelExecutionError("Each ID must have exactly one original row.")
		if set(data["split"].unique()) != {"train", "validation"}:
			raise ModelExecutionError(
				"Processed split data must contain nonempty 'train' and 'validation' partitions."
			)
		if data.groupby("group_id", sort=False)["split"].nunique().ne(1).any():
			raise ModelExecutionError(
				"A prompt group must not cross training and validation sets."
			)

		if validation_fraction is not None:
			share = float(data["split"].eq("validation").mean())
			if abs(share / validation_fraction - 1) >= MAX_VALIDATION_ROW_DEVIATION:
				raise ModelExecutionError(
					"The saved validation row fraction does not match validation.validation_fraction. "
					"Run 'uv run preprocess' with the selected --config."
				)


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	try:
		with path.open("rb") as source:
			for chunk in iter(lambda: source.read(1024 * 1024), b""):
				digest.update(chunk)
	except OSError as error:
		raise ModelExecutionError(f"Could not fingerprint {path}: {error}") from error
	return digest.hexdigest()
