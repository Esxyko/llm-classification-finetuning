"""Validate and synthesize fixed-holdout prediction files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import ModelExecutionError, ResultSynthesisError
from ..models.common import TrainingInputLoader
from .publisher import ResultPublisher


@dataclass(frozen=True, slots=True)
class ReducedResults:
	"""Canonical records and their two-vote confusion matrix."""

	records: pd.DataFrame
	confusion_matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class ResultSynthesisResult:
	"""Summary of a completed result synthesis."""

	source_dir: Path
	workbook_path: Path
	confusion_matrix_path: Path
	predictions: int
	records: int
	incorrects: int
	average_loss: float


class AugmentationReducer:
	"""Reduce orientation pairs into canonical source records."""

	PROBABILITY_COLUMNS = (
		"winner_model_a",
		"winner_model_b",
		"winner_tie",
	)
	SWAPPED_PROBABILITY_ORDER = (1, 0, 2)
	AVERAGED_AGGREGATION = "ab_swap_average"

	def reduce(
		self,
		predictions: pd.DataFrame,
		*,
		prediction_aggregation: str | None = None,
	) -> ReducedResults:
		"""Return one record per ID using the run's aggregation policy."""
		original = predictions.loc[~predictions["is_swapped"]].copy()
		swapped = predictions.loc[predictions["is_swapped"]].copy()
		if original.empty:
			raise ResultSynthesisError("The processed data contains no source records.")
		if original["id"].duplicated().any() or swapped["id"].duplicated().any():
			raise ResultSynthesisError(
				"Each ID must have exactly one original and one swapped prediction."
			)

		original_ids = original["id"].tolist()
		swapped = swapped.set_index("id", drop=False)
		missing_swapped_ids = [
			row_id for row_id in original_ids if row_id not in swapped.index
		]
		if missing_swapped_ids or len(swapped) != len(original):
			raise ResultSynthesisError(
				"Each ID must have exactly one original and one swapped prediction."
			)
		swapped = swapped.loc[original_ids]

		expected = original["label"].to_numpy(dtype=np.int64)
		swapped_expected = swapped["label"].to_numpy(dtype=np.int64)
		translated_swapped_expected = np.take(
			np.array((1, 0, 2), dtype=np.int64),
			swapped_expected,
		)
		if not np.array_equal(expected, translated_swapped_expected):
			raise ResultSynthesisError(
				"Swapped labels do not translate back to their source labels."
			)

		original_probabilities = original.loc[
			:, list(self.PROBABILITY_COLUMNS)
		].to_numpy(dtype=np.float64)
		swapped_probabilities = swapped.loc[:, list(self.PROBABILITY_COLUMNS)].to_numpy(
			dtype=np.float64
		)
		canonical_swapped_probabilities = swapped_probabilities[
			:, self.SWAPPED_PROBABILITY_ORDER
		]

		if prediction_aggregation == self.AVERAGED_AGGREGATION:
			if not np.allclose(
				original_probabilities,
				canonical_swapped_probabilities,
				rtol=0.0,
				atol=1e-6,
			):
				raise ResultSynthesisError(
					"Averaged original and swapped predictions do not match."
				)
			actual = np.argmax(original_probabilities, axis=1)
			incorrects = (actual != expected).astype(np.int8)
			row_indices = np.arange(len(expected))
			expected_probability = np.clip(
				original_probabilities[row_indices, expected],
				np.finfo(np.float64).tiny,
				1.0,
			)
			records = pd.DataFrame(
				{
					"id": original["id"].to_numpy(),
					"split": original["split"].to_numpy(),
					"expected_labels": expected.astype(np.int8),
					"actual_labels": [str(label) for label in actual],
					"incorrects": incorrects,
					"contributed_loss": -np.log(expected_probability),
				}
			)
			confusion = np.zeros((3, 3), dtype=np.int64)
			np.add.at(confusion, (expected, actual), 1)
			return ReducedResults(records=records, confusion_matrix=confusion)
		if prediction_aggregation is not None:
			raise ResultSynthesisError(
				f"Unknown prediction aggregation: {prediction_aggregation}."
			)

		original_actual = np.argmax(original_probabilities, axis=1)
		swapped_actual = np.argmax(canonical_swapped_probabilities, axis=1)
		incorrects = (original_actual != expected).astype(np.int8)
		incorrects += (swapped_actual != expected).astype(np.int8)

		row_indices = np.arange(len(expected))
		epsilon = np.finfo(np.float64).eps
		original_expected_probability = np.clip(
			original_probabilities[row_indices, expected],
			epsilon,
			1.0 - epsilon,
		)
		swapped_expected_probability = np.clip(
			canonical_swapped_probabilities[row_indices, expected],
			epsilon,
			1.0 - epsilon,
		)
		contributed_loss = (
			-np.log(original_expected_probability)
			- np.log(swapped_expected_probability)
		) / 2.0

		actual_labels = [
			f"{int(original_label)},{int(swapped_label)}"
			for original_label, swapped_label in zip(
				original_actual,
				swapped_actual,
				strict=True,
			)
		]
		records = pd.DataFrame(
			{
				"id": original["id"].to_numpy(),
				"split": original["split"].to_numpy(),
				"expected_labels": expected.astype(np.int8),
				"actual_labels": actual_labels,
				"incorrects": incorrects,
				"contributed_loss": contributed_loss,
			}
		)

		confusion = np.zeros((3, 3), dtype=np.int64)
		paired_actual = np.column_stack((original_actual, swapped_actual))
		np.add.at(
			confusion,
			(np.repeat(expected, 2), paired_actual.reshape(-1)),
			1,
		)
		return ReducedResults(records=records, confusion_matrix=confusion)


class ResultSynthesizer:
	"""Coordinate result discovery, validation, reduction, and publishing."""

	OUTPUT_DIR_NAME = ResultPublisher.OUTPUT_DIR_NAME
	RESERVED_SOURCE_DIR_NAMES = (OUTPUT_DIR_NAME, "submit", "test")
	WORKBOOK_NAME = ResultPublisher.WORKBOOK_NAME
	CONFUSION_MATRIX_NAME = ResultPublisher.CONFUSION_MATRIX_NAME
	REQUIRED_COLUMNS = (
		"id",
		"winner_model_a",
		"winner_model_b",
		"winner_tie",
	)
	REFERENCE_COLUMNS = ("id", "label", "split", "is_swapped")
	PROBABILITY_TOLERANCE = 1e-6

	def __init__(
		self,
		processed_path: Path,
		results_root: Path,
		validation_fraction: float = 0.2,
	) -> None:
		self._processed_path = processed_path.resolve()
		self._validation_fraction = validation_fraction
		self._results_root = results_root.resolve()
		self._reducer = AugmentationReducer()
		self._publisher = ResultPublisher(self._results_root)

	def synthesize(self, subfolder: str | None = None) -> ResultSynthesisResult:
		"""Generate comprehensive artifacts for one completed holdout run."""
		try:
			source_dir = self._select_source_dir(subfolder)
			prediction_aggregation = self._read_prediction_aggregation(source_dir)
			reference = self._read_reference()
			predictions = self._read_validation_predictions(source_dir, reference)
			reduced = self._reducer.reduce(
				predictions,
				prediction_aggregation=prediction_aggregation,
			)
			workbook_path, matrix_path = self._publisher.publish(
				reduced.records, reduced.confusion_matrix
			)
		except ResultSynthesisError:
			raise
		except Exception as error:
			raise ResultSynthesisError(
				f"Could not synthesize validation results: {error}"
			) from error

		records = reduced.records
		return ResultSynthesisResult(
			source_dir=source_dir,
			workbook_path=workbook_path,
			confusion_matrix_path=matrix_path,
			predictions=(
				len(records)
				if prediction_aggregation == AugmentationReducer.AVERAGED_AGGREGATION
				else len(predictions)
			),
			records=len(records),
			incorrects=int(records["incorrects"].sum()),
			average_loss=float(records["contributed_loss"].mean()),
		)

	def _select_source_dir(self, subfolder: str | None) -> Path:
		if not self._results_root.is_dir():
			raise ResultSynthesisError(
				f"Results directory not found: {self._results_root}"
			)

		if subfolder is not None:
			candidate_name = subfolder.strip()
			candidate_path = Path(candidate_name)
			if (
				not candidate_name
				or candidate_path.is_absolute()
				or len(candidate_path.parts) != 1
				or candidate_path.name != candidate_name
				or candidate_name.casefold()
				in {name.casefold() for name in self.RESERVED_SOURCE_DIR_NAMES}
			):
				raise ResultSynthesisError(
					"SUBFOLDER must name a validation result directory and "
					"cannot be a reserved output directory."
				)
			source_dir = (self._results_root / candidate_name).resolve()
			if source_dir.parent != self._results_root or not source_dir.is_dir():
				raise ResultSynthesisError(f"Result subfolder not found: {source_dir}")
			return source_dir

		candidates = [
			path
			for path in self._results_root.iterdir()
			if path.is_dir()
			and path.name.casefold()
			not in {name.casefold() for name in self.RESERVED_SOURCE_DIR_NAMES}
			and not path.name.casefold().startswith(
				f".{self.OUTPUT_DIR_NAME.casefold()}-"
			)
		]
		if not candidates:
			raise ResultSynthesisError(
				f"No result subfolders found under {self._results_root}."
			)
		return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))

	def _read_reference(self) -> pd.DataFrame:
		"""Validate the processed artifact and retain only the holdout rows."""
		try:
			data = (
				TrainingInputLoader(
					self._processed_path, validation_fraction=self._validation_fraction
				)
				.load()
				.data
			)
		except ModelExecutionError as error:
			raise ResultSynthesisError(str(error)) from error
		return data.loc[
			data["split"].eq("validation"), list(self.REFERENCE_COLUMNS)
		].reset_index(drop=True)

	@staticmethod
	def _read_prediction_aggregation(source_dir: Path) -> str | None:
		metrics_path = source_dir / "metrics.json"
		if not metrics_path.is_file():
			raise ResultSynthesisError(
				"Holdout reports require metrics.json. Legacy five-fold results are not supported; run the model again with the new split."
			)
		try:
			metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
		except (OSError, UnicodeError, json.JSONDecodeError) as error:
			raise ResultSynthesisError(
				f"Could not read result metadata {metrics_path}: {error}"
			) from error
		if not isinstance(metrics, dict):
			raise ResultSynthesisError("Result metadata must be a JSON object.")
		if metrics.get("validation_strategy") != "holdout":
			raise ResultSynthesisError(
				"Legacy five-fold results are not supported. Preserve the old run and generate a new holdout run with the selected model."
			)
		aggregation = metrics.get("prediction_aggregation")
		if aggregation is not None and not isinstance(aggregation, str):
			raise ResultSynthesisError(
				"Result prediction aggregation must be a string."
			)
		return aggregation

	def _read_validation_predictions(
		self, source_dir: Path, reference: pd.DataFrame
	) -> pd.DataFrame:
		"""Match exactly one validation.csv to the saved validation orientation order."""
		csv_paths = [path for path in source_dir.glob("*.csv") if path.is_file()]
		if len(csv_paths) != 1 or csv_paths[0].name != "validation.csv":
			raise ResultSynthesisError(
				f"Expected only validation.csv in {source_dir}. Legacy fold CSVs are not supported; generate a new holdout run."
			)
		csv_data = self._read_prediction_csv(csv_paths[0])
		if (
			not csv_data["id"]
			.reset_index(drop=True)
			.equals(reference["id"].reset_index(drop=True))
		):
			raise ResultSynthesisError(
				"validation.csv ID order does not exactly match the configured validation set; use the processed data for this run."
			)
		predictions = reference.copy()
		for column in self.REQUIRED_COLUMNS[1:]:
			predictions[column] = csv_data[column].to_numpy(dtype=np.float64)
		return predictions

	def _read_prediction_csv(self, csv_path: Path) -> pd.DataFrame:
		try:
			data = pd.read_csv(csv_path)
		except Exception as error:
			raise ResultSynthesisError(
				f"Could not read prediction CSV {csv_path}: {error}"
			) from error

		if tuple(data.columns) != self.REQUIRED_COLUMNS:
			expected = ", ".join(self.REQUIRED_COLUMNS)
			raise ResultSynthesisError(
				f"{csv_path.name} must contain exactly these columns in order: "
				f"{expected}."
			)
		if data.empty:
			raise ResultSynthesisError(f"{csv_path.name} contains no predictions.")

		try:
			numeric_ids = pd.to_numeric(data["id"], errors="raise")
			probabilities = data.loc[:, list(self.REQUIRED_COLUMNS[1:])].to_numpy(
				dtype=np.float64
			)
		except (TypeError, ValueError) as error:
			raise ResultSynthesisError(
				f"{csv_path.name} contains non-numeric IDs or probabilities."
			) from error

		numeric_id_values = numeric_ids.to_numpy(dtype=np.float64)
		if (
			not np.isfinite(numeric_id_values).all()
			or not np.equal(numeric_id_values, np.floor(numeric_id_values)).all()
		):
			raise ResultSynthesisError(f"{csv_path.name} IDs must be finite integers.")
		if not np.isfinite(probabilities).all():
			raise ResultSynthesisError(
				f"{csv_path.name} probabilities must all be finite."
			)
		if ((probabilities < 0.0) | (probabilities > 1.0)).any():
			raise ResultSynthesisError(
				f"{csv_path.name} probabilities must be between 0 and 1."
			)
		if not np.allclose(
			probabilities.sum(axis=1),
			1.0,
			rtol=0.0,
			atol=self.PROBABILITY_TOLERANCE,
		):
			raise ResultSynthesisError(
				f"{csv_path.name} probability rows must sum to 1 within "
				f"{self.PROBABILITY_TOLERANCE:g}."
			)

		data = data.copy()
		data["id"] = numeric_ids.astype("int64")
		return data
