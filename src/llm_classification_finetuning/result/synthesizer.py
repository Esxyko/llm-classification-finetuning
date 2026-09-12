"""Validate and synthesize k-fold prediction files."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import ResultSynthesisError
from .writers import ConfusionMatrixWriter, RecordsWorkbookWriter


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
    folds: int
    predictions: int
    records: int
    incorrects: int
    average_loss: float


class AugmentationReducer:
    """Reduce original/swapped predictions into canonical source records."""

    PROBABILITY_COLUMNS = (
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    )
    SWAPPED_PROBABILITY_ORDER = (1, 0, 2)

    def reduce(self, predictions: pd.DataFrame) -> ReducedResults:
        """Return one record per ID while retaining both canonical predictions."""
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
                "fold": original["fold"].to_numpy(dtype=np.int16),
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

    OUTPUT_DIR_NAME = "comprehensive"
    RESERVED_SOURCE_DIR_NAMES = (OUTPUT_DIR_NAME, "test")
    WORKBOOK_NAME = "records.xlsx"
    CONFUSION_MATRIX_NAME = "confusion_matrix.png"
    REQUIRED_COLUMNS = (
        "id",
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    )
    REFERENCE_COLUMNS = ("id", "label", "fold", "is_swapped")
    PROBABILITY_TOLERANCE = 1e-6

    def __init__(
        self,
        processed_path: Path,
        n_splits: int,
        results_root: Path,
    ) -> None:
        self._processed_path = processed_path.resolve()
        self._n_splits = n_splits
        self._results_root = results_root.resolve()
        self._reducer = AugmentationReducer()
        self._matrix_writer = ConfusionMatrixWriter()
        self._workbook_writer = RecordsWorkbookWriter()

    def synthesize(self, subfolder: str | None = None) -> ResultSynthesisResult:
        """Generate comprehensive artifacts for one cross-validation run."""
        try:
            source_dir = self._select_source_dir(subfolder)
            reference = self._read_reference()
            predictions, fold_count = self._read_fold_predictions(
                source_dir,
                reference,
            )
            reduced = self._reducer.reduce(predictions)
            workbook_path, matrix_path = self._stage_and_publish(reduced)
        except ResultSynthesisError:
            raise
        except Exception as error:
            raise ResultSynthesisError(
                f"Could not synthesize fold results: {error}"
            ) from error

        records = reduced.records
        return ResultSynthesisResult(
            source_dir=source_dir,
            workbook_path=workbook_path,
            confusion_matrix_path=matrix_path,
            folds=fold_count,
            predictions=len(predictions),
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
                    "SUBFOLDER must name a cross-validation result directory and "
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
        if not self._processed_path.is_file():
            raise ResultSynthesisError(
                f"Processed fold data not found: {self._processed_path}"
            )
        try:
            reference = pd.read_parquet(
                self._processed_path,
                columns=list(self.REFERENCE_COLUMNS),
            )
        except Exception as error:
            raise ResultSynthesisError(
                f"Could not read processed fold data {self._processed_path}: {error}"
            ) from error

        if (
            reference.empty
            or reference.loc[:, list(self.REFERENCE_COLUMNS)].isna().any().any()
        ):
            raise ResultSynthesisError(
                "Processed fold data must contain complete validation records."
            )
        if not reference["label"].isin((0, 1, 2)).all():
            raise ResultSynthesisError("Processed labels must be 0, 1, or 2.")

        expected_folds = set(range(self._n_splits))
        actual_folds = set(int(value) for value in reference["fold"].unique())
        if actual_folds != expected_folds:
            raise ResultSynthesisError(
                "Processed fold data does not contain exactly the configured folds "
                f"0 through {self._n_splits - 1}."
            )

        pair_summary = reference.groupby("id", sort=False).agg(
            rows=("id", "size"),
            folds=("fold", "nunique"),
            originals=("is_swapped", lambda values: int((~values).sum())),
            swapped=("is_swapped", lambda values: int(values.sum())),
        )
        invalid_pairs = pair_summary.loc[
            pair_summary["rows"].ne(2)
            | pair_summary["folds"].ne(1)
            | pair_summary["originals"].ne(1)
            | pair_summary["swapped"].ne(1)
        ]
        if not invalid_pairs.empty:
            raise ResultSynthesisError(
                "Processed fold data must contain one original and one swapped row "
                "per ID in the same fold."
            )

        return reference.reset_index(drop=True)

    def _read_fold_predictions(
        self,
        source_dir: Path,
        reference: pd.DataFrame,
    ) -> tuple[pd.DataFrame, int]:
        csv_paths = sorted(
            (path for path in source_dir.glob("*.csv") if path.is_file()),
            key=lambda path: path.name.casefold(),
        )
        if len(csv_paths) != self._n_splits:
            raise ResultSynthesisError(
                f"Expected {self._n_splits} top-level CSV files in {source_dir}, "
                f"found {len(csv_paths)}."
            )

        fold_references: dict[int, pd.DataFrame] = {}
        for fold in range(self._n_splits):
            fold_data = reference.loc[reference["fold"].eq(fold)]
            if not fold_data.empty:
                fold_references[fold] = fold_data.copy()
        matched_folds: set[int] = set()
        prediction_frames: list[pd.DataFrame] = []

        for csv_path in csv_paths:
            csv_data = self._read_prediction_csv(csv_path)
            matching_folds = [
                fold
                for fold, fold_reference in fold_references.items()
                if csv_data["id"]
                .reset_index(drop=True)
                .equals(fold_reference["id"].reset_index(drop=True))
            ]
            if len(matching_folds) != 1:
                raise ResultSynthesisError(
                    f"{csv_path.name} ID order does not exactly match one configured "
                    "validation fold."
                )

            fold = matching_folds[0]
            if fold in matched_folds:
                raise ResultSynthesisError(
                    f"More than one CSV matches validation fold {fold}."
                )
            matched_folds.add(fold)

            fold_predictions = fold_references[fold].copy()
            for column in self.REQUIRED_COLUMNS[1:]:
                fold_predictions[column] = csv_data[column].to_numpy(dtype=np.float64)
            fold_predictions["_reference_order"] = fold_predictions.index
            prediction_frames.append(fold_predictions)

        expected_folds = set(range(self._n_splits))
        if matched_folds != expected_folds:
            missing = ", ".join(
                str(fold) for fold in sorted(expected_folds - matched_folds)
            )
            raise ResultSynthesisError(
                f"Missing prediction CSVs for fold(s): {missing}."
            )

        predictions = (
            pd.concat(prediction_frames, ignore_index=True)
            .sort_values("_reference_order", kind="stable")
            .drop(columns="_reference_order")
            .reset_index(drop=True)
        )
        return predictions, len(matched_folds)

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

    def _stage_and_publish(self, reduced: ReducedResults) -> tuple[Path, Path]:
        output_dir = self._results_root / self.OUTPUT_DIR_NAME
        if output_dir.is_symlink():
            raise ResultSynthesisError(
                f"Comprehensive output path cannot be a symlink: {output_dir}"
            )
        if output_dir.exists() and not output_dir.is_dir():
            raise ResultSynthesisError(
                f"Comprehensive output path is not a directory: {output_dir}"
            )

        with tempfile.TemporaryDirectory(
            dir=self._results_root,
            prefix=f".{self.OUTPUT_DIR_NAME}-",
        ) as temporary_dir:
            staging_dir = Path(temporary_dir)
            staged_workbook = staging_dir / self.WORKBOOK_NAME
            staged_matrix = staging_dir / self.CONFUSION_MATRIX_NAME
            try:
                self._workbook_writer.write(reduced.records, staged_workbook)
                self._matrix_writer.write(reduced.confusion_matrix, staged_matrix)
            except Exception as error:
                raise ResultSynthesisError(
                    f"Could not generate comprehensive artifacts: {error}"
                ) from error

            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                for child in output_dir.iterdir():
                    if child.is_symlink() or child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()

                workbook_path = output_dir / self.WORKBOOK_NAME
                matrix_path = output_dir / self.CONFUSION_MATRIX_NAME
                os.replace(staged_workbook, workbook_path)
                os.replace(staged_matrix, matrix_path)
            except OSError as error:
                raise ResultSynthesisError(
                    f"Could not publish comprehensive artifacts to {output_dir}: {error}"
                ) from error

        return workbook_path, matrix_path
