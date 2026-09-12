"""Tests for k-fold result synthesis."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

# Keep matplotlib's font cache inside the test process's writable temp area.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd

from llm_classification_finetuning.errors import ResultSynthesisError
from llm_classification_finetuning.result.synthesizer import (
    AugmentationReducer,
    ResultSynthesizer,
)


class AugmentationReducerTests(unittest.TestCase):
    """Verify canonical label translation and pair metrics."""

    def test_reduces_predictions_into_canonical_pairs(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "id": 10,
                    "label": 0,
                    "fold": 0,
                    "is_swapped": False,
                    "winner_model_a": 0.8,
                    "winner_model_b": 0.1,
                    "winner_tie": 0.1,
                },
                {
                    "id": 11,
                    "label": 0,
                    "fold": 0,
                    "is_swapped": False,
                    "winner_model_a": 0.1,
                    "winner_model_b": 0.8,
                    "winner_tie": 0.1,
                },
                {
                    "id": 10,
                    "label": 1,
                    "fold": 0,
                    "is_swapped": True,
                    "winner_model_a": 0.1,
                    "winner_model_b": 0.8,
                    "winner_tie": 0.1,
                },
                {
                    "id": 11,
                    "label": 1,
                    "fold": 0,
                    "is_swapped": True,
                    "winner_model_a": 0.1,
                    "winner_model_b": 0.8,
                    "winner_tie": 0.1,
                },
            ]
        )

        reduced = AugmentationReducer().reduce(predictions)

        self.assertEqual(reduced.records["expected_labels"].tolist(), [0, 0])
        self.assertEqual(reduced.records["actual_labels"].tolist(), ["0,0", "1,0"])
        self.assertEqual(reduced.records["incorrects"].tolist(), [0, 1])
        self.assertAlmostEqual(
            float(reduced.records.loc[0, "contributed_loss"]),
            -math.log(0.8),
        )
        self.assertAlmostEqual(
            float(reduced.records.loc[1, "contributed_loss"]),
            (-math.log(0.1) - math.log(0.8)) / 2,
        )
        np.testing.assert_array_equal(
            reduced.confusion_matrix,
            np.array([[3, 1, 0], [0, 0, 0], [0, 0, 0]]),
        )


class ResultSynthesizerTests(unittest.TestCase):
    """Verify discovery, validation, artifact generation, and publishing."""

    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.results_root = self.root / "results"
        self.results_root.mkdir()
        self.processed_path = self.root / "train_folds.parquet"
        self.reference = self._build_reference()
        self.reference.to_parquet(self.processed_path, index=False)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_uses_latest_run_and_publishes_complete_artifacts(self) -> None:
        older = self._write_run("older")
        newer = self._write_run("newer")
        os.utime(older, (1_700_000_000, 1_700_000_000))
        os.utime(newer, (1_700_000_100, 1_700_000_100))

        output_dir = self.results_root / "comprehensive"
        output_dir.mkdir()
        (output_dir / "stale.txt").write_text("stale", encoding="utf-8")

        automatic = self._synthesizer().synthesize()

        self.assertEqual(automatic.source_dir, newer.resolve())
        self.assertEqual(automatic.folds, 2)
        self.assertEqual(automatic.predictions, 8)
        self.assertEqual(automatic.records, 4)
        self.assertEqual(automatic.incorrects, 1)
        self.assertEqual(
            {path.name for path in output_dir.iterdir()},
            {"records.xlsx", "confusion_matrix.png"},
        )
        self.assertTrue(
            automatic.confusion_matrix_path.read_bytes().startswith(
                b"\x89PNG\r\n\x1a\n"
            )
        )
        with zipfile.ZipFile(automatic.workbook_path) as workbook:
            worksheet_xml = workbook.read("xl/worksheets/sheet1.xml").decode()
            table_xml = workbook.read("xl/tables/table1.xml").decode()
        self.assertIn('ySplit="1"', worksheet_xml)
        self.assertIn(
            '<f>IFERROR(SUBTOTAL(101,F2:F5),"")</f>',
            worksheet_xml,
        )
        self.assertIn('ref="A1:F5"', table_xml)

        explicit = self._synthesizer().synthesize("older")
        self.assertEqual(explicit.source_dir, older.resolve())
        self.assertEqual(explicit.records, automatic.records)
        self.assertAlmostEqual(explicit.average_loss, automatic.average_loss)

    def test_rejects_reordered_ids_without_clearing_previous_output(self) -> None:
        bad_run = self._write_run("bad-order", reorder_fold_zero=True)
        output_dir = self.results_root / "comprehensive"
        output_dir.mkdir()
        sentinel = output_dir / "previous.txt"
        sentinel.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(ResultSynthesisError, "ID order"):
            self._synthesizer().synthesize(bad_run.name)

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_rejects_invalid_probabilities(self) -> None:
        bad_run = self._write_run("bad-probability")
        fold_zero_path = bad_run / "anything-zero.csv"
        fold_zero = pd.read_csv(fold_zero_path)
        fold_zero.loc[0, "winner_model_a"] = 1.1
        fold_zero.to_csv(fold_zero_path, index=False)

        with self.assertRaisesRegex(ResultSynthesisError, "between 0 and 1"):
            self._synthesizer().synthesize(bad_run.name)

    def _synthesizer(self) -> ResultSynthesizer:
        return ResultSynthesizer(
            processed_path=self.processed_path,
            n_splits=2,
            results_root=self.results_root,
        )

    @staticmethod
    def _build_reference() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": [10, 11, 20, 21, 10, 11, 20, 21],
                "label": [0, 0, 1, 2, 1, 1, 0, 2],
                "fold": [0, 0, 1, 1, 0, 0, 1, 1],
                "is_swapped": [False] * 4 + [True] * 4,
            }
        )

    def _write_run(
        self,
        name: str,
        *,
        reorder_fold_zero: bool = False,
    ) -> Path:
        run_dir = self.results_root / name
        run_dir.mkdir()
        fold_zero = pd.DataFrame(
            {
                "id": [10, 11, 10, 11],
                "winner_model_a": [0.8, 0.1, 0.1, 0.1],
                "winner_model_b": [0.1, 0.8, 0.8, 0.8],
                "winner_tie": [0.1, 0.1, 0.1, 0.1],
            }
        )
        if reorder_fold_zero:
            fold_zero = fold_zero.iloc[[1, 0, 3, 2]].reset_index(drop=True)
        fold_one = pd.DataFrame(
            {
                "id": [20, 21, 20, 21],
                "winner_model_a": [0.1, 0.1, 0.8, 0.1],
                "winner_model_b": [0.8, 0.1, 0.1, 0.1],
                "winner_tie": [0.1, 0.8, 0.1, 0.8],
            }
        )
        fold_zero.to_csv(run_dir / "anything-zero.csv", index=False)
        fold_one.to_csv(run_dir / "anything-one.csv", index=False)
        return run_dir


if __name__ == "__main__":
    unittest.main()
