"""Publish model validation runs and test submissions."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ...errors import ModelExecutionError
from .trainer import FoldTrainingResult


class ModelResultPublisher:
    """Atomically publish fold predictions under a datetime run name."""

    PROBABILITY_COLUMNS = (
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    )

    def __init__(self, results_root: Path, result_prefix: str) -> None:
        self._results_root = results_root.resolve()
        self._result_prefix = result_prefix

    def publish(
        self,
        fold_results: tuple[FoldTrainingResult, ...],
        model_name: str,
        *,
        prediction_aggregation: str | None = None,
    ) -> tuple[Path, float, int]:
        """Stage every artifact, then expose one complete run directory."""
        try:
            self._results_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ModelExecutionError(
                f"Could not create results directory {self._results_root}: {error}"
            ) from error

        generated_at = datetime.now(UTC).astimezone()
        run_name = generated_at.strftime(f"{self._result_prefix}-%Y%m%d-%H%M%S")
        output_dir = self._results_root / run_name
        if output_dir.exists():
            raise ModelExecutionError(
                f"Model result directory already exists for this second: {output_dir}"
            )

        if prediction_aggregation == "ab_swap_average" and any(
            len(result.ids) % 2 for result in fold_results
        ):
            raise ModelExecutionError(
                "Averaged fold predictions must contain complete orientation pairs."
            )
        prediction_divisor = 2 if prediction_aggregation == "ab_swap_average" else 1
        prediction_count = sum(
            len(result.ids) // prediction_divisor for result in fold_results
        )
        if prediction_count == 0:
            raise ModelExecutionError(
                "Model training produced no validation predictions."
            )
        weighted_loss = sum(
            result.final_loss * (len(result.ids) // prediction_divisor)
            for result in fold_results
        )
        average_loss = weighted_loss / prediction_count
        metrics = {
            "run_name": run_name,
            "generated_at": generated_at.isoformat(timespec="seconds"),
            "model_name": model_name,
            "folds": len(fold_results),
            "predictions": prediction_count,
            "average_out_of_fold_loss": average_loss,
            "fold_metrics": [
                {
                    "fold": result.fold,
                    "predictions": len(result.ids) // prediction_divisor,
                    "final_validation_loss": result.final_loss,
                    "epochs": [asdict(epoch) for epoch in result.epochs],
                }
                for result in fold_results
            ],
        }
        if prediction_aggregation is not None:
            metrics["prediction_aggregation"] = prediction_aggregation

        try:
            with tempfile.TemporaryDirectory(
                dir=self._results_root,
                prefix=f".{self._result_prefix}-",
            ) as temporary_dir:
                staging_dir = Path(temporary_dir)
                for result in fold_results:
                    prediction_frame = pd.DataFrame(
                        {
                            "id": result.ids,
                            **{
                                column: result.probabilities[:, index]
                                for index, column in enumerate(self.PROBABILITY_COLUMNS)
                            },
                        }
                    )
                    self._validate_probabilities(prediction_frame, result.fold)
                    prediction_frame.to_csv(
                        staging_dir / f"fold-{result.fold}.csv",
                        index=False,
                    )
                (staging_dir / "metrics.json").write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(staging_dir, output_dir)
        except ModelExecutionError:
            raise
        except Exception as error:
            raise ModelExecutionError(
                f"Could not publish model results to {output_dir}: {error}"
            ) from error
        return output_dir, average_loss, prediction_count

    def _validate_probabilities(self, predictions: pd.DataFrame, fold: int) -> None:
        values = predictions.loc[:, list(self.PROBABILITY_COLUMNS)].to_numpy(
            dtype=np.float64
        )
        if not _probabilities_are_valid(values):
            raise ModelExecutionError(f"Fold {fold} produced invalid probabilities.")


class ModelSubmissionPublisher:
    """Atomically publish one timestamped competition submission."""

    PROBABILITY_COLUMNS = ModelResultPublisher.PROBABILITY_COLUMNS

    def __init__(self, results_root: Path, result_prefix: str) -> None:
        self._results_root = results_root.resolve()
        self._result_prefix = result_prefix

    def publish(
        self,
        ids: tuple[int, ...],
        probabilities: np.ndarray,
    ) -> tuple[Path, Path]:
        """Write a submission CSV in a newly exposed test-run directory."""
        if probabilities.shape != (len(ids), len(self.PROBABILITY_COLUMNS)):
            raise ModelExecutionError(
                "Test probabilities do not match the test ID count."
            )
        if not _probabilities_are_valid(probabilities):
            raise ModelExecutionError(
                "Model test inference produced invalid probabilities."
            )

        generated_at = datetime.now(UTC).astimezone()
        output_dir = self._results_root / generated_at.strftime(
            f"{self._result_prefix}-%Y%m%d-%H%M%S"
        )
        submission_path = output_dir / "submission.csv"
        try:
            self._results_root.mkdir(parents=True, exist_ok=True)
            if output_dir.exists():
                raise ModelExecutionError(
                    "Model test result directory already exists for this second: "
                    f"{output_dir}"
                )
            with tempfile.TemporaryDirectory(
                dir=self._results_root,
                prefix=f".{self._result_prefix}-",
            ) as temporary_dir:
                staging_dir = Path(temporary_dir)
                pd.DataFrame(
                    {
                        "id": ids,
                        **{
                            column: probabilities[:, index]
                            for index, column in enumerate(self.PROBABILITY_COLUMNS)
                        },
                    }
                ).to_csv(staging_dir / "submission.csv", index=False)
                os.replace(staging_dir, output_dir)
        except ModelExecutionError:
            raise
        except Exception as error:
            raise ModelExecutionError(
                f"Could not publish model test results to {output_dir}: {error}"
            ) from error
        return output_dir, submission_path


def _probabilities_are_valid(values: np.ndarray) -> bool:
    return bool(
        values.size > 0
        and np.isfinite(values).all()
        and not (values < 0.0).any()
        and not (values > 1.0).any()
        and np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    )
