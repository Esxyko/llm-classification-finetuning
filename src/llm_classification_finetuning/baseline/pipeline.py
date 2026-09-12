"""Coordinate baseline cross-validation, model builds, and test inference."""

from __future__ import annotations

import gc
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import AppConfig
from ..errors import BaselineError
from .cache import CachedEmbeddings, EmbeddingCache, EmbeddingCacheKey
from .checkpoint import BaselineCheckpointStore, HeadCheckpointCompatibility
from .data import (
    BaselineData,
    BaselineDataRepository,
    BaselineTestDataRepository,
    PairwiseText,
)
from .extractor import QwenEmbeddingExtractor
from .hardware import GPUEnvironment
from .predictor import BaselinePredictor
from .trainer import BaselineCrossValidator, FoldTrainingResult


class BaselineMode(Enum):
    """Supported baseline command execution modes."""

    CROSS_VALIDATION = "cross-validation"
    BUILD = "build"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    """Summary of a completed baseline cross-validation run."""

    output_dir: Path
    folds: int
    predictions: int
    average_loss: float
    cache_path: Path
    cache_reused: bool


@dataclass(frozen=True, slots=True)
class BaselineBuildResult:
    """Summary of a completed full-data model build."""

    checkpoint_path: Path
    training_rows: int
    epochs: int
    final_training_loss: float
    cache_path: Path
    cache_reused: bool


@dataclass(frozen=True, slots=True)
class BaselineTestResult:
    """Summary of a completed competition test inference run."""

    output_dir: Path
    submission_path: Path
    predictions: int
    cache_path: Path
    cache_reused: bool


BaselineExecutionResult = BaselineRunResult | BaselineBuildResult | BaselineTestResult


class BaselineResultPublisher:
    """Atomically publish fold predictions under a datetime run name."""

    PROBABILITY_COLUMNS = (
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    )

    def __init__(self, results_root: Path) -> None:
        self._results_root = results_root.resolve()

    def publish(
        self,
        fold_results: tuple[FoldTrainingResult, ...],
        model_name: str,
    ) -> tuple[Path, float, int]:
        """Stage every artifact, then expose one complete run directory."""
        try:
            self._results_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise BaselineError(
                f"Could not create results directory {self._results_root}: {error}"
            ) from error

        generated_at = datetime.now(UTC).astimezone()
        run_name = generated_at.strftime("baseline-%Y%m%d-%H%M%S")
        output_dir = self._results_root / run_name
        if output_dir.exists():
            raise BaselineError(
                f"Baseline result directory already exists for this second: {output_dir}"
            )

        prediction_count = sum(len(result.ids) for result in fold_results)
        if prediction_count == 0:
            raise BaselineError("Baseline training produced no validation predictions.")
        weighted_loss = sum(
            result.final_loss * len(result.ids) for result in fold_results
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
                    "predictions": len(result.ids),
                    "final_validation_loss": result.final_loss,
                    "epochs": [asdict(epoch) for epoch in result.epochs],
                }
                for result in fold_results
            ],
        }

        try:
            with tempfile.TemporaryDirectory(
                dir=self._results_root,
                prefix=".baseline-",
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
        except BaselineError:
            raise
        except Exception as error:
            raise BaselineError(
                f"Could not publish baseline results to {output_dir}: {error}"
            ) from error
        return output_dir, average_loss, prediction_count

    def _validate_probabilities(self, predictions: pd.DataFrame, fold: int) -> None:
        values = predictions.loc[:, list(self.PROBABILITY_COLUMNS)].to_numpy(
            dtype=np.float64
        )
        if not _probabilities_are_valid(values):
            raise BaselineError(f"Fold {fold} produced invalid probabilities.")


class BaselineSubmissionPublisher:
    """Atomically publish one timestamped competition submission."""

    PROBABILITY_COLUMNS = BaselineResultPublisher.PROBABILITY_COLUMNS

    def __init__(self, results_root: Path) -> None:
        self._results_root = results_root.resolve()

    def publish(
        self,
        ids: tuple[int, ...],
        probabilities: np.ndarray,
    ) -> tuple[Path, Path]:
        """Write a submission CSV in a newly exposed test-run directory."""
        if probabilities.shape != (len(ids), len(self.PROBABILITY_COLUMNS)):
            raise BaselineError("Test probabilities do not match the test ID count.")
        if not _probabilities_are_valid(probabilities):
            raise BaselineError(
                "Baseline test inference produced invalid probabilities."
            )

        generated_at = datetime.now(UTC).astimezone()
        output_dir = self._results_root / generated_at.strftime(
            "baseline-%Y%m%d-%H%M%S"
        )
        submission_path = output_dir / "submission.csv"
        try:
            self._results_root.mkdir(parents=True, exist_ok=True)
            if output_dir.exists():
                raise BaselineError(
                    "Baseline test result directory already exists for this second: "
                    f"{output_dir}"
                )
            with tempfile.TemporaryDirectory(
                dir=self._results_root,
                prefix=".baseline-",
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
        except BaselineError:
            raise
        except Exception as error:
            raise BaselineError(
                f"Could not publish baseline test results to {output_dir}: {error}"
            ) from error
        return output_dir, submission_path


class BaselinePipeline:
    """Run cross-validation, build, or test inference for the baseline."""

    CACHE_SCHEMA_VERSION = 1
    CHECKPOINT_SCHEMA_VERSION = 1

    def __init__(self, config: AppConfig, project_root: Path) -> None:
        self._config = config
        self._training_repository = BaselineDataRepository(
            processed_path=config.data.processed_path,
            n_splits=config.cross_validation.n_splits,
        )
        self._test_repository = BaselineTestDataRepository(config.data.raw_dir)
        self._training_cache = EmbeddingCache(config.data.processed_path)
        self._test_cache = EmbeddingCache(
            config.data.processed_path,
            artifact_stem="baseline_test_embeddings",
        )
        project_root = project_root.resolve()
        self._checkpoint_store = BaselineCheckpointStore(
            project_root / "models" / "baseline" / "head.pt"
        )
        self._result_publisher = BaselineResultPublisher(project_root / "results")
        self._submission_publisher = BaselineSubmissionPublisher(
            project_root / "results" / "test"
        )

    def run(
        self,
        *,
        mode: BaselineMode = BaselineMode.CROSS_VALIDATION,
        refresh_cache: bool = False,
    ) -> BaselineExecutionResult:
        """Execute the selected baseline mode and return its summary."""
        if mode is BaselineMode.TEST:
            return self._run_test(refresh_cache=refresh_cache)

        device = GPUEnvironment(self._config.gpu).configure()
        data = self._training_repository.load()
        embeddings, cache_reused = self._load_or_extract_embeddings(
            records=data.canonical_texts,
            fingerprint=data.fingerprint,
            serializer_version=self._training_repository.serializer_version,
            cache=self._training_cache,
            refresh_cache=refresh_cache,
            device=device,
        )
        trainer = BaselineCrossValidator(
            config=self._config.baseline,
            n_splits=self._config.cross_validation.n_splits,
            device=device,
        )
        if mode is BaselineMode.BUILD:
            return self._build(data, embeddings, trainer, cache_reused)
        if mode is not BaselineMode.CROSS_VALIDATION:
            raise BaselineError(f"Unsupported baseline mode: {mode!r}")

        fold_results = trainer.train(data.reference, embeddings)
        output_dir, average_loss, prediction_count = self._result_publisher.publish(
            fold_results,
            self._config.baseline.model_name,
        )
        return BaselineRunResult(
            output_dir=output_dir,
            folds=len(fold_results),
            predictions=prediction_count,
            average_loss=average_loss,
            cache_path=self._training_cache.tensor_path,
            cache_reused=cache_reused,
        )

    def _build(
        self,
        data: BaselineData,
        embeddings: CachedEmbeddings,
        trainer: BaselineCrossValidator,
        cache_reused: bool,
    ) -> BaselineBuildResult:
        trained = trainer.train_all(data.reference, embeddings)
        self._checkpoint_store.save(
            trained.head,
            self._checkpoint_compatibility(
                self._training_repository.serializer_version
            ),
            backbone_hidden_size=embeddings.hidden_size,
            training_source_sha256=data.fingerprint,
        )
        return BaselineBuildResult(
            checkpoint_path=self._checkpoint_store.path,
            training_rows=trained.rows,
            epochs=len(trained.training_losses),
            final_training_loss=trained.training_losses[-1],
            cache_path=self._training_cache.tensor_path,
            cache_reused=cache_reused,
        )

    def _run_test(self, *, refresh_cache: bool) -> BaselineTestResult:
        data = self._test_repository.load()
        checkpoint = self._checkpoint_store.load(
            self._checkpoint_compatibility(self._test_repository.serializer_version)
        )
        device = GPUEnvironment(self._config.gpu).configure()
        embeddings, cache_reused = self._load_or_extract_embeddings(
            records=data.canonical_texts,
            fingerprint=data.fingerprint,
            serializer_version=self._test_repository.serializer_version,
            cache=self._test_cache,
            refresh_cache=refresh_cache,
            device=device,
        )
        probabilities = BaselinePredictor(
            config=self._config.baseline,
            device=device,
        ).predict(checkpoint, embeddings)
        output_dir, submission_path = self._submission_publisher.publish(
            data.ids,
            probabilities,
        )
        return BaselineTestResult(
            output_dir=output_dir,
            submission_path=submission_path,
            predictions=len(data.ids),
            cache_path=self._test_cache.tensor_path,
            cache_reused=cache_reused,
        )

    def _load_or_extract_embeddings(
        self,
        *,
        records: tuple[PairwiseText, ...],
        fingerprint: str,
        serializer_version: str,
        cache: EmbeddingCache,
        refresh_cache: bool,
        device: torch.device,
    ) -> tuple[CachedEmbeddings, bool]:
        expected_ids = torch.tensor(
            [record.row_id for record in records],
            dtype=torch.int64,
        )
        cache_key = self._cache_key(fingerprint, serializer_version)
        embeddings = None
        if not refresh_cache:
            embeddings = cache.load(cache_key, expected_ids)

        cache_reused = embeddings is not None
        if embeddings is None:
            print("No matching embedding cache found; extracting Qwen features.")
            extractor = QwenEmbeddingExtractor(
                gpu_config=self._config.gpu,
                baseline_config=self._config.baseline,
                device=device,
            )
            embeddings = extractor.extract(records)
            cache.save(cache_key, embeddings)
            print(f"Wrote embedding cache: {cache.tensor_path}")
            del extractor
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"Reused embedding cache: {cache.tensor_path}")
        return embeddings, cache_reused

    def _cache_key(
        self,
        fingerprint: str,
        serializer_version: str,
    ) -> EmbeddingCacheKey:
        return EmbeddingCacheKey(
            schema_version=self.CACHE_SCHEMA_VERSION,
            processed_sha256=fingerprint,
            model_name=self._config.baseline.model_name,
            max_length=self._config.baseline.max_length,
            serializer_version=serializer_version,
            precision=self._config.gpu.precision,
            attention_implementation=self._config.gpu.attention_implementation,
            allow_tf32=self._config.gpu.allow_tf32,
        )

    def _checkpoint_compatibility(
        self,
        serializer_version: str,
    ) -> HeadCheckpointCompatibility:
        return HeadCheckpointCompatibility(
            schema_version=self.CHECKPOINT_SCHEMA_VERSION,
            model_name=self._config.baseline.model_name,
            max_length=self._config.baseline.max_length,
            serializer_version=serializer_version,
            precision=self._config.gpu.precision,
            attention_implementation=self._config.gpu.attention_implementation,
            allow_tf32=self._config.gpu.allow_tf32,
            classifier_hidden_size=self._config.baseline.hidden_size,
        )


def _probabilities_are_valid(values: np.ndarray) -> bool:
    return bool(
        values.size > 0
        and np.isfinite(values).all()
        and not (values < 0.0).any()
        and not (values > 1.0).any()
        and np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    )
