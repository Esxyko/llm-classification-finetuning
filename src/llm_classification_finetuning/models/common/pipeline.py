"""Coordinate model cross-validation, builds, and test inference."""

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

from ...config import AppConfig
from ...errors import ModelExecutionError
from ..profile import ModelProfile
from .cache import CachedEmbeddings, EmbeddingCache, EmbeddingCacheKey
from .checkpoint import HeadCheckpointCompatibility, ModelCheckpointStore
from .data import (
    ModelData,
    ModelDataRepository,
    ModelTestDataRepository,
    PairwiseText,
)
from .extractor import QwenEmbeddingExtractor
from .hardware import GPUEnvironment
from .predictor import ModelPredictor
from .trainer import FoldTrainingResult, ModelCrossValidator


class ModelMode(Enum):
    """Supported model command execution modes."""

    CROSS_VALIDATION = "cross-validation"
    BUILD = "build"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class ModelRunResult:
    """Summary of a completed model cross-validation run."""

    output_dir: Path
    folds: int
    predictions: int
    average_loss: float
    cache_path: Path
    cache_reused: bool


@dataclass(frozen=True, slots=True)
class ModelBuildResult:
    """Summary of a completed full-data model build."""

    checkpoint_path: Path
    training_rows: int
    epochs: int
    final_training_loss: float
    cache_path: Path
    cache_reused: bool


@dataclass(frozen=True, slots=True)
class ModelTestResult:
    """Summary of a completed competition test inference run."""

    output_dir: Path
    submission_path: Path
    predictions: int
    cache_path: Path
    cache_reused: bool


ModelExecutionResult = ModelRunResult | ModelBuildResult | ModelTestResult


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

        prediction_count = sum(len(result.ids) for result in fold_results)
        if prediction_count == 0:
            raise ModelExecutionError(
                "Model training produced no validation predictions."
            )
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
            raise ModelExecutionError(
                f"Fold {fold} produced invalid probabilities."
            )


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


class ModelPipeline:
    """Run cross-validation, build, or test inference for one model profile."""

    CACHE_SCHEMA_VERSION = 1
    CHECKPOINT_SCHEMA_VERSION = 2

    def __init__(
        self,
        config: AppConfig,
        project_root: Path,
        profile: ModelProfile,
    ) -> None:
        self._config = config
        self._profile = profile
        self._model_config = profile.resolve_config(config)
        self._training_repository = ModelDataRepository(
            processed_path=config.data.processed_path,
            n_splits=config.cross_validation.n_splits,
        )
        self._test_repository = ModelTestDataRepository(config.data.raw_dir)
        self._training_cache = EmbeddingCache(
            config.data.processed_path,
            artifact_stem=f"{profile.artifact_stem}_embeddings",
        )
        self._test_cache = EmbeddingCache(
            config.data.processed_path,
            artifact_stem=f"{profile.artifact_stem}_test_embeddings",
        )
        project_root = project_root.resolve()
        self._checkpoint_store = ModelCheckpointStore(
            project_root / "models" / profile.artifact_stem / "head.pt",
            selector=profile.selector,
        )
        self._result_publisher = ModelResultPublisher(
            project_root / "results",
            result_prefix=profile.result_prefix,
        )
        self._submission_publisher = ModelSubmissionPublisher(
            project_root / "results" / "test",
            result_prefix=profile.result_prefix,
        )

    def run(
        self,
        *,
        mode: ModelMode = ModelMode.CROSS_VALIDATION,
        refresh_cache: bool = False,
    ) -> ModelExecutionResult:
        """Execute the selected model mode and return its summary."""
        if mode is ModelMode.TEST:
            return self._run_test(refresh_cache=refresh_cache)

        devices = GPUEnvironment(self._config.gpu).configure()
        device = devices[0]
        data = self._training_repository.load()
        embeddings, cache_reused = self._load_or_extract_embeddings(
            records=data.canonical_texts,
            fingerprint=data.fingerprint,
            serializer_version=self._training_repository.serializer_version,
            cache=self._training_cache,
            refresh_cache=refresh_cache,
            devices=devices,
        )
        trainer = ModelCrossValidator(
            config=self._model_config,
            n_splits=self._config.cross_validation.n_splits,
            device=device,
            config_key=self._profile.config_key,
        )
        if mode is ModelMode.BUILD:
            return self._build(data, embeddings, trainer, cache_reused)
        if mode is not ModelMode.CROSS_VALIDATION:
            raise ModelExecutionError(f"Unsupported model mode: {mode!r}")

        fold_results = trainer.train(data.reference, embeddings)
        output_dir, average_loss, prediction_count = self._result_publisher.publish(
            fold_results,
            self._model_config.model_name,
        )
        return ModelRunResult(
            output_dir=output_dir,
            folds=len(fold_results),
            predictions=prediction_count,
            average_loss=average_loss,
            cache_path=self._training_cache.tensor_path,
            cache_reused=cache_reused,
        )

    def _build(
        self,
        data: ModelData,
        embeddings: CachedEmbeddings,
        trainer: ModelCrossValidator,
        cache_reused: bool,
    ) -> ModelBuildResult:
        trained = trainer.train_all(data.reference, embeddings)
        self._checkpoint_store.save(
            trained.head,
            self._checkpoint_compatibility(
                self._training_repository.serializer_version
            ),
            backbone_hidden_size=embeddings.hidden_size,
            training_source_sha256=data.fingerprint,
        )
        return ModelBuildResult(
            checkpoint_path=self._checkpoint_store.path,
            training_rows=trained.rows,
            epochs=len(trained.training_losses),
            final_training_loss=trained.training_losses[-1],
            cache_path=self._training_cache.tensor_path,
            cache_reused=cache_reused,
        )

    def _run_test(self, *, refresh_cache: bool) -> ModelTestResult:
        data = self._test_repository.load()
        checkpoint = self._checkpoint_store.load(
            self._checkpoint_compatibility(self._test_repository.serializer_version)
        )
        devices = GPUEnvironment(self._config.gpu).configure()
        device = devices[0]
        embeddings, cache_reused = self._load_or_extract_embeddings(
            records=data.canonical_texts,
            fingerprint=data.fingerprint,
            serializer_version=self._test_repository.serializer_version,
            cache=self._test_cache,
            refresh_cache=refresh_cache,
            devices=devices,
        )
        probabilities = ModelPredictor(
            config=self._model_config,
            device=device,
            selector=self._profile.selector,
            config_key=self._profile.config_key,
        ).predict(checkpoint, embeddings)
        output_dir, submission_path = self._submission_publisher.publish(
            data.ids,
            probabilities,
        )
        return ModelTestResult(
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
        devices: tuple[torch.device, ...],
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
                model_config=self._model_config,
                devices=devices,
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
            model_name=self._model_config.model_name,
            max_length=self._model_config.max_length,
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
            model_name=self._model_config.model_name,
            max_length=self._model_config.max_length,
            serializer_version=serializer_version,
            precision=self._config.gpu.precision,
            attention_implementation=self._config.gpu.attention_implementation,
            allow_tf32=self._config.gpu.allow_tf32,
            classifier_hidden_size=self._model_config.hidden_size,
        )


def _probabilities_are_valid(values: np.ndarray) -> bool:
    return bool(
        values.size > 0
        and np.isfinite(values).all()
        and not (values < 0.0).any()
        and not (values > 1.0).any()
        and np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    )
