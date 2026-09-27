"""Coordinate mono-input cross-validation, builds, and test inference."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from ...config import AppConfig
from ...errors import ModelExecutionError
from ..common.checkpoint import ModelCheckpointStore
from ..common.compatibility import ModelCompatibility
from ..common.hardware import GPUEnvironment
from ..common.pipeline import (
    ModelBuildResult,
    ModelExecutionResult,
    ModelMode,
    ModelRunResult,
    ModelTestResult,
)
from ..common.publishers import ModelResultPublisher, ModelSubmissionPublisher
from ..profile import ModelProfile
from .cache import MonoInputCachedEmbeddings, MonoInputEmbeddingCache
from .data import (
    MonoInputModelData,
    MonoInputModelDataRepository,
    MonoInputTestDataRepository,
    MonoInputText,
)
from .extractor import MonoInputEmbeddingExtractor
from .predictor import MonoInputPredictor
from .trainer import MonoInputCrossValidator


class MonoInputModelPipeline:
    """Run every supported mode for the structured mono-input model."""

    CACHE_SCHEMA_VERSION = 1
    CHECKPOINT_SCHEMA_VERSION = 3

    def __init__(
        self,
        config: AppConfig,
        project_root: Path,
        profile: ModelProfile,
    ) -> None:
        self._config = config
        self._profile = profile
        self._model_config = profile.resolve_config(config)
        self._compatibility = ModelCompatibility(
            gpu=config.gpu,
            model=self._model_config,
            cache_schema_version=self.CACHE_SCHEMA_VERSION,
            checkpoint_schema_version=self.CHECKPOINT_SCHEMA_VERSION,
        )
        self._training_repository = MonoInputModelDataRepository(
            processed_path=config.data.processed_path,
            n_splits=config.cross_validation.n_splits,
        )
        self._test_repository = MonoInputTestDataRepository(config.data.raw_dir)
        self._training_cache = MonoInputEmbeddingCache(
            config.data.processed_path,
            artifact_stem=f"{profile.artifact_stem}_embeddings",
        )
        self._test_cache = MonoInputEmbeddingCache(
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
        if mode is ModelMode.TEST:
            return self._run_test(refresh_cache=refresh_cache)

        devices = GPUEnvironment(self._config.gpu).configure()
        data = self._training_repository.load()
        embeddings, cache_reused = self._load_or_extract_embeddings(
            records=data.texts,
            fingerprint=data.fingerprint,
            serializer_version=self._training_repository.serializer_version,
            cache=self._training_cache,
            refresh_cache=refresh_cache,
            devices=devices,
        )
        trainer = MonoInputCrossValidator(
            config=self._model_config,
            n_splits=self._config.cross_validation.n_splits,
            device=devices[0],
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
            prediction_aggregation="ab_swap_average",
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
        data: MonoInputModelData,
        embeddings: MonoInputCachedEmbeddings,
        trainer: MonoInputCrossValidator,
        cache_reused: bool,
    ) -> ModelBuildResult:
        trained = trainer.train_all(data.reference, embeddings)
        self._checkpoint_store.save(
            trained.head,
            self._compatibility.checkpoint(
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
            self._compatibility.checkpoint(self._test_repository.serializer_version)
        )
        devices = GPUEnvironment(self._config.gpu).configure()
        embeddings, cache_reused = self._load_or_extract_embeddings(
            records=data.texts,
            fingerprint=data.fingerprint,
            serializer_version=self._test_repository.serializer_version,
            cache=self._test_cache,
            refresh_cache=refresh_cache,
            devices=devices,
        )
        probabilities = MonoInputPredictor(
            config=self._model_config,
            device=devices[0],
            selector=self._profile.selector,
            config_key=self._profile.config_key,
        ).predict(checkpoint, embeddings, data.ids)
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
        records: tuple[MonoInputText, ...],
        fingerprint: str,
        serializer_version: str,
        cache: MonoInputEmbeddingCache,
        refresh_cache: bool,
        devices: tuple[torch.device, ...],
    ) -> tuple[MonoInputCachedEmbeddings, bool]:
        expected_ids = torch.tensor(
            [record.row_id for record in records],
            dtype=torch.int64,
        )
        expected_swapped = torch.tensor(
            [record.is_swapped for record in records],
            dtype=torch.bool,
        )
        cache_key = self._compatibility.cache_key(fingerprint, serializer_version)
        embeddings = None
        if not refresh_cache:
            embeddings = cache.load(cache_key, expected_ids, expected_swapped)
        cache_reused = embeddings is not None
        if embeddings is None:
            print(
                "No matching mono-input embedding cache found; "
                "extracting Qwen features."
            )
            extractor = MonoInputEmbeddingExtractor(
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


__all__ = ("MonoInputModelPipeline",)
