"""Coordinate model holdout validation, builds, and submission inference."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from ...config import AppConfig, ModelConfig
from ...errors import ModelExecutionError
from ..profile import ModelProfile
from .attention import SubmissionAttention
from .cache import CachedEmbeddings, EmbeddingCache
from .checkpoint import ModelCheckpointStore
from .compatibility import ModelCompatibility
from .contracts import (
	ModelBuildResult,
	ModelExecutionResult,
	ModelMode,
	ModelRunResult,
	ModelSubmitResult,
)
from .data import (
	ModelData,
	ModelDataRepository,
	ModelTestDataRepository,
	PairwiseText,
)
from .extractor import QwenEmbeddingExtractor
from .hardware import GPUEnvironment
from .predictor import ModelPredictor
from .publishers import ModelResultPublisher, ModelSubmissionPublisher
from .trainer import ModelTrainer


class ModelPipeline:
	"""Run holdout validation, build, or submission inference for one model profile."""

	CACHE_SCHEMA_VERSION = 2
	CHECKPOINT_SCHEMA_VERSION = 3

	def __init__(
		self,
		config: AppConfig,
		project_root: Path,
		profile: ModelProfile,
		checkpoint_tag: str | None = None,
	) -> None:
		self._config = config
		self._profile = profile
		self._model_config = profile.resolve_config(config)
		if not isinstance(self._model_config, ModelConfig):
			raise TypeError("Frozen model execution requires a ModelConfig.")
		self._compatibility = ModelCompatibility(
			gpu=config.gpu,
			model=self._model_config,
			cache_schema_version=self.CACHE_SCHEMA_VERSION,
			checkpoint_schema_version=self.CHECKPOINT_SCHEMA_VERSION,
		)
		self._training_repository = ModelDataRepository(
			processed_path=config.data.processed_path,
			validation_fraction=config.validation.validation_fraction,
		)
		self._test_repository = ModelTestDataRepository(config.data.raw_dir)
		self._embedding_cache = EmbeddingCache(
			config.data.processed_path.parent / "embedding_cache"
		)
		project_root = project_root.resolve()
		checkpoint_name = f"head_{checkpoint_tag}.pt" if checkpoint_tag else "head.pt"
		self._checkpoint_store = ModelCheckpointStore(
			project_root / "models" / profile.artifact_stem / checkpoint_name,
			selector=profile.selector,
			checkpoint_tag=checkpoint_tag,
		)
		self._result_publisher = ModelResultPublisher(
			project_root / "results",
			result_prefix=profile.result_prefix,
		)
		self._submission_publisher = ModelSubmissionPublisher(
			project_root / "results" / "submit",
			result_prefix=profile.result_prefix,
		)

	def run(
		self,
		*,
		mode: ModelMode = ModelMode.VALIDATION,
		refresh_cache: bool = False,
	) -> ModelExecutionResult:
		"""Execute the selected model mode and return its summary."""
		if mode is ModelMode.SUBMIT:
			with SubmissionAttention(self._config.gpu).activate():
				return self._run_submit(refresh_cache=refresh_cache)

		device = GPUEnvironment(self._config.gpu).configure()
		data = self._training_repository.load()
		embeddings, cache_reused = self._load_or_extract_embeddings(
			records=data.canonical_texts,
			refresh_cache=refresh_cache,
			device=device,
		)
		trainer = ModelTrainer(
			config=self._model_config,
			device=device,
			config_key=self._profile.config_key,
		)
		if mode is ModelMode.BUILD:
			return self._build(data, embeddings, trainer, cache_reused)
		if mode is not ModelMode.VALIDATION:
			raise ModelExecutionError(f"Unsupported model mode: {mode!r}")

		validation_result = trainer.train(data.reference, embeddings)
		output_dir, validation_loss, prediction_count = self._result_publisher.publish(
			validation_result,
			self._config,
			self._profile,
			prediction_aggregation="ab_swap_average",
		)
		return ModelRunResult(
			output_dir=output_dir,
			predictions=prediction_count,
			validation_loss=validation_loss,
			cache_path=self._embedding_cache.root,
			cache_reused=cache_reused,
		)

	def _build(
		self,
		data: ModelData,
		embeddings: CachedEmbeddings,
		trainer: ModelTrainer,
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
			cache_path=self._embedding_cache.root,
			cache_reused=cache_reused,
		)

	def _run_submit(self, *, refresh_cache: bool) -> ModelSubmitResult:
		submit_devices = GPUEnvironment(self._config.gpu).configure_submit()
		device = submit_devices[0]
		data = self._test_repository.load()
		checkpoint = self._checkpoint_store.load(
			self._compatibility.checkpoint(self._test_repository.serializer_version)
		)
		embeddings, cache_reused = self._load_or_extract_embeddings(
			records=data.canonical_texts,
			refresh_cache=refresh_cache,
			device=device,
			submit_devices=submit_devices,
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
		return ModelSubmitResult(
			output_dir=output_dir,
			submission_path=submission_path,
			predictions=len(data.ids),
			cache_path=self._embedding_cache.root,
			cache_reused=cache_reused,
		)

	def _load_or_extract_embeddings(
		self,
		*,
		records: tuple[PairwiseText, ...],
		refresh_cache: bool,
		device: torch.device,
		submit_devices: tuple[torch.device, torch.device] | None = None,
	) -> tuple[CachedEmbeddings, bool]:
		extractor = QwenEmbeddingExtractor(
			gpu_config=self._config.gpu,
			model_config=self._model_config,
			device=device,
			token_cache_dir=self._config.data.processed_path.parent / "token_cache",
			refresh_cache=refresh_cache,
			submit_devices=submit_devices,
		)
		try:
			tokens = extractor.prepare(records)
			input_keys = extractor.input_keys(tokens)
			cache_key = self._compatibility.cache_key(
				base_revision=extractor.base_revision,
				local_weights_sha256=extractor.local_weights_sha256,
				hidden_size=extractor.hidden_size,
			)
			rows = self._embedding_cache.load(
				cache_key, input_keys, refresh_cache=refresh_cache
			)
			cache_reused = not rows.missing
			if rows.missing:
				extractor.extract(
					tokens,
					[positions[0] for positions in rows.missing.values()],
					rows.features,
				)
				for positions in rows.missing.values():
					if len(positions) > 1:
						rows.features[list(positions[1:])] = rows.features[
							positions[0]
						].clone()
				self._embedding_cache.save(cache_key, rows)
			paired = rows.features.view(len(records), 2, extractor.hidden_size)
			embeddings = CachedEmbeddings(
				ids=torch.tensor(
					[record.row_id for record in records], dtype=torch.int64
				),
				h_a=paired[:, 0],
				h_b=paired[:, 1],
			)
			print(
				f"Embedding cache: {rows.reused_entries:,} reused, {len(rows.missing):,} encoded unique branches; "
				f"{self._embedding_cache.path(cache_key)}",
				flush=True,
			)
			return embeddings, cache_reused
		finally:
			del extractor
			gc.collect()
			if torch.cuda.is_initialized():
				torch.cuda.empty_cache()
