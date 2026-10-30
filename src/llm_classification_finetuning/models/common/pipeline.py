"""Coordinate model holdout validation and builds."""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from transformers import AutoConfig

from ...config import AppConfig, ModelConfig
from ...errors import ModelExecutionError
from ...model_storage import load_pretrained
from ..profile import ModelProfile
from .cache import CachedEmbeddings, EmbeddingCache
from .checkpoint import ModelCheckpointStore
from .compatibility import ModelCompatibility
from .contracts import (
	ModelBuildResult,
	ModelExecutionResult,
	ModelMode,
	ModelRunResult,
)
from .data import (
	ConversationSerializer,
	ModelData,
	ModelDataRepository,
	PairwiseText,
)
from .extractor import QwenEmbeddingExtractor
from .hardware import GPUEnvironment, GPUMemoryMonitor
from .publishers import ModelResultPublisher
from .token_cache import TokenizationCache
from .trainer import ModelTrainer


class ModelPipeline:
	"""Run holdout validation or a full-data build for one model profile."""

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
		base_config = load_pretrained(
			AutoConfig.from_pretrained, self._model_config.model_name
		)
		self._base_revision = getattr(base_config, "_commit_hash", None)
		self._token_cache = TokenizationCache.from_pretrained(
			config.cache_dir / "token_cache",
			self._model_config.model_name,
			revision=self._base_revision,
		)
		serializer = ConversationSerializer(self._token_cache.tokenizer)
		self._training_repository = ModelDataRepository(
			processed_path=config.data.processed_path,
			serializer=serializer,
			validation_fraction=config.validation.validation_fraction,
		)
		self._embedding_cache = EmbeddingCache(config.cache_dir / "embedding_cache")
		project_root = project_root.resolve()
		checkpoint_name = f"head_{checkpoint_tag}.pt" if checkpoint_tag else "head.pt"
		self._checkpoint_store = ModelCheckpointStore(
			project_root / "models" / profile.artifact_stem / checkpoint_name,
		)
		self._result_publisher = ModelResultPublisher(
			config.result_dir,
			result_prefix=profile.result_prefix,
		)

	def run(
		self,
		*,
		mode: ModelMode = ModelMode.VALIDATION,
		refresh_cache: bool = False,
		gpu_memory_monitor: GPUMemoryMonitor | None = None,
	) -> ModelExecutionResult:
		"""Execute the selected model mode and return its summary."""
		if gpu_memory_monitor is not None:
			gpu_memory_monitor.start()
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
			gpu_memory_monitor=gpu_memory_monitor,
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

	def _load_or_extract_embeddings(
		self,
		*,
		records: tuple[PairwiseText, ...],
		refresh_cache: bool,
		device: torch.device,
	) -> tuple[CachedEmbeddings, bool]:
		extractor = QwenEmbeddingExtractor(
			gpu_config=self._config.gpu,
			model_config=self._model_config,
			device=device,
			token_cache_dir=self._config.cache_dir / "token_cache",
			token_cache=self._token_cache,
			revision=self._base_revision,
			refresh_cache=refresh_cache,
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
