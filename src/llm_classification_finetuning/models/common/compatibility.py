"""Build embedding-cache and head-checkpoint compatibility keys."""

from __future__ import annotations

from dataclasses import dataclass

from ...config import GPUConfig, ModelConfig
from .cache import EmbeddingCacheKey
from .checkpoint import HeadCheckpointCompatibility


@dataclass(frozen=True, slots=True)
class ModelCompatibility:
	gpu: GPUConfig
	model: ModelConfig
	cache_schema_version: int
	checkpoint_schema_version: int

	def cache_key(
		self,
		*,
		base_revision: str | None,
		local_weights_sha256: str | None,
		hidden_size: int,
	) -> EmbeddingCacheKey:
		return EmbeddingCacheKey(
			schema_version=self.cache_schema_version,
			model_name=self.model.model_name
			if local_weights_sha256 is None
			else "local-content",
			base_revision=base_revision if local_weights_sha256 is None else None,
			local_weights_sha256=local_weights_sha256,
			hidden_size=hidden_size,
			precision=self.gpu.precision,
		)

	def checkpoint(self, serializer_version: str) -> HeadCheckpointCompatibility:
		return HeadCheckpointCompatibility(
			schema_version=self.checkpoint_schema_version,
			model_name=self.model.model_name,
			max_length=self.model.max_length,
			serializer_version=serializer_version,
			precision=self.gpu.precision,
			attention_implementation=self.gpu.attention_implementation,
			allow_tf32=self.gpu.allow_tf32,
			classifier_hidden_size=self.model.hidden_size,
			ab_swap=self.model.ab_swap,
		)
