"""Build unchanged embedding-cache and head-checkpoint compatibility keys."""

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

    def cache_key(self, fingerprint: str, serializer_version: str) -> EmbeddingCacheKey:
        return EmbeddingCacheKey(
            schema_version=self.cache_schema_version,
            processed_sha256=fingerprint,
            model_name=self.model.model_name,
            max_length=self.model.max_length,
            serializer_version=serializer_version,
            precision=self.gpu.precision,
            attention_implementation=self.gpu.attention_implementation,
            allow_tf32=self.gpu.allow_tf32,
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
        )
