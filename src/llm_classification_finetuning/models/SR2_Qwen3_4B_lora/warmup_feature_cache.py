"""Reuse adapter-free warmup branch features independently of head training state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from ...errors import ModelExecutionError
from ..common import EmbeddingCache, EmbeddingCacheKey, EmbeddingCacheRows
from .data import BranchTokenDataset
from .warmup_cache import WarmupCacheContext


@dataclass(frozen=True, slots=True)
class WarmupFeatureCacheEntry:
	"""Checked branch hits and the representative positions still needing extraction."""

	key: EmbeddingCacheKey
	rows: EmbeddingCacheRows

	@property
	def missing_indices(self) -> tuple[int, ...]:
		return tuple(positions[0] for positions in self.rows.missing.values())


class WarmupFeatureCache:
	"""Persist exact pooled feature dtypes using the shared branch embedding store."""

	SCHEMA_VERSION = 1
	EXTRACTION_VERSION = 1
	POOLING = "last-real-token-right-padding-no-mask-v1"

	def __init__(self, root: Path, *, model_label: str = "LoRA") -> None:
		self._cache = EmbeddingCache(root)
		self._model_label = model_label

	def prepare(
		self,
		dataset: BranchTokenDataset,
		context: WarmupCacheContext,
		*,
		model_name: str,
		precision: str,
		feature_dtype: torch.dtype,
		refresh_cache: bool = False,
	) -> WarmupFeatureCacheEntry:
		"""Hash complete branch tokens and load vectors into the current canonical order."""
		if context.base_revision is None and context.local_weights_sha256 is None:
			raise ModelExecutionError(
				"Warmup feature caching requires an immutable backbone identity."
			)
		local = context.local_weights_sha256 is not None
		key = EmbeddingCacheKey(
			schema_version=self.SCHEMA_VERSION,
			model_name="local-content" if local else model_name,
			base_revision=None if local else context.base_revision,
			local_weights_sha256=context.local_weights_sha256,
			hidden_size=context.backbone_hidden_size,
			precision=precision,
			pooling=self.POOLING,
			storage_dtype=str(feature_dtype).removeprefix("torch."),
			extraction_version=self.EXTRACTION_VERSION,
		)
		input_keys = [
			hashlib.sha256(
				memoryview(dataset[index].astype("<i4", copy=False))
			).hexdigest()
			for index in range(len(dataset))
		]
		rows = self._cache.load(key, input_keys, refresh_cache=refresh_cache)
		print(
			f"{self._model_label} warmup feature cache: "
			f"reused={rows.reused_entries:,}, missing={len(rows.missing):,} unique branches; "
			f"{self._cache.path(key)}",
			flush=True,
		)
		return WarmupFeatureCacheEntry(key, rows)

	def save(self, entry: WarmupFeatureCacheEntry) -> None:
		"""Populate duplicate branch positions before atomically publishing new vectors."""
		for positions in entry.rows.missing.values():
			if len(positions) > 1:
				entry.rows.features[list(positions[1:])] = entry.rows.features[
					positions[0]
				].clone()
		self._cache.save(entry.key, entry.rows)
