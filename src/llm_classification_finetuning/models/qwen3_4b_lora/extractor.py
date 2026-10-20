"""Extract temporary frozen-backbone features for head-only warmup."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ...errors import ModelExecutionError
from .data import PairwiseBatch, PairwiseFeatureStore
from .model import LoRAQwenClassifier


class WarmupEmbeddingExtractor:
	"""Keep paired warmup features in CPU memory until joint training starts."""

	@torch.no_grad()
	def extract(
		self,
		model: LoRAQwenClassifier,
		loader: DataLoader[PairwiseBatch],
		row_ids: tuple[int, ...],
	) -> PairwiseFeatureStore:
		model.eval()
		features = torch.empty(
			(len(loader.dataset), 2, model.backbone_hidden_size), dtype=torch.float32
		)
		start = 0
		for index, batch in enumerate(loader):
			pooled = torch.stack(model.encode(batch), dim=1)
			stop = start + len(batch.labels)
			features[start:stop].copy_(pooled)
			del pooled
			if not torch.isfinite(features[start:stop]).all():
				raise ModelExecutionError(
					"LoRA warmup extraction produced non-finite features."
				)
			start = stop
			if index == 0 or (index + 1) % 100 == 0:
				print(
					f"LoRA warmup extraction batch {index + 1:,}/{len(loader):,}.",
					flush=True,
				)
		return PairwiseFeatureStore(row_ids, features)
