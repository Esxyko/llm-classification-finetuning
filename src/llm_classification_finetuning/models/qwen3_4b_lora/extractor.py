"""Extract temporary frozen-backbone features for head-only warmup."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from ...errors import ModelExecutionError
from .data import PairwiseBatch
from .model import LoRAQwenClassifier


class WarmupEmbeddingExtractor:
	"""Keep paired warmup features in CPU memory until joint training starts."""

	@torch.no_grad()
	def extract(
		self, model: LoRAQwenClassifier, loader: DataLoader[PairwiseBatch]
	) -> TensorDataset:
		model.eval()
		features = torch.empty(
			(len(loader.dataset), 2, model.backbone_hidden_size), dtype=torch.float32
		)
		labels = torch.empty(len(loader.dataset), dtype=torch.long)
		start = 0
		for index, batch in enumerate(loader):
			pooled = torch.stack(model.encode(batch), dim=1)
			if not torch.isfinite(pooled).all():
				raise ModelExecutionError(
					"LoRA warmup extraction produced non-finite features."
				)
			stop = start + len(batch.labels)
			features[start:stop].copy_(pooled)
			labels[start:stop].copy_(batch.labels)
			start = stop
			if index == 0 or (index + 1) % 100 == 0:
				print(
					f"LoRA warmup extraction batch {index + 1:,}/{len(loader):,}.",
					flush=True,
				)
		return TensorDataset(features, labels)
