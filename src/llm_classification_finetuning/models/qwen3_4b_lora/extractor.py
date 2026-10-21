"""Extract temporary pooled features for head-only warmup and inference."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ...errors import ModelExecutionError
from .data import BranchBatch, PairwiseFeatureStore
from .model import LoRAQwenClassifier


class LoRAEmbeddingExtractor:
	"""Keep paired features from the current backbone in temporary CPU memory."""

	def __init__(self, model_label: str = "LoRA") -> None:
		self._model_label = model_label

	@torch.no_grad()
	def extract(
		self,
		model: LoRAQwenClassifier,
		loader: DataLoader[BranchBatch],
		row_ids: tuple[int, ...],
		*,
		stage: str,
	) -> PairwiseFeatureStore:
		model.eval()
		features = torch.empty(
			(len(loader.dataset), model.backbone_hidden_size),
			dtype=model.feature_dtype,
			device="cpu",
		)
		start = 0
		for index, batch in enumerate(loader):
			pooled = model.encode_branch(batch)
			stop = start + batch.input_ids.shape[0]
			features[start:stop].copy_(pooled)
			del pooled
			if not torch.isfinite(features[start:stop]).all():
				raise ModelExecutionError(
					f"{self._model_label} {stage} extraction produced non-finite features."
				)
			start = stop
			if index == 0 or (index + 1) % 100 == 0:
				print(
					f"{self._model_label} {stage} extraction branch batch {index + 1:,}/{len(loader):,}.",
					flush=True,
				)
		return PairwiseFeatureStore(
			row_ids, features.view(len(row_ids), 2, model.backbone_hidden_size)
		)
