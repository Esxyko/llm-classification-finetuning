"""Apply the same label smoothing to logits and averaged log probabilities."""

from __future__ import annotations

from typing import Literal

from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F


class ClassificationLoss(CrossEntropyLoss):
	"""Cross entropy with uniform label smoothing for either training objective."""

	def __init__(
		self,
		label_smoothing: float = 0.0,
		*,
		reduction: Literal["mean", "sum"] = "mean",
	) -> None:
		super().__init__(label_smoothing=label_smoothing, reduction=reduction)

	def from_log_probabilities(
		self, log_probabilities: Tensor, labels: Tensor
	) -> Tensor:
		"""Smooth targets after A/B probability averaging, retaining its gradients."""
		loss = F.nll_loss(log_probabilities, labels, reduction=self.reduction)
		if self.label_smoothing == 0.0:
			return loss
		uniform_loss = -log_probabilities.mean(dim=-1)
		uniform_loss = (
			uniform_loss.sum() if self.reduction == "sum" else uniform_loss.mean()
		)
		return (1.0 - self.label_smoothing) * loss + self.label_smoothing * uniform_loss
