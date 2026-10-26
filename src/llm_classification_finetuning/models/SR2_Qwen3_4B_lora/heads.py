"""LoRA classification heads with shared precision and construction policies."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..common.model import PairwiseClassificationHead

PAIRWISE_MLP = "pairwise_mlp"
EXPT = "swap_equivariant_reparam_v2"
EXPT2 = "swap_equivariant_tie_from_wins_v2"


class _LoRAClassificationHead(PairwiseClassificationHead):
	"""Use BF16 head computation with FP32 parameters and returned logits."""

	head_architecture = PAIRWISE_MLP

	def __init__(
		self,
		backbone_hidden_size: int,
		hidden_size: int,
		dropout: float,
		*,
		compute_dtype: torch.dtype,
	) -> None:
		super().__init__(backbone_hidden_size, hidden_size, dropout)
		self._compute_dtype = compute_dtype

	def forward(self, h_a: Tensor, h_b: Tensor) -> Tensor:
		"""Keep losses and orientation averaging outside reduced-precision compute."""
		with torch.autocast(
			device_type=h_a.device.type,
			dtype=torch.bfloat16,
			enabled=self._compute_dtype == torch.bfloat16,
		):
			logits = super().forward(h_a, h_b)
		return logits.float()


class SwapEquivariantClassificationHead(nn.Module):
	"""Project symmetric and directional features into three shared-readout scores."""

	head_architecture = EXPT
	EMBEDDING_COMBINATION = "sum_signed_diff_abs_diff"

	def __init__(
		self,
		backbone_hidden_size: int,
		hidden_size: int,
		dropout: float,
		*,
		compute_dtype: torch.dtype,
	) -> None:
		super().__init__()
		self._compute_dtype = compute_dtype
		# The fused bias holds b_u and b_w in the same order as the output chunks.
		self.symmetric_projection = nn.Linear(
			2 * backbone_hidden_size, 2 * hidden_size, bias=True
		)
		self.direction_projection = nn.Linear(
			backbone_hidden_size, hidden_size, bias=False
		)
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(dropout)
		self.readout = nn.Linear(hidden_size, 1, bias=False)

	@classmethod
	def embedding_combination(cls) -> str:
		"""Return the marker for the fixed sum, signed and absolute differences."""
		return cls.EMBEDDING_COMBINATION

	def forward(self, h_a: Tensor, h_b: Tensor) -> Tensor:
		"""Return FP32 A-win, B-win, and tie logits with a shared win dropout mask."""
		if h_a.shape != h_b.shape or h_a.ndim != 2:
			raise ValueError("h_a and h_b must have the same [batch, hidden] shape.")
		with torch.autocast(
			device_type=h_a.device.type,
			dtype=torch.bfloat16,
			enabled=self._compute_dtype == torch.bfloat16,
		):
			difference = h_a - h_b
			features = torch.cat((h_a + h_b, difference.abs()), dim=-1)
			u, w = self.symmetric_projection(features).chunk(2, dim=-1)
			v = self.direction_projection(difference)
			hidden = self.activation(torch.stack((u + v, u - v, w), dim=1))
			if self.training and self.dropout.p:
				# One win mask and one independent tie mask per feature pair.
				mask = self.dropout(
					hidden.new_ones((hidden.shape[0], 2, hidden.shape[-1]))
				)
				hidden = hidden * mask[:, [0, 0, 1], :]
			logits = self.readout(hidden).squeeze(-1)
		return logits.float()


class SwapEquivariantTieFromWinsClassificationHead(nn.Module):
	"""Derive a symmetric tie representation from the two activated win branches."""

	head_architecture = EXPT2
	EMBEDDING_COMBINATION = "sum_signed_diff_abs_diff"

	def __init__(
		self,
		backbone_hidden_size: int,
		hidden_size: int,
		dropout: float,
		*,
		compute_dtype: torch.dtype,
	) -> None:
		super().__init__()
		self._compute_dtype = compute_dtype
		self.symmetric_projection = nn.Linear(
			2 * backbone_hidden_size, hidden_size, bias=True
		)
		self.direction_projection = nn.Linear(
			backbone_hidden_size, hidden_size, bias=False
		)
		self.tie_projection = nn.Linear(hidden_size, hidden_size, bias=True)
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(dropout)
		self.readout = nn.Linear(hidden_size, 1, bias=False)
		# A/B wins share one bias; the tie branch has its own prior.
		self.readout_bias = nn.Parameter(torch.zeros(2, dtype=torch.float32))

	@classmethod
	def embedding_combination(cls) -> str:
		"""Return the marker for the fixed sum, signed and absolute differences."""
		return cls.EMBEDDING_COMBINATION

	def forward(self, h_a: Tensor, h_b: Tensor) -> Tensor:
		"""Return FP32 A-win, B-win, and tie logits with a shared win dropout mask."""
		if h_a.shape != h_b.shape or h_a.ndim != 2:
			raise ValueError("h_a and h_b must have the same [batch, hidden] shape.")
		with torch.autocast(
			device_type=h_a.device.type,
			dtype=torch.bfloat16,
			enabled=self._compute_dtype == torch.bfloat16,
		):
			difference = h_a - h_b
			features = torch.cat((h_a + h_b, difference.abs()), dim=-1)
			u = self.symmetric_projection(features)
			v = self.direction_projection(difference)
			wins = self.activation(torch.stack((u + v, u - v), dim=1))
			# Average the undropped representations before the single tie projection.
			tie = self.activation(self.tie_projection(wins.mean(dim=1)))
			hidden = torch.cat((wins, tie.unsqueeze(1)), dim=1)
			if self.training and self.dropout.p:
				mask = self.dropout(
					hidden.new_ones((hidden.shape[0], 2, hidden.shape[-1]))
				)
				hidden = hidden * mask[:, [0, 0, 1], :]
			logits = self.readout(hidden).squeeze(-1)
		return logits.float() + self.readout_bias[[0, 0, 1]]


def create_classification_head(
	backbone_hidden_size: int,
	hidden_size: int,
	dropout: float,
	*,
	compute_dtype: torch.dtype,
	head_architecture: str = EXPT2,
) -> nn.Module:
	"""Construct an EXPT2 head by default, retaining legacy head definitions."""
	if head_architecture == PAIRWISE_MLP:
		return _LoRAClassificationHead(
			backbone_hidden_size,
			hidden_size,
			dropout,
			compute_dtype=compute_dtype,
		)
	if head_architecture == EXPT:
		return SwapEquivariantClassificationHead(
			backbone_hidden_size,
			hidden_size,
			dropout,
			compute_dtype=compute_dtype,
		)
	if head_architecture == EXPT2:
		return SwapEquivariantTieFromWinsClassificationHead(
			backbone_hidden_size,
			hidden_size,
			dropout,
			compute_dtype=compute_dtype,
		)
	raise ValueError(f"Unknown classification head architecture: {head_architecture}.")
