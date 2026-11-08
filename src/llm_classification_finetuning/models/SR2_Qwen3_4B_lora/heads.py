"""LoRA classification heads with shared precision and construction policies."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..common.model import PairwiseClassificationHead

PAIRWISE_MLP = "pairwise_mlp"
EXPT = "swap_equivariant_reparam_v2"
EXPT2 = "swap_equivariant_tie_from_wins_v2"
SR2 = "swap_equivariant_abs_diff_v7"
# Bump this version whenever the SR2 initialization policy changes.
SR2_INITIALIZATION_VERSION = "scaled_uniform_softplus_tie_v3"


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


class SwapEquivariantAbsoluteDifferenceClassificationHead(nn.Module):
	"""Combine cross-projected wins with sign-constrained absolute-difference ties."""

	head_architecture = SR2
	EMBEDDING_COMBINATION = "cross_projected_linear_abs_diff"

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
		# Contiguous weight blocks store L.T and M.T without a per-forward copy.
		self.input_projection = nn.Linear(
			backbone_hidden_size, 2 * hidden_size, bias=False
		)
		self.win_projection_bias = nn.Parameter(
			torch.zeros(hidden_size, dtype=torch.float32)
		)
		self.win_readout = nn.Linear(hidden_size, 1, bias=False)
		# Softplus constrains the effective tie weights and scalar bias each forward.
		self.raw_w_tie = nn.Parameter(
			torch.empty((1, hidden_size), dtype=torch.float32)
		)
		self.raw_b_tie = nn.Parameter(torch.empty((), dtype=torch.float32))
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(dropout)
		for layer, bound in (
			(
				self.input_projection,
				math.sqrt(3.0 / (2 * self.input_projection.in_features)),
			),
			(self.win_readout, 0.1 * math.sqrt(3.0 / self.win_readout.in_features)),
		):
			nn.init.uniform_(layer.weight, -bound, bound)
			if layer.bias is not None:
				nn.init.zeros_(layer.bias)
		tie_bound = 0.1 * math.sqrt(3.0 / hidden_size)
		with torch.no_grad():
			nn.init.uniform_(self.raw_w_tie, -tie_bound, tie_bound)
			tie_magnitude = self.raw_w_tie.abs().clamp_min(
				torch.finfo(torch.float32).eps
			)
			tie_bias = self.raw_b_tie.new_full((), 1e-4)
			for raw_parameter, value in (
				(self.raw_w_tie, tie_magnitude),
				(self.raw_b_tie, tie_bias),
			):
				# Stable inverse softplus, including small effective initial values.
				raw_parameter.copy_(value + torch.log(-torch.expm1(-value)))

	@classmethod
	def embedding_combination(cls) -> str:
		"""Return the marker for cross-projected wins and absolute-difference ties."""
		return cls.EMBEDDING_COMBINATION

	def forward(self, h_a: Tensor, h_b: Tensor) -> Tensor:
		"""Return FP32 A-win, B-win, and tie logits with shared win dropout."""
		if h_a.shape != h_b.shape or h_a.ndim != 2:
			raise ValueError("h_a and h_b must have the same [batch, hidden] shape.")
		with torch.autocast(
			device_type=h_a.device.type,
			dtype=torch.bfloat16,
			enabled=self._compute_dtype == torch.bfloat16,
		):
			# One linear call covers both branches and both L/M weight blocks.
			branches = torch.stack((h_a, h_b), dim=1)
			self_features, cross_features = self.input_projection(branches).chunk(
				2, dim=-1
			)
			wins = self.activation(
				self_features
				- cross_features.flip(1)
				+ self.win_projection_bias.to(dtype=self_features.dtype)
			)
			with torch.autocast(device_type=h_a.device.type, enabled=False):
				l_weight, m_weight = self.input_projection.weight.chunk(2, dim=0)
				# Form the weight sum and input difference in FP32 before projection.
				weight_sum = l_weight.float() + m_weight.float()
				input_difference = h_a.float() - h_b.float()
				w_tie = -F.softplus(self.raw_w_tie)
				b_tie = F.softplus(self.raw_b_tie)
			# |(A-B)(L+M)| is the absolute difference of the win preactivations.
			tie_features = F.linear(input_difference, weight_sum).float().abs()
			if self.training and self.dropout.p:
				# One standard, scaled mask per sample is reused only by the two wins.
				win_mask = self.dropout(torch.ones_like(wins[:, 0]))
				wins = wins * win_mask.unsqueeze(1)
			win_logits = self.win_readout(wins).squeeze(-1).float()
			tie_logits = F.linear(tie_features, w_tie).float()
		tie = tie_logits + b_tie
		return torch.cat((win_logits, tie), dim=-1)


def create_classification_head(
	backbone_hidden_size: int,
	hidden_size: int,
	dropout: float,
	*,
	compute_dtype: torch.dtype,
	head_architecture: str = SR2,
) -> nn.Module:
	"""Construct the current SR2 head by default, retaining legacy definitions."""
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
	if head_architecture == SR2:
		return SwapEquivariantAbsoluteDifferenceClassificationHead(
			backbone_hidden_size,
			hidden_size,
			dropout,
			compute_dtype=compute_dtype,
		)
	raise ValueError(f"Unknown classification head architecture: {head_architecture}.")
