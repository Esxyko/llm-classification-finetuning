"""LoRA classification heads with shared precision and construction policies."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..common.model import PairwiseClassificationHead

PAIRWISE_MLP = "pairwise_mlp"
EXPT = "swap_equivariant_reparam_v2"
EXPT2 = "swap_equivariant_tie_from_wins_v2"
SR2 = "swap_equivariant_abs_diff_v9"
# Bump this version whenever the SR2 initialization policy changes.
SR2_INITIALIZATION_VERSION = "scaled_uniform_clamped_gelu_tie_v5"


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
	"""Read symmetric sums and absolute differences of clamped win activations."""

	head_architecture = SR2
	EMBEDDING_COMBINATION = "cross_projected_clamped_win_sum_diff"

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
		# Scaled softplus keeps the effective tie weights positive.
		self.u = nn.Parameter(torch.empty((1, hidden_size), dtype=torch.float32))
		self.raw_t = nn.Parameter(torch.empty((), dtype=torch.float32))
		self.tie_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
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
		mean = 0.2 / (3.0 * hidden_size)
		with torch.no_grad():
			nn.init.uniform_(self.u, 0.5 * mean, 1.5 * mean)
			initial_u = self.u / 0.01
			# Stable inverse softplus: x + log(-expm1(-x)).
			self.u.copy_(0.01 * (initial_u + torch.log(-torch.expm1(-initial_u))))
			initial_t = self.raw_t.new_full((), 2.0)
			self.raw_t.copy_(initial_t + torch.log(-torch.expm1(-initial_t)))

	@classmethod
	def embedding_combination(cls) -> str:
		"""Return the marker for clamped win sums and absolute differences."""
		return cls.EMBEDDING_COMBINATION

	def forward(self, h_a: Tensor, h_b: Tensor) -> Tensor:
		"""Return FP32 A-win, B-win, and tie logits with shared win dropout."""
		if h_a.shape != h_b.shape or h_a.ndim != 2:
			raise ValueError("h_a and h_b must have the same [batch, hidden] shape.")
		device_type = h_a.device.type
		# One linear call covers both branches and both L/M weight blocks.
		branches = torch.stack((h_a, h_b), dim=1)
		if self._compute_dtype == torch.bfloat16:
			branches = branches.to(dtype=torch.bfloat16)
			w_in = self.input_projection.weight.bfloat16()
			b1 = self.win_projection_bias.bfloat16()
			w_out = self.win_readout.weight.bfloat16()
		else:
			w_in = self.input_projection.weight
			b1 = self.win_projection_bias
			w_out = self.win_readout.weight
		autocast_guard = (
			torch.autocast(device_type=device_type, enabled=False)
			if device_type in {"cuda", "cpu"} and torch.is_autocast_enabled(device_type)
			else nullcontext()
		)
		with autocast_guard:
			self_features, cross_features = F.linear(branches, w_in).chunk(2, dim=-1)
			wins = self.activation(
				self_features
				- cross_features.flip(1)
				+ (
					b1
					if b1.dtype == self_features.dtype
					else b1.to(dtype=self_features.dtype)
				)
			)
			if self.training and self.dropout.p:
				# One standard, scaled mask per sample is shared by both win branches.
				win_mask = self.dropout(
					wins.new_ones((wins.shape[0], 1, wins.shape[2]))
				)
				wins = wins * win_mask
			win_logits = F.linear(wins, w_out).squeeze(-1).float()
			c_a, c_b = wins.float().clamp_min(0).unbind(dim=1)
			tie_sum = c_a + c_b
			tie_diff = (c_a - c_b).abs()
			u_eff = F.softplus(self.u, beta=100.0)
			t = 1.0 + F.softplus(self.raw_t)
			tie_features = torch.addcmul(tie_sum, tie_diff, t, value=-1.0)
			tie = F.linear(tie_features, u_eff, self.tie_bias.unsqueeze(0))
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
