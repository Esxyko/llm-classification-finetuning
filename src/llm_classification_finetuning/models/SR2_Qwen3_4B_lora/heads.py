"""LoRA classification heads with shared precision and construction policies."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..common.model import PairwiseClassificationHead

PAIRWISE_MLP = "pairwise_mlp"
EXPT = "swap_equivariant_reparam_v2"
EXPT2 = "swap_equivariant_tie_from_wins_v2"
SR2 = "swap_equivariant_harmonic_discrepancy_v2"


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


class SwapEquivariantHarmonicDiscrepancyClassificationHead(nn.Module):
	"""Combine cross-projected wins with harmonic and discrepancy representations."""

	head_architecture = SR2
	EMBEDDING_COMBINATION = "cross_projected_harmonic_abs_diff"
	EPSILON = 1e-8

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
		# L carries B1 and is reused for both branches; M has no bias.
		self.self_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=True)
		self.cross_projection = nn.Linear(backbone_hidden_size, hidden_size, bias=False)
		# N carries B2; R, S, and T have no output biases.
		self.discrepancy_projection = nn.Linear(hidden_size, hidden_size, bias=True)
		self.win_readout = nn.Linear(hidden_size, 1, bias=False)
		self.tie_readout = nn.Linear(hidden_size, 1, bias=False)
		self.discrepancy_readout = nn.Linear(hidden_size, 1, bias=False)
		self.tie_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(dropout)
		for layer in (
			self.self_projection,
			self.cross_projection,
			self.discrepancy_projection,
			self.win_readout,
			self.tie_readout,
			self.discrepancy_readout,
		):
			nn.init.xavier_uniform_(layer.weight, gain=1.0)
			if layer.bias is not None:
				nn.init.zeros_(layer.bias)

	@classmethod
	def embedding_combination(cls) -> str:
		"""Return the marker for cross projections, fusion, and absolute differences."""
		return cls.EMBEDDING_COMBINATION

	def forward(self, h_a: Tensor, h_b: Tensor) -> Tensor:
		"""Return FP32 logits with batched win readout and a unit tie correction."""
		if h_a.shape != h_b.shape or h_a.ndim != 2:
			raise ValueError("h_a and h_b must have the same [batch, hidden] shape.")
		with torch.autocast(
			device_type=h_a.device.type,
			dtype=torch.bfloat16,
			enabled=self._compute_dtype == torch.bfloat16,
		):
			# Batch both branches so L and M each require only one linear call.
			branches = torch.stack((h_a, h_b), dim=1)
			wins = self.activation(
				self.self_projection(branches) + self.cross_projection(branches).flip(1)
			)
			p1, p2 = wins.unbind(dim=1)
			# Both derived representations use the undropped win features.
			with torch.autocast(device_type=h_a.device.type, enabled=False):
				p1_fp32, p2_fp32 = wins.float().unbind(dim=1)
				denominator = p1_fp32.abs() + p2_fp32.abs() + self.EPSILON
				fused = 2 * p1_fp32 * p2_fp32 / denominator
			discrepancy = self.activation(self.discrepancy_projection((p1 - p2).abs()))
			if self.training and self.dropout.p:
				# One standard, scaled mask per sample is reused only by the two wins.
				win_mask = self.dropout(torch.ones_like(p1))
				wins = wins * win_mask.unsqueeze(1)
			# Separate dropout calls draw independent fusion and discrepancy masks.
			fused = self.dropout(fused)
			discrepancy = self.dropout(discrepancy)
			win_logits = self.win_readout(wins).squeeze(-1).float()
			s = self.tie_readout(fused).float()
			q = self.discrepancy_readout(discrepancy).float()
		tie = s - q + self.tie_bias
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
		return SwapEquivariantHarmonicDiscrepancyClassificationHead(
			backbone_hidden_size,
			hidden_size,
			dropout,
			compute_dtype=compute_dtype,
		)
	raise ValueError(f"Unknown classification head architecture: {head_architecture}.")
