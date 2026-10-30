"""Frozen Qwen3 pairwise classifier and trainable comparison head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import AutoModel

from ...model_storage import load_model_weights


class PairwiseClassificationHead(nn.Module):
	"""Classify pooled A/B states and their elementwise absolute difference."""

	EMBEDDING_COMBINATION = "concat_a_b_abs_diff"

	def __init__(
		self,
		backbone_hidden_size: int,
		hidden_size: int,
		dropout: float,
	) -> None:
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(backbone_hidden_size * 3, hidden_size),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, 3),
		)

	@classmethod
	def embedding_combination(cls) -> str:
		"""Return the result marker for the fixed classifier inputs."""
		return cls.EMBEDDING_COMBINATION

	def forward(self, h_a: Tensor, h_b: Tensor) -> Tensor:
		"""Return A-win, B-win, and tie logits for paired pooled states."""
		if h_a.shape != h_b.shape or h_a.ndim != 2:
			raise ValueError("h_a and h_b must have the same [batch, hidden] shape.")
		features = (h_a, h_b, torch.abs(h_a - h_b))
		return self.network(torch.cat(features, dim=-1))


class PairwiseQwenClassifier(nn.Module):
	"""Encode both branches sequentially through one frozen Qwen3 backbone."""

	def __init__(
		self,
		model_name: str,
		classifier_hidden_size: int,
		dropout: float,
		device: torch.device,
		dtype: torch.dtype,
		attention_implementation: str,
		*,
		revision: str | None = None,
	) -> None:
		super().__init__()
		self.device = device
		self.backbone = load_model_weights(
			AutoModel.from_pretrained,
			model_name,
			revision=revision,
			dtype=dtype,
			attn_implementation=attention_implementation,
			device_map={"": str(device)},
		)
		self.backbone.config.use_cache = False
		self.base_revision = getattr(self.backbone.config, "_commit_hash", None)
		self.backbone.requires_grad_(False)
		self.backbone.eval()
		self.backbone_hidden_size = int(self.backbone.config.hidden_size)
		self.classifier = PairwiseClassificationHead(
			backbone_hidden_size=self.backbone_hidden_size,
			hidden_size=classifier_hidden_size,
			dropout=dropout,
		).to(device)

	def train(self, mode: bool = True) -> PairwiseQwenClassifier:
		"""Change head mode without ever enabling stochastic backbone behavior."""
		super().train(mode)
		self.backbone.eval()
		return self

	@torch.no_grad()
	def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
		"""Return last-token pooled states with shape [batch, 2, hidden]."""
		if input_ids.ndim != 3 or input_ids.shape[1] != 2:
			raise ValueError("input_ids must have shape [batch, 2, length].")
		if attention_mask.shape != input_ids.shape:
			raise ValueError("attention_mask must match input_ids shape.")

		return torch.stack(
			[
				self.encode_branch(input_ids[:, branch], attention_mask[:, branch])
				for branch in range(2)
			],
			dim=1,
		)

	@torch.no_grad()
	def encode_branch(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
		"""Pool independently cacheable branches without computing their counterparts."""
		if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
			raise ValueError(
				"Branch tokens and masks must have matching [batch, length] shapes."
			)
		input_device = self.backbone.get_input_embeddings().weight.device
		device_ids = input_ids.to(input_device, non_blocking=True)
		device_mask = attention_mask.to(input_device, non_blocking=True)
		hidden_states = self.backbone(
			input_ids=device_ids,
			attention_mask=device_mask,
			use_cache=False,
			return_dict=True,
		).last_hidden_state
		return self._last_token_pool(hidden_states, device_mask).to(self.device)

	@staticmethod
	def _last_token_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
		"""Pool the final non-padding token of each response branch."""
		attention_mask = attention_mask.to(hidden_states.device, non_blocking=True)
		sequence_length = attention_mask.shape[1]
		positions = torch.arange(sequence_length, device=attention_mask.device)
		positions = positions.view(1, sequence_length).expand_as(attention_mask)
		last_positions = positions.masked_fill(attention_mask.eq(0), -1).amax(dim=-1)
		if last_positions.lt(0).any():
			raise ValueError("Every response branch must contain at least one token.")
		gather_indices = last_positions[:, None, None].expand(
			-1,
			1,
			hidden_states.shape[-1],
		)
		return hidden_states.gather(dim=1, index=gather_indices).squeeze(dim=1)

	def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
		"""Return three pairwise logits from tokenized A/B inputs."""
		pooled = self.encode(input_ids, attention_mask)
		classifier_dtype = next(self.classifier.parameters()).dtype
		return self.classifier(
			pooled[:, 0].to(dtype=classifier_dtype),
			pooled[:, 1].to(dtype=classifier_dtype),
		)
