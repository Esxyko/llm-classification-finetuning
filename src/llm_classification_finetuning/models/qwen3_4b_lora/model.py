"""Shared Qwen backbone with trainable LoRA adapters and a pairwise MLP."""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from transformers import AutoModel

from ...config import GPUConfig, LoRAModelConfig
from ..common import ABSwapAverager, GPUEnvironment, PairwiseClassificationHead
from .data import PairwiseBatch


class LoRAQwenClassifier(nn.Module):
	"""Encode both branches through one differentiable adapter-bearing backbone."""

	def __init__(
		self,
		config: LoRAModelConfig,
		gpu: GPUConfig,
		device: torch.device,
		*,
		revision: str | None = None,
		device_map: dict[str, str] | None = None,
	) -> None:
		super().__init__()
		self.device = device
		self.dtype = GPUEnvironment.PRECISION_DTYPES[gpu.precision]
		backbone = AutoModel.from_pretrained(
			config.model_name,
			revision=revision,
			dtype=self.dtype,
			attn_implementation=gpu.attention_implementation,
			device_map=device_map,
		)
		backbone.config.use_cache = False
		backbone.requires_grad_(False)
		self.base_revision = getattr(backbone.config, "_commit_hash", None)
		self.backbone_hidden_size = int(backbone.config.hidden_size)
		self.backbone = get_peft_model(
			backbone,
			LoraConfig(
				task_type=TaskType.FEATURE_EXTRACTION,
				r=config.lora_rank,
				lora_alpha=config.lora_alpha,
				lora_dropout=config.lora_dropout,
				target_modules=list(config.lora_target_modules),
				bias="none",
			),
			autocast_adapter_dtype=True,
		)
		if device_map is None:
			self.backbone.to(device)
		if config.gradient_checkpointing and device_map is None:
			self.backbone.get_base_model().gradient_checkpointing_enable(
				gradient_checkpointing_kwargs={"use_reentrant": False}
			)
		if device_map is not None:
			self.backbone.get_base_model().gradient_checkpointing_disable()
		self.classifier = PairwiseClassificationHead(
			self.backbone_hidden_size, config.hidden_size, config.dropout
		).to(device=device, dtype=torch.float32)

	def encode(self, batch: PairwiseBatch) -> tuple[Tensor, Tensor]:
		"""Pool the last real token of each branch, retaining both gradient graphs."""
		pooled: list[Tensor] = []
		input_device = self.backbone.get_input_embeddings().weight.device
		for branch in (batch.branch_a, batch.branch_b):
			input_ids = branch.input_ids.to(input_device, non_blocking=True)
			mask = (
				branch.attention_mask.to(input_device, non_blocking=True)
				if branch.attention_mask is not None
				else None
			)
			with torch.autocast(
				device_type="cuda",
				dtype=self.dtype,
				enabled=self.dtype != torch.float32,
			):
				hidden = self.backbone(
					input_ids=input_ids,
					attention_mask=mask,
					use_cache=False,
					return_dict=True,
				).last_hidden_state
				last_positions = branch.last_positions.to(
					hidden.device, non_blocking=True
				)
				rows = torch.arange(hidden.shape[0], device=hidden.device)
				pooled.append(
					hidden[rows, last_positions].to(
						device=self.device, dtype=torch.float32
					)
				)
				del hidden
		return pooled[0], pooled[1]

	def forward(self, batch: PairwiseBatch) -> Tensor:
		"""Return FP32 A-win, B-win, and tie logits for the supplied orientation."""
		return self.classifier(*self.encode(batch))

	def log_probabilities(self, batch: PairwiseBatch) -> Tensor:
		"""Average both head orientations after encoding each branch once."""
		a, b = self.encode(batch)
		return ABSwapAverager.log_probabilities(self.classifier, a, b)
