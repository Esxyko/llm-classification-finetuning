"""Shared Qwen backbone with trainable LoRA adapters and a comparison head."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from ...config import GPUConfig, LoRAModelConfig, WarmupModelConfig
from ...errors import ModelExecutionError
from ...model_storage import load_model_weights, load_pretrained
from ..common import ABSwapAverager, GPUEnvironment
from .data import BranchBatch, PairwiseBatch
from .heads import EXPT, EXPT2, SR2, create_classification_head


class LoRAQwenClassifier(nn.Module):
	"""Encode both branches through one differentiable adapter-bearing backbone."""

	def __init__(
		self,
		config: WarmupModelConfig,
		gpu: GPUConfig,
		device: torch.device,
		*,
		revision: str | None = None,
		quantize_backbone: bool = False,
		warmup_only: bool = False,
		head_architecture: str = SR2,
	) -> None:
		super().__init__()
		if not warmup_only and not isinstance(config, LoRAModelConfig):
			raise TypeError("Adapter initialization requires a LoRAModelConfig.")
		if warmup_only and quantize_backbone:
			raise ValueError("Standalone warmup requires a non-quantized backbone.")
		self.warmup_only = warmup_only
		self.head_architecture = head_architecture
		self._gradient_checkpointing_allowed = not warmup_only
		self._gradient_checkpointing_threshold = (
			config.gradient_checkpointing_threshold if not warmup_only else 0
		)
		self.device = device
		self.dtype = GPUEnvironment.PRECISION_DTYPES[gpu.precision]
		self.feature_dtype = (
			torch.bfloat16 if self.dtype == torch.bfloat16 else torch.float32
		)
		if warmup_only:
			# Config/tokenizer identities remain mandatory even on complete feature hits.
			base_config = load_pretrained(
				AutoConfig.from_pretrained, config.model_name, revision=revision
			)
			self.base_revision = getattr(base_config, "_commit_hash", None)
			self.backbone_hidden_size = int(base_config.hidden_size)
			self.backbone = None
			self._frozen_model_name = config.model_name
			self._frozen_attention = gpu.attention_implementation
			self._gradient_checkpointing_enabled = False
			self._initialize_head(config)
			return
		quantization_config = (
			BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_use_double_quant=True,
				bnb_4bit_compute_dtype=self.dtype,
			)
			if quantize_backbone
			else None
		)
		backbone = load_model_weights(
			AutoModel.from_pretrained,
			config.model_name,
			revision=revision,
			dtype=self.dtype,
			attn_implementation=gpu.attention_implementation,
			device_map={"": str(device)},
			quantization_config=quantization_config,
		)
		backbone.config.use_cache = False
		backbone.requires_grad_(False)
		if quantize_backbone:
			# Select checkpointing per pairwise batch with our SDPA recomputation context.
			backbone = prepare_model_for_kbit_training(
				backbone, use_gradient_checkpointing=False, auto_clear_cache=False
			)
			if self.dtype == torch.bfloat16:
				# K-bit preparation upcasts these weights, promoting the residual stream.
				# Restore BF16 storage/output while RMSNorm keeps FP32 variance math.
				backbone.get_input_embeddings().to(dtype=torch.bfloat16)
				for module in backbone.modules():
					if isinstance(module, Qwen3RMSNorm):
						module.to(dtype=torch.bfloat16)
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
			# Keep adapter parameters, their gradients, and AdamW moments in FP32.
			autocast_adapter_dtype=True,
		)
		if self.dtype == torch.bfloat16:
			# Fix the input policy once for forward and checkpoint recomputation.
			# Autocast handles adapter matmuls with FP32 adapter parameters.
			for module in self.backbone.modules():
				if isinstance(module, BaseTunerLayer):
					module.cast_input_dtype_enabled = False
		self._gradient_checkpointing_enabled = self.backbone.is_gradient_checkpointing
		self._set_gradient_checkpointing(False)
		self._initialize_head(config)

	def _initialize_head(self, config: WarmupModelConfig) -> None:
		"""Allocate real random head parameters, independently of deferred extraction."""
		self.classifier = create_classification_head(
			self.backbone_hidden_size,
			config.hidden_size,
			config.dropout,
			compute_dtype=self.feature_dtype,
			head_architecture=self.head_architecture,
		).to(device=self.device, dtype=torch.float32)

	def load_frozen_backbone(self) -> None:
		"""Load weights only for missing features without advancing the head's RNG."""
		if self.backbone is not None:
			return
		if not self.warmup_only:
			raise ModelExecutionError(
				"Joint training requires an initialized backbone."
			)
		with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
			backbone = load_model_weights(
				AutoModel.from_pretrained,
				self._frozen_model_name,
				revision=self.base_revision,
				dtype=self.dtype,
				attn_implementation=self._frozen_attention,
				device_map={"": str(self.device)},
				quantization_config=None,
			)
		if (
			getattr(backbone.config, "_commit_hash", None) != self.base_revision
			or int(backbone.config.hidden_size) != self.backbone_hidden_size
		):
			raise ModelExecutionError(
				"Loaded frozen backbone does not match the prepared model identity."
			)
		backbone.config.use_cache = False
		backbone.requires_grad_(False)
		backbone.gradient_checkpointing_disable()
		backbone.disable_input_require_grads()
		self.backbone = backbone.eval()

	def train(self, mode: bool = True) -> LoRAQwenClassifier:
		"""Disable checkpointing for evaluation; joint encoding selects training state."""
		super().train(mode)
		if self.warmup_only and self.backbone is not None:
			self.backbone.eval()
		if not mode:
			self._set_gradient_checkpointing(False)
		return self

	def _set_gradient_checkpointing(self, enabled: bool) -> None:
		"""Switch only when needed and remove input-gradient hooks when disabling."""
		enabled = enabled and self._gradient_checkpointing_allowed
		if enabled == self._gradient_checkpointing_enabled:
			return
		backbone = self.backbone if self.warmup_only else self.backbone.get_base_model()
		if enabled:
			backbone.gradient_checkpointing_enable(
				gradient_checkpointing_kwargs={
					"use_reentrant": False,
					"context_fn": self._checkpoint_contexts,
				}
			)
		else:
			backbone.gradient_checkpointing_disable()
			# Transformers may leave these hooks installed on PEFT-wrapped backbones.
			backbone.disable_input_require_grads()
		self._gradient_checkpointing_enabled = enabled

	def encode(self, batch: PairwiseBatch) -> tuple[Tensor, Tensor]:
		"""Pool the last real token of each branch, retaining both gradient graphs."""
		padded_tokens = (
			batch.branch_a.input_ids.numel() + batch.branch_b.input_ids.numel()
		)
		self._set_gradient_checkpointing(
			self.training
			and torch.is_grad_enabled()
			and (
				self._gradient_checkpointing_threshold == 0
				or padded_tokens > self._gradient_checkpointing_threshold
			)
		)
		# Retain one policy for both branches and the caller's complete backward pass.
		return self.encode_branch(batch.branch_a), self.encode_branch(batch.branch_b)

	def encode_branch(self, branch: BranchBatch) -> Tensor:
		"""Encode one batch of branches for extraction or differentiable training."""
		if self.backbone is None:
			self.load_frozen_backbone()
		if not self.training or not torch.is_grad_enabled():
			self._set_gradient_checkpointing(False)
		input_device = self.backbone.get_input_embeddings().weight.device
		input_ids = branch.input_ids.to(input_device, non_blocking=True)
		with (
			self._backbone_context(),
			torch.autocast(
				device_type="cuda",
				dtype=self.dtype,
				enabled=self.dtype != torch.float32,
			),
		):
			# Qwen retains causal attention, so real tokens cannot see right padding.
			# Omitting the padding mask lets SDPA use native GQA.
			hidden = self.backbone(
				input_ids=input_ids,
				attention_mask=None,
				use_cache=False,
				return_dict=True,
			).last_hidden_state
			last_positions = branch.last_positions.to(hidden.device, non_blocking=True)
			rows = torch.arange(hidden.shape[0], device=hidden.device)
			pooled = hidden[rows, last_positions].to(
				device=self.device, dtype=self.feature_dtype
			)
			del hidden
		return pooled

	def forward(self, batch: PairwiseBatch) -> Tensor:
		"""Return FP32 A-win, B-win, and tie logits for the supplied orientation."""
		return self.classifier(*self.encode(batch))

	def log_probabilities(self, batch: PairwiseBatch) -> Tensor:
		"""Return the selected head's original-only or averaged log probabilities."""
		a, b = self.encode(batch)
		if self.head_architecture in {EXPT, EXPT2, SR2}:
			return F.log_softmax(self.classifier(a, b), dim=-1)
		return ABSwapAverager.log_probabilities(self.classifier, a, b)

	@staticmethod
	def _attention_context() -> AbstractContextManager[object]:
		"""Require a fused SDPA kernel and restore backend flags after encoding."""
		return sdpa_kernel(
			[
				SDPBackend.FLASH_ATTENTION,
				SDPBackend.EFFICIENT_ATTENTION,
				SDPBackend.CUDNN_ATTENTION,
			]
		)

	@contextmanager
	def _backbone_context(self) -> Iterator[None]:
		"""Scope fused attention; adapter input dtypes are fixed at initialization."""
		with self._attention_context():
			yield

	def _checkpoint_contexts(
		self,
	) -> tuple[AbstractContextManager[object], AbstractContextManager[object]]:
		"""Restore fused attention for checkpoint forward and recomputation."""
		# Checkpoint forwards inherit the outer attention context; backward runs later.
		return self._attention_context(), self._backbone_context()
