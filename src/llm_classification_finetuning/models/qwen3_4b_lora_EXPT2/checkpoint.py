"""Atomically store portable adapter/head checkpoints without base weights."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch import Tensor

from ...config import GPUConfig, LoRAModelConfig
from ...errors import ModelExecutionError
from .heads import EXPT2, PAIRWISE_MLP, create_classification_head
from .model import LoRAQwenClassifier


@dataclass(frozen=True, slots=True)
class LoRACheckpointCompatibility:
	"""Model identity and training settings recorded with adapters and head."""

	model_name: str
	max_length: int
	serializer_version: str
	classifier_hidden_size: int
	classifier_dropout: float
	lora_rank: int
	lora_alpha: int
	lora_dropout: float
	lora_target_modules: tuple[str, ...]
	ab_swap: str
	schema_version: int = 2
	class_count: int = 3
	pooling: str = "last-nonpadding-token"
	truncation_side: str = "right"
	bias: str = "none"
	training_quantization: str = "none"
	head_architecture: str = EXPT2

	@classmethod
	def from_config(
		cls,
		config: LoRAModelConfig,
		serializer_version: str,
		*,
		quantize_training: bool = False,
		head_architecture: str = EXPT2,
	) -> LoRACheckpointCompatibility:
		return cls(
			model_name=config.model_name,
			max_length=config.max_length,
			serializer_version=serializer_version,
			classifier_hidden_size=config.hidden_size,
			classifier_dropout=config.dropout,
			lora_rank=config.lora_rank,
			lora_alpha=config.lora_alpha,
			lora_dropout=config.lora_dropout,
			lora_target_modules=tuple(sorted(config.lora_target_modules)),
			ab_swap=config.ab_swap,
			training_quantization="bnb-nf4-double" if quantize_training else "none",
			head_architecture=head_architecture,
		)


@dataclass(frozen=True, slots=True)
class LoadedLoRACheckpoint:
	"""Validated CPU tensors and the exact Hub base revision used for training."""

	adapter_state_dict: dict[str, Tensor]
	head_state_dict: dict[str, Tensor]
	backbone_hidden_size: int
	base_revision: str | None


class LoRACheckpointStore:
	"""Publish one complete checkpoint and reject partial or incompatible weights."""

	def __init__(
		self, path: Path, build_command: str, *, model_label: str = "LoRA"
	) -> None:
		self.path = path.resolve()
		self._build_command = build_command
		self._model_label = model_label

	def save(
		self,
		model: LoRAQwenClassifier,
		compatibility: LoRACheckpointCompatibility,
		*,
		training_source_sha256: str,
		gpu: GPUConfig,
	) -> None:
		"""Stage adapters, head, and metadata together, then atomically replace."""
		temporary_path: Path | None = None
		try:
			adapters = self._cpu_state(
				get_peft_model_state_dict(model.backbone, save_embedding_layers=False)
			)
			head = self._cpu_state(model.classifier.state_dict())
			self._validate_state(adapters, "adapter_state_dict")
			self._validate_state(head, "head_state_dict")
			self.path.parent.mkdir(parents=True, exist_ok=True)
			with tempfile.NamedTemporaryFile(
				dir=self.path.parent, prefix=".adapter-", suffix=".tmp.pt", delete=False
			) as temporary_file:
				temporary_path = Path(temporary_file.name)
			torch.save(
				{
					"metadata": {
						**asdict(compatibility),
						"backbone_hidden_size": model.backbone_hidden_size,
						"base_revision": model.base_revision,
						"training_source_sha256": training_source_sha256,
						"training_gpu": asdict(gpu),
						"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
					},
					"adapter_state_dict": adapters,
					"head_state_dict": head,
				},
				temporary_path,
			)
			os.replace(temporary_path, self.path)
			temporary_path = None
		except Exception as error:
			raise ModelExecutionError(
				f"Could not save {self._model_label} checkpoint {self.path}: {error}"
			) from error
		finally:
			if temporary_path is not None:
				try:
					temporary_path.unlink(missing_ok=True)
				except OSError:
					pass

	def load(self, expected: LoRACheckpointCompatibility) -> LoadedLoRACheckpoint:
		"""Validate training identity without imposing the original GPU settings."""
		if not self.path.is_file():
			raise ModelExecutionError(
				f"{self._model_label} checkpoint not found: {self.path}. Run '{self._build_command}' "
				"with the matching --config first."
			)
		try:
			checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
			if not isinstance(checkpoint, dict) or not isinstance(
				checkpoint.get("metadata"), dict
			):
				raise TypeError("checkpoint and metadata must be mappings")
			# Checkpoints predating QLoRA were trained with a non-quantized base.
			metadata = {
				"training_quantization": "none",
				"head_architecture": PAIRWISE_MLP,
				**checkpoint["metadata"],
			}
			mismatches = [
				name
				for name, value in asdict(expected).items()
				if name not in {"max_length", "ab_swap"} and metadata.get(name) != value
			]
			if mismatches:
				raise ValueError(
					f"incompatible configuration ({', '.join(mismatches)}). "
					f"Run '{self._build_command}' with the matching --config "
					"to rebuild the checkpoint."
				)
			hidden_size = metadata.get("backbone_hidden_size")
			if (
				isinstance(hidden_size, bool)
				or not isinstance(hidden_size, int)
				or hidden_size <= 0
			):
				raise ValueError("backbone_hidden_size must be a positive integer")
			revision = metadata.get("base_revision")
			if revision is not None and (
				not isinstance(revision, str) or not revision.strip()
			):
				raise ValueError("base_revision must be non-empty text or null")
			head_state = self._validate_state(
				checkpoint.get("head_state_dict"), "head_state_dict"
			)
			with torch.device("meta"):
				expected_head = create_classification_head(
					hidden_size,
					expected.classifier_hidden_size,
					expected.classifier_dropout,
					compute_dtype=torch.float32,
					head_architecture=expected.head_architecture,
				).state_dict()
			if head_state.keys() != expected_head.keys() or any(
				head_state[name].shape != parameter.shape
				for name, parameter in expected_head.items()
			):
				raise ValueError(
					"classification head tensors do not match the required architecture. "
					f"Run '{self._build_command}' to rebuild the checkpoint."
				)
			return LoadedLoRACheckpoint(
				self._validate_state(
					checkpoint.get("adapter_state_dict"), "adapter_state_dict"
				),
				head_state,
				hidden_size,
				revision,
			)
		except Exception as error:
			raise ModelExecutionError(
				f"Could not load {self._model_label} checkpoint {self.path}: {error}"
			) from error

	def restore(
		self, model: LoRAQwenClassifier, checkpoint: LoadedLoRACheckpoint
	) -> None:
		"""Require a complete adapter tensor set before assigning any weights."""
		try:
			if model.backbone_hidden_size != checkpoint.backbone_hidden_size:
				raise ValueError("backbone hidden size does not match")
			if model.base_revision != checkpoint.base_revision:
				raise ValueError("backbone revision does not match")
			expected = get_peft_model_state_dict(
				model.backbone, save_embedding_layers=False
			)
			if expected.keys() != checkpoint.adapter_state_dict.keys():
				raise ValueError(
					"checkpoint does not contain the exact configured adapter tensor set"
				)
			for name, value in expected.items():
				if value.shape != checkpoint.adapter_state_dict[name].shape:
					raise ValueError(f"adapter shape does not match for {name}")
			set_peft_model_state_dict(model.backbone, checkpoint.adapter_state_dict)
			model.classifier.load_state_dict(checkpoint.head_state_dict, strict=True)
		except Exception as error:
			raise ModelExecutionError(
				f"Could not restore {self._model_label} checkpoint {self.path}: {error}"
			) from error

	@staticmethod
	def _cpu_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
		return {
			name: value.detach().to(device="cpu").contiguous()
			for name, value in state.items()
		}

	@staticmethod
	def _validate_state(value: Any, name: str) -> dict[str, Tensor]:
		if not isinstance(value, dict) or not value:
			raise ValueError(f"{name} must be a non-empty tensor mapping")
		if not all(
			isinstance(key, str) and isinstance(tensor, Tensor)
			for key, tensor in value.items()
		):
			raise ValueError(f"{name} must contain named tensors")
		if not all(torch.isfinite(tensor).all() for tensor in value.values()):
			raise ValueError(f"{name} contains non-finite weights")
		return value
