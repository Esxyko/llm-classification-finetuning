"""Publish independent frozen-SR2 heads after validating staged CPU weights."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from ...config import GPUConfig, SR2ModelConfig
from ...errors import ModelExecutionError
from ..SR2_Qwen3_4B_lora.heads import create_classification_head
from ..SR2_Qwen3_4B_lora.model import LoRAQwenClassifier
from ..SR2_Qwen3_4B_lora.warmup_cache import WarmupCacheContext
from ..SR2_Qwen3_4B_lora.warmup_feature_cache import WarmupFeatureCache


class SR2HeadCheckpointStore:
	"""Save FP32 head parameters without adapters, base weights, or optimizer state."""

	SCHEMA_VERSION = 1

	def __init__(self, path: Path) -> None:
		self.path = path.resolve()

	def check_writable(self) -> None:
		"""Probe the head destination before configuring CUDA or loading weights."""
		try:
			self.path.parent.mkdir(parents=True, exist_ok=True)
			if self.path.exists():
				if not self.path.is_file():
					raise ValueError("checkpoint destination must be a regular file")
				if not os.access(self.path, os.W_OK):
					raise PermissionError("existing checkpoint is not writable")
			with tempfile.NamedTemporaryFile(
				dir=self.path.parent, prefix=".head-", suffix=".write-check"
			) as probe:
				probe.write(b"\0")
				probe.flush()
		except Exception as error:
			raise ModelExecutionError(
				f"Cannot write SR2 head checkpoint {self.path}: {error}"
			) from error

	def save(
		self,
		model: LoRAQwenClassifier,
		config: SR2ModelConfig,
		gpu: GPUConfig,
		*,
		context: WarmupCacheContext,
		training_source_sha256: str,
	) -> None:
		"""Read back the staged head before replacing the previous artifact."""
		temporary_path: Path | None = None
		try:
			if not model.warmup_only or any(
				parameter.requires_grad for parameter in model.backbone.parameters()
			):
				raise ValueError(
					"SR2 head checkpoints require an adapter-free frozen backbone"
				)
			metadata = {
				**asdict(context),
				"schema_version": self.SCHEMA_VERSION,
				"model_name": config.model_name,
				"head_architecture": model.head_architecture,
				"class_count": 3,
				"pooling": WarmupFeatureCache.POOLING,
				"training_source_sha256": training_source_sha256,
				"training_config": asdict(config),
				"training_gpu": asdict(gpu),
				"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
			}
			self.path.parent.mkdir(parents=True, exist_ok=True)
			with tempfile.NamedTemporaryFile(
				dir=self.path.parent, prefix=".head-", suffix=".tmp.pt", delete=False
			) as temporary_file:
				temporary_path = Path(temporary_file.name)
			torch.save(
				{
					"metadata": metadata,
					"head_state_dict": {
						name: parameter.detach().to(device="cpu", dtype=torch.float32)
						for name, parameter in model.classifier.state_dict().items()
					},
				},
				temporary_path,
			)
			self._validate(temporary_path, metadata, config)
			os.replace(temporary_path, self.path)
			temporary_path = None
		except Exception as error:
			raise ModelExecutionError(
				f"Could not save SR2 head checkpoint {self.path}: {error}"
			) from error
		finally:
			if temporary_path is not None:
				with suppress(OSError):
					temporary_path.unlink(missing_ok=True)

	@staticmethod
	def _validate(path: Path, metadata: dict[str, Any], config: SR2ModelConfig) -> None:
		checkpoint = torch.load(path, map_location="cpu", weights_only=True)
		if not isinstance(checkpoint, dict) or checkpoint.get("metadata") != metadata:
			raise ValueError(
				"staged checkpoint metadata does not match the trained model"
			)
		state = checkpoint.get("head_state_dict")
		with torch.device("meta"):
			expected = create_classification_head(
				metadata["backbone_hidden_size"],
				config.hidden_size,
				config.dropout,
				compute_dtype=torch.float32,
				head_architecture=metadata["head_architecture"],
			).state_dict()
		if not isinstance(state, dict) or state.keys() != expected.keys():
			raise ValueError(
				"staged classification head has incompatible parameter names"
			)
		for name, parameter in expected.items():
			value = state[name]
			if (
				not isinstance(value, torch.Tensor)
				or value.dtype != torch.float32
				or value.shape != parameter.shape
				or not bool(torch.isfinite(value).all())
			):
				raise ValueError(
					f"staged classification head has an invalid tensor: {name}"
				)
