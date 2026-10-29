"""Persist portable adapter/head checkpoints and merged backbone exports."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors import safe_open
from torch import Tensor
from transformers import (
	AutoConfig,
	AutoTokenizer,
	PreTrainedModel,
	PreTrainedTokenizerBase,
)

from ...config import GPUConfig, LoRAModelConfig
from ...errors import ModelExecutionError
from .heads import PAIRWISE_MLP, SR2, create_classification_head
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
	head_architecture: str = SR2

	@classmethod
	def from_config(
		cls,
		config: LoRAModelConfig,
		serializer_version: str,
		*,
		quantize_training: bool = False,
		head_architecture: str = SR2,
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

	def check_writable(self) -> None:
		"""Probe the destination before training without touching existing weights."""
		try:
			self.path.parent.mkdir(parents=True, exist_ok=True)
			if self.path.exists():
				if not self.path.is_file():
					raise ValueError("checkpoint destination must be a regular file")
				if not os.access(self.path, os.W_OK):
					raise PermissionError("existing checkpoint is not writable")
			with tempfile.NamedTemporaryFile(
				dir=self.path.parent, prefix=".adapter-", suffix=".write-check"
			) as probe:
				probe.write(b"\0")
				probe.flush()
		except Exception as error:
			raise ModelExecutionError(
				f"Cannot write {self._model_label} checkpoint {self.path}: {error}"
			) from error

	def save(
		self,
		model: LoRAQwenClassifier,
		compatibility: LoRACheckpointCompatibility,
		*,
		training_source_sha256: str,
		gpu: GPUConfig,
	) -> None:
		"""Reload and validate staged CPU tensors before atomic publication."""
		temporary_path: Path | None = None
		try:
			adapters = self._cpu_state(
				get_peft_model_state_dict(model.backbone, save_embedding_layers=False)
			)
			head = self._cpu_state(model.classifier.state_dict())
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
			# Release the CPU snapshot before loading a second copy for validation.
			del adapters, head
			self._load_checkpoint(temporary_path, compatibility)
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
		return self._load_checkpoint(self.path, expected)

	def _load_checkpoint(
		self, path: Path, expected: LoRACheckpointCompatibility
	) -> LoadedLoRACheckpoint:
		"""Apply identical CPU loading checks to staged and published checkpoints."""
		if not path.is_file():
			raise ModelExecutionError(
				f"{self._model_label} checkpoint not found: {path}. Run '{self._build_command}' "
				"with the matching --config first."
			)
		try:
			checkpoint = torch.load(path, map_location="cpu", weights_only=True)
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
				f"Could not load {self._model_label} checkpoint {path}: {error}"
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


class MergedBackboneStore:
	"""Export a trained backbone after saving its independent adapter/head checkpoint."""

	WINDOWS_RENAME_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8)

	def __init__(self, path: Path) -> None:
		# Resolve the parent only so destination links remain visible to preflight.
		self.path = path.parent.resolve() / path.name
		self._backup_path = self.path.with_name(f".{self.path.name}-backup")

	def check_writable(self) -> None:
		"""Check the export destination and recover an interrupted directory promotion."""
		try:
			self._prepare_destination()
			if self.path.exists() and not os.access(self.path, os.W_OK):
				raise PermissionError(
					"existing merged backbone directory is not writable"
				)
			with tempfile.TemporaryDirectory(
				dir=self.path.parent, prefix=f".{self.path.name}-write-check-"
			) as directory:
				(Path(directory) / "probe").write_bytes(b"\0")
		except Exception as error:
			raise ModelExecutionError(
				f"Cannot write merged backbone {self.path}: {error}"
			) from error

	def save(
		self,
		model: LoRAQwenClassifier,
		tokenizer: PreTrainedTokenizerBase,
		*,
		model_name: str,
		training_source_sha256: str,
		checkpoint_path: Path,
	) -> Path:
		"""Consume the trained PEFT backbone, staging all files before replacement."""
		staging_path: Path | None = None
		try:
			self._prepare_destination()
			with checkpoint_path.open("rb") as checkpoint_file:
				checkpoint_sha256 = hashlib.file_digest(
					checkpoint_file, "sha256"
				).hexdigest()
			staging_path = Path(
				tempfile.mkdtemp(
					dir=self.path.parent, prefix=f".{self.path.name}-stage-"
				)
			)
			print(f"Exporting merged backbone: {self.path}.", flush=True)
			model.eval()
			with torch.no_grad():
				backbone = model.backbone.to(device="cpu").merge_and_unload(
					safe_merge=True
				)
				if any(
					isinstance(module, BaseTunerLayer) for module in backbone.modules()
				):
					raise ValueError("merged backbone still contains adapter layers")
				# Store AutoModel keys, without restoring the source reward-head prefixes.
				backbone.save_pretrained(
					staging_path, max_shard_size="5GB", save_original_format=False
				)
			tokenizer.save_pretrained(staging_path)
			metadata = {
				"model_name": model_name,
				"base_revision": model.base_revision,
				"training_source_sha256": training_source_sha256,
				"checkpoint_file": checkpoint_path.name,
				"checkpoint_sha256": checkpoint_sha256,
				"dtype": str(backbone.dtype).removeprefix("torch."),
				"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
			}
			(staging_path / "export_metadata.json").write_text(
				json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
				encoding="utf-8",
			)
			self._validate_export(staging_path, backbone, tokenizer)
			self._publish(staging_path)
			staging_path = None
			return self.path
		except Exception as error:
			raise ModelExecutionError(
				f"Could not export merged backbone to {self.path}: {error}. "
				f"The saved adapter/head checkpoint remains available at {checkpoint_path}."
			) from error
		finally:
			if staging_path is not None:
				try:
					shutil.rmtree(staging_path)
				except FileNotFoundError:
					pass
				except OSError as error:
					with suppress(Exception):
						print(
							f"Could not remove merged backbone staging directory "
							f"{staging_path}: {error}",
							file=sys.stderr,
							flush=True,
						)

	def _prepare_destination(self) -> None:
		self.path.parent.mkdir(parents=True, exist_ok=True)
		for directory in (self.path, self._backup_path):
			if directory.is_symlink() or directory.is_junction():
				raise ValueError(
					f"merged backbone directory cannot be a link: {directory}"
				)
			if directory.exists() and not directory.is_dir():
				raise ValueError(
					f"merged backbone path is not a directory: {directory}"
				)
		if self._backup_path.exists():
			if self.path.exists():
				raise ValueError(
					f"previous merged backbone backup needs review: {self._backup_path}; "
					f"current export: {self.path}"
				)
			self._replace_directory(self._backup_path, self.path)

	@staticmethod
	def _validate_export(
		path: Path, backbone: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
	) -> None:
		"""Validate local config/tokenizer and every shard without reloading weights."""
		config = AutoConfig.from_pretrained(path, local_files_only=True)
		if (
			config.model_type != backbone.config.model_type
			or config.hidden_size != backbone.config.hidden_size
		):
			raise ValueError("exported model configuration does not match the backbone")
		restored_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
		if (
			restored_tokenizer.get_vocab() != tokenizer.get_vocab()
			or restored_tokenizer.special_tokens_map != tokenizer.special_tokens_map
			or restored_tokenizer.chat_template != tokenizer.chat_template
		):
			raise ValueError("exported tokenizer does not match the training tokenizer")

		expected = backbone.state_dict()
		if not expected or any(
			part.startswith("lora_") for name in expected for part in name.split(".")
		):
			raise ValueError(
				"export must contain complete backbone weights without LoRA"
			)
		index_path = path / "model.safetensors.index.json"
		weight_map: dict[str, str] | None = None
		if index_path.is_file():
			index = json.loads(index_path.read_text(encoding="utf-8"))
			weight_map = index.get("weight_map")
			if (
				not isinstance(weight_map, dict)
				or weight_map.keys() != expected.keys()
				or not all(isinstance(name, str) for name in weight_map.values())
			):
				raise ValueError(
					"exported shard index does not match the backbone weights"
				)
			filenames = set(weight_map.values())
		else:
			filenames = {"model.safetensors"}
		if any(Path(name).name != name for name in filenames) or filenames != {
			file.name for file in path.glob("*.safetensors")
		}:
			raise ValueError("exported weight files do not match the shard index")
		seen: set[str] = set()
		for filename in sorted(filenames):
			with safe_open(path / filename, framework="pt", device="cpu") as weights:
				for name in sorted(weights.keys()):
					if name not in expected or name in seen:
						raise ValueError(
							f"unexpected or duplicated backbone tensor: {name}"
						)
					if weight_map is not None and weight_map[name] != filename:
						raise ValueError(
							f"backbone tensor is in the wrong shard: {name}"
						)
					if (
						tuple(weights.get_slice(name).get_shape())
						!= expected[name].shape
					):
						raise ValueError(
							f"exported backbone tensor shape does not match: {name}"
						)
					if weights.get_tensor(name).dtype != expected[name].dtype:
						raise ValueError(
							f"exported backbone tensor dtype does not match: {name}"
						)
					seen.add(name)
		if seen != expected.keys():
			raise ValueError("export is missing backbone tensors")

	def _publish(self, staging_path: Path) -> None:
		previous_moved = False
		if self.path.exists():
			self._replace_directory(self.path, self._backup_path)
			previous_moved = True
		try:
			self._replace_directory(staging_path, self.path)
		except OSError as error:
			if previous_moved:
				try:
					self._replace_directory(self._backup_path, self.path)
				except OSError as restore_error:
					raise ModelExecutionError(
						f"Could not publish to {self.path} or restore the previous merged "
						f"backbone. Previous export remains at {self._backup_path}. "
						f"Publish error: {error}; restore error: {restore_error}"
					) from restore_error
			raise
		if previous_moved:
			try:
				shutil.rmtree(self._backup_path)
			except OSError as error:
				raise ModelExecutionError(
					f"New merged backbone is available at {self.path}, but the previous "
					f"export backup could not be removed from {self._backup_path}: {error}"
				) from error

	@classmethod
	def _replace_directory(cls, source: Path, destination: Path) -> None:
		"""Retry transient Windows sharing errors without changing rename semantics."""
		for delay in (*cls.WINDOWS_RENAME_RETRY_DELAYS, None):
			try:
				os.replace(source, destination)
				return
			except OSError as error:
				if os.name != "nt" or error.winerror not in (5, 32) or delay is None:
					raise
				time.sleep(delay)
