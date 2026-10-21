"""Persist completed head warmup independently of joint-training settings."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW

from ...config import GPUConfig, LoRAModelConfig
from ...errors import ModelExecutionError
from ..common import CacheStorage, PairwiseClassificationHead


@dataclass(frozen=True, slots=True)
class WarmupCacheContext:
	"""Training IDs and immutable encoding identities supplied by batch preparation."""

	training_ids_sha256: str
	serializer_version: str
	base_revision: str | None
	local_weights_sha256: str | None
	backbone_hidden_size: int
	tokenizer_key: str


@dataclass(frozen=True, slots=True)
class WarmupCacheEntry:
	"""A checked cache lookup, including misses, reusable throughout one fit."""

	metadata: dict[str, Any]
	state: dict[str, Any] | None


class HeadWarmupCache:
	"""Atomically store FP32 heads, named AdamW moments, scaling and RNG state."""

	SCHEMA_VERSION = 3

	def __init__(
		self,
		root: Path,
		config: LoRAModelConfig,
		gpu: GPUConfig,
		device: torch.device,
		*,
		quantize_training: bool = False,
		model_label: str = "LoRA",
	) -> None:
		self.root = root
		self._config = config
		self._gpu = gpu
		self._device = device
		self._quantize_training = quantize_training
		self._model_label = model_label

	def path(self, entry: WarmupCacheEntry) -> Path:
		key = hashlib.sha256(
			json.dumps(entry.metadata, sort_keys=True, separators=(",", ":")).encode(
				"utf-8"
			)
		).hexdigest()
		return self.root / f"{key}.pt"

	def prepare(
		self, context: WarmupCacheContext, *, seed: int, refresh: bool = False
	) -> WarmupCacheEntry:
		"""Load and fully check an optional cache without changing global RNG state."""
		config = self._config
		metadata = {
			**asdict(context),
			"schema_version": self.SCHEMA_VERSION,
			"model_name": config.model_name
			if context.local_weights_sha256 is None
			else "local-content",
			"training_quantization": "bnb-nf4-double"
			if self._quantize_training
			else "none",
			"classifier_hidden_size": config.hidden_size,
			"classifier_dropout": config.dropout,
			"ab_swap": config.ab_swap,
			"warmup_epochs": config.warmup_epochs,
			"warmup_training_batch_size": config.warmup_training_batch_size,
			"warmup_learning_rate": config.warmup_learning_rate,
			"seed": seed,
			"weight_decay": config.weight_decay,
			"max_grad_norm": config.max_grad_norm,
		}
		entry = WarmupCacheEntry(metadata, None)
		if refresh:
			return entry
		try:
			with CacheStorage(self.path(entry)).open("r") as session:
				payload = torch.load(
					session.path, map_location="cpu", weights_only=True
				)
			self._validate(payload, metadata)
			return WarmupCacheEntry(metadata, payload)
		except (
			OSError,
			ValueError,
			TypeError,
			RuntimeError,
			EOFError,
			pickle.UnpicklingError,
			KeyError,
			IndexError,
			OverflowError,
		):
			return entry

	def save(
		self,
		entry: WarmupCacheEntry,
		head: nn.Module,
		optimizer: AdamW,
		scaler: torch.amp.GradScaler,
	) -> None:
		"""Publish only after the final warmup validation has completed."""
		path = self.path(entry)
		try:
			numpy_state = np.random.get_state()
			payload = {
				"metadata": entry.metadata,
				"source_precision": self._gpu.precision,
				"head_state_dict": {
					name: self._cpu_tensor(value)
					for name, value in head.state_dict().items()
				},
				"optimizer_state": {
					name: {
						key: self._cpu_tensor(value)
						for key, value in optimizer.state.get(parameter, {}).items()
					}
					for name, parameter in head.named_parameters()
				},
				"scaler_state": scaler.state_dict(),
				"rng_state": {
					"python": random.getstate(),
					"numpy": (
						numpy_state[0],
						numpy_state[1].tolist(),
						numpy_state[2],
						numpy_state[3],
						numpy_state[4],
					),
					"torch": torch.get_rng_state(),
					"cuda": torch.cuda.get_rng_state(self._device)
					if self._device.type == "cuda"
					else None,
				},
			}
			self._validate(payload, entry.metadata)
			with CacheStorage(path).open("w") as session:
				torch.save(payload, session.path)
		except Exception as error:
			raise ModelExecutionError(
				f"Could not save {self._model_label} warmup cache {path}: {error}"
			) from error

	def restore(
		self,
		entry: WarmupCacheEntry,
		head: nn.Module,
		optimizer: AdamW,
		scaler: torch.amp.GradScaler,
	) -> None:
		"""Restore the head only, retaining fresh adapters and current optimizer groups."""
		state = entry.state
		if state is None:
			raise ValueError("Cannot restore a missing warmup cache.")
		parameters = dict(head.named_parameters())
		if parameters.keys() != state["head_state_dict"].keys() or any(
			parameter.shape != state["head_state_dict"][name].shape
			for name, parameter in parameters.items()
		):
			raise ModelExecutionError(
				"The model head does not match its warmup cache context."
			)
		head.load_state_dict(state["head_state_dict"], strict=True)
		for name, parameter in parameters.items():
			values = state["optimizer_state"][name]
			if values:
				# AdamW's non-capturable step counter stays on CPU; moments follow FP32 parameters.
				optimizer.state[parameter] = {
					key: value.clone()
					if key == "step"
					else value.to(parameter.device).clone()
					for key, value in values.items()
				}
		if state["source_precision"] == self._gpu.precision == "fp16":
			scaler.load_state_dict(state["scaler_state"])
		rng = state["rng_state"]
		random.setstate(rng["python"])
		np.random.set_state(self._numpy_state(rng["numpy"]))
		torch.set_rng_state(rng["torch"])
		if self._device.type == "cuda":
			torch.cuda.set_rng_state(rng["cuda"], self._device)

	def _validate(self, payload: Any, metadata: dict[str, Any]) -> None:
		"""Validate every restorable component before assigning live model state."""
		if not isinstance(payload, dict) or payload.get("metadata") != metadata:
			raise ValueError("Warmup metadata does not match.")
		precision = payload["source_precision"]
		if precision not in {"fp16", "bf16", "fp32"}:
			raise ValueError("Invalid warmup precision.")
		# Meta construction checks the shared architecture without allocating weights or drawing RNG.
		with torch.device("meta"):
			expected = PairwiseClassificationHead(
				metadata["backbone_hidden_size"],
				self._config.hidden_size,
				self._config.dropout,
			).state_dict()
		head = payload["head_state_dict"]
		moments = payload["optimizer_state"]
		if not isinstance(head, dict) or head.keys() != expected.keys():
			raise ValueError("Incomplete warmup head.")
		if not isinstance(moments, dict) or moments.keys() != expected.keys():
			raise ValueError("Incomplete warmup optimizer state.")
		initialized = [bool(values) for values in moments.values()]
		if any(initialized) and not all(initialized):
			raise ValueError("Partially initialized warmup optimizer state.")
		for name, parameter in expected.items():
			self._validate_tensor(head[name], parameter.shape)
			values = moments[name]
			if not isinstance(values, dict) or (
				values and values.keys() != {"step", "exp_avg", "exp_avg_sq"}
			):
				raise ValueError("Invalid AdamW state.")
			if values:
				self._validate_tensor(values["step"], torch.Size([]))
				step = values["step"].item()
				if step < 0 or not float(step).is_integer():
					raise ValueError("Invalid AdamW step.")
				for key in ("exp_avg", "exp_avg_sq"):
					self._validate_tensor(values[key], parameter.shape)
				if (values["exp_avg_sq"] < 0).any():
					raise ValueError("Invalid AdamW second moment.")
		self._validate_scaler(payload["scaler_state"], precision)
		rng = payload["rng_state"]
		if not isinstance(rng, dict):
			raise TypeError("Missing warmup RNG state.")
		random.Random().setstate(rng["python"])
		np.random.RandomState().set_state(self._numpy_state(rng["numpy"]))
		self._validate_rng(rng["torch"], torch.device("cpu"))
		if rng["cuda"] is not None:
			if self._device.type == "cuda":
				self._validate_rng(rng["cuda"], self._device)
			elif not isinstance(rng["cuda"], Tensor):
				raise ValueError("Invalid CUDA RNG state.")
		elif self._device.type == "cuda":
			raise ValueError("Missing CUDA RNG state.")

	@staticmethod
	def _cpu_tensor(value: Tensor) -> Tensor:
		return value.detach().to(device="cpu").contiguous().clone()

	@staticmethod
	def _validate_tensor(value: Any, shape: torch.Size) -> None:
		if (
			not isinstance(value, Tensor)
			or value.device.type != "cpu"
			or value.dtype != torch.float32
			or value.shape != shape
			or not torch.isfinite(value).all()
		):
			raise ValueError("Invalid FP32 warmup tensor.")

	@staticmethod
	def _validate_rng(value: Any, device: torch.device) -> None:
		if (
			not isinstance(value, Tensor)
			or value.dtype != torch.uint8
			or value.ndim != 1
		):
			raise ValueError("Invalid Torch RNG state.")
		torch.Generator(device=device).set_state(value)

	@staticmethod
	def _numpy_state(value: Any) -> tuple:
		if not isinstance(value, tuple) or len(value) != 5 or value[0] != "MT19937":
			raise ValueError("Invalid NumPy RNG state.")
		keys = value[1]
		if (
			not isinstance(keys, list)
			or len(keys) != 624
			or any(type(key) is not int or not 0 <= key <= 0xFFFFFFFF for key in keys)
			or type(value[2]) is not int
			or not 0 <= value[2] <= 624
			or type(value[3]) is not int
			or value[3] not in (0, 1)
			or not isinstance(value[4], (int, float))
			or not math.isfinite(value[4])
		):
			raise ValueError("Invalid NumPy RNG fields.")
		return value[0], np.array(keys, dtype=np.uint32), value[2], value[3], value[4]

	@staticmethod
	def _validate_scaler(state: Any, precision: str) -> None:
		if not isinstance(state, dict):
			raise TypeError("Invalid gradient scaler state.")
		if precision != "fp16":
			if state:
				raise ValueError("Unexpected gradient scaler state.")
			return
		if state.keys() != {
			"scale",
			"growth_factor",
			"backoff_factor",
			"growth_interval",
			"_growth_tracker",
		}:
			raise ValueError("Incomplete gradient scaler state.")
		for key in ("scale", "growth_factor", "backoff_factor"):
			if (
				type(state[key]) not in (float, int)
				or not math.isfinite(state[key])
				or state[key] <= 0
			):
				raise ValueError("Invalid gradient scaler value.")
		if state["growth_factor"] <= 1 or state["backoff_factor"] >= 1:
			raise ValueError("Invalid gradient scaler factors.")
		if (
			type(state["growth_interval"]) is not int
			or state["growth_interval"] <= 0
			or type(state["_growth_tracker"]) is not int
			or state["_growth_tracker"] < 0
		):
			raise ValueError("Invalid gradient scaler counters.")
