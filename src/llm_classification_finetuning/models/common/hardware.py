"""Configure CUDA devices, monitor peak memory, and map submission layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from ...config import GPUConfig
from ...errors import ModelExecutionError

if TYPE_CHECKING:
	from transformers import PretrainedConfig


class GPUEnvironment:
	"""Validate training or submission devices without changing compute precision."""

	PRECISION_DTYPES: ClassVar[dict[str, torch.dtype]] = {
		"bf16": torch.bfloat16,
		"fp16": torch.float16,
		"fp32": torch.float32,
	}

	def __init__(self, config: GPUConfig) -> None:
		self._config = config

	@property
	def dtype(self) -> torch.dtype:
		"""Return the configured backbone weight dtype."""
		return self.PRECISION_DTYPES[self._config.precision]

	def configure(self) -> torch.device:
		"""Validate and configure the requested single CUDA device."""
		return self._configure_devices((self._config.device,))[0]

	def configure_submit(self) -> tuple[torch.device, torch.device]:
		"""Validate both submission GPUs and activate the first for inputs and head."""
		devices = self._configure_devices(self._config.submit_devices)
		return devices[0], devices[1]

	@staticmethod
	def submission_device_map(
		config: PretrainedConfig, devices: tuple[torch.device, torch.device]
	) -> dict[str, str]:
		"""Place contiguous, complete Qwen3 decoder layers across exactly two GPUs."""
		if config.model_type != "qwen3" or config.num_hidden_layers < 2:
			raise ModelExecutionError(
				"Dual-GPU submission requires a Qwen3 backbone with at least two decoder layers."
			)
		first, second = (str(device) for device in devices)
		split = config.num_hidden_layers // 2
		return {
			"embed_tokens": first,
			"rotary_emb": first,
			**{
				f"layers.{index}": first if index < split else second
				for index in range(config.num_hidden_layers)
			},
			"norm": second,
		}

	def _configure_devices(self, names: tuple[str, ...]) -> tuple[torch.device, ...]:
		if not torch.cuda.is_available():
			raise ModelExecutionError(
				"CUDA is not available. Model execution requires a configured CUDA GPU."
			)

		devices = tuple(torch.device(name) for name in names)
		if len(set(devices)) != len(devices):
			raise ModelExecutionError("Submission GPUs must be distinct CUDA devices.")
		available_devices = torch.cuda.device_count()
		for device in devices:
			if (
				device.type != "cuda"
				or device.index is None
				or device.index >= available_devices
			):
				raise ModelExecutionError(
					f"Configured GPU {device} is unavailable; found "
					f"{available_devices} CUDA device(s)."
				)
			with torch.cuda.device(device):
				if (
					self._config.precision == "bf16"
					and not torch.cuda.is_bf16_supported()
				):
					raise ModelExecutionError(
						f"Configured GPU {device} does not support bf16. Set "
						"gpu.precision to fp16 or fp32."
					)

		torch.cuda.set_device(devices[0])
		torch.backends.cuda.matmul.allow_tf32 = self._config.allow_tf32
		return devices


@dataclass(frozen=True, slots=True)
class GPUMemoryPeak:
	"""Absolute allocator peaks in bytes for one device and model run."""

	device: str
	peak_allocated_bytes: int
	peak_reserved_bytes: int


class GPUMemoryMonitor:
	"""Track one model run's allocator peaks without clearing cached memory."""

	def __init__(self, config: GPUConfig) -> None:
		self._config = config
		self._device: torch.device | None = None

	def start(self) -> None:
		"""Validate the configured GPU, finish prior work, and reset peaks once."""
		if self._device is not None:
			raise ModelExecutionError("GPU memory monitoring has already started.")
		device = GPUEnvironment(self._config).configure()
		try:
			torch.cuda.synchronize(device)
			torch.cuda.reset_peak_memory_stats(device)
		except Exception as error:
			raise ModelExecutionError(
				f"Could not start GPU memory monitoring on {device}: {error}"
			) from error
		self._device = device

	def snapshot(self, *, synchronize: bool = True) -> GPUMemoryPeak:
		"""Read peaks, optionally skipping synchronization after execution fails."""
		if self._device is None:
			raise ModelExecutionError("GPU memory monitoring has not started.")
		try:
			if synchronize:
				torch.cuda.synchronize(self._device)
			return GPUMemoryPeak(
				device=str(self._device),
				peak_allocated_bytes=torch.cuda.max_memory_allocated(self._device),
				peak_reserved_bytes=torch.cuda.max_memory_reserved(self._device),
			)
		except Exception as error:
			raise ModelExecutionError(
				f"Could not read GPU memory peaks on {self._device}: {error}"
			) from error
