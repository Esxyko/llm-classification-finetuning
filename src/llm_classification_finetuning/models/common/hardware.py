"""Configure the project-selected CUDA device."""

from __future__ import annotations

from typing import ClassVar

import torch

from ...config import GPUConfig
from ...errors import ModelExecutionError


class GPUEnvironment:
	"""Validate and activate the configured CUDA device."""

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
		if not torch.cuda.is_available():
			raise ModelExecutionError(
				"CUDA is not available. Model execution requires a configured CUDA GPU."
			)

		device = torch.device(self._config.device)
		available_devices = torch.cuda.device_count()
		if device.index is None or device.index >= available_devices:
			raise ModelExecutionError(
				f"Configured GPU {device} is unavailable; found "
				f"{available_devices} CUDA device(s)."
			)

		torch.cuda.set_device(device)
		if self._config.precision == "bf16" and not torch.cuda.is_bf16_supported():
			raise ModelExecutionError(
				f"Configured GPU {device} does not support bf16. Set "
				"gpu.precision to fp16 or fp32."
			)
		torch.backends.cuda.matmul.allow_tf32 = self._config.allow_tf32
		return device
