"""Configure the project-selected single CUDA device."""

from __future__ import annotations

from typing import ClassVar

import torch

from ..config import GPUConfig
from ..errors import BaselineError


class GPUEnvironment:
    """Validate and activate one configured CUDA device."""

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
        """Validate, select, and configure the requested CUDA device."""
        if not torch.cuda.is_available():
            raise BaselineError(
                "CUDA is not available. The baseline requires one configured CUDA GPU."
            )

        device = torch.device(self._config.device)
        device_index = device.index if device.index is not None else 0
        if device_index >= torch.cuda.device_count():
            raise BaselineError(
                f"Configured GPU {self._config.device} is unavailable; found "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )
        torch.cuda.set_device(device)
        if self._config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise BaselineError(
                "The configured GPU does not support bf16. Set gpu.precision to "
                "fp16 or fp32."
            )
        torch.backends.cuda.matmul.allow_tf32 = self._config.allow_tf32
        return device
