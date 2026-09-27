"""Configure the project-selected CUDA devices."""

from __future__ import annotations

from typing import ClassVar

import torch

from ..config import GPUConfig
from ..errors import BaselineError


class GPUEnvironment:
    """Validate the configured CUDA devices and activate the primary one."""

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

    def configure(self) -> tuple[torch.device, ...]:
        """Validate and configure the requested CUDA devices."""
        if not torch.cuda.is_available():
            raise BaselineError(
                "CUDA is not available. The baseline requires configured CUDA GPUs."
            )

        devices = tuple(torch.device(value) for value in self._config.devices)
        available_devices = torch.cuda.device_count()
        unavailable = [
            str(device)
            for device in devices
            if device.index is None or device.index >= available_devices
        ]
        if unavailable:
            configured = ", ".join(unavailable)
            raise BaselineError(
                f"Configured GPU(s) {configured} are unavailable; found "
                f"{available_devices} CUDA device(s)."
            )

        for device in devices:
            torch.cuda.set_device(device)
            if (
                self._config.precision == "bf16"
                and not torch.cuda.is_bf16_supported()
            ):
                raise BaselineError(
                    f"Configured GPU {device} does not support bf16. Set "
                    "gpu.precision to fp16 or fp32."
                )

        torch.cuda.set_device(devices[0])
        torch.backends.cuda.matmul.allow_tf32 = self._config.allow_tf32
        return devices
