"""Build reusable non-quantized head warmup from the saved training partition."""

from __future__ import annotations

import gc
from contextlib import suppress
from dataclasses import replace

import torch

from ..config import AppConfig
from ..errors import ConfigurationError, ModelExecutionError
from .common import GPUEnvironment, GPUMemoryMonitor, WarmupBuildResult
from .SR2_Qwen3_4B_lora import PROFILE, LoRATrainer, LoRATrainingBatchRepository


class WarmupModelPipeline:
	"""Reuse adapter training infrastructure without adapters or joint training."""

	def __init__(self, config: AppConfig) -> None:
		if config.warmup is None:
			raise ConfigurationError(
				"Model warmup requires a 'warmup' section in the selected configuration. "
				"Select a matching YAML with --config."
			)
		if config.warmup.warmup_epochs <= 0:
			raise ConfigurationError(
				"Standalone warmup requires warmup.warmup_epochs > 0."
			)
		self._model_config = replace(config.warmup, ab_swap=PROFILE.ab_swap_override)
		self._config = replace(config, warmup=self._model_config)

	def run(
		self,
		*,
		refresh_cache: bool = False,
		gpu_memory_monitor: GPUMemoryMonitor | None = None,
	) -> WarmupBuildResult:
		"""Build or reuse train-partition warmup and evaluate both saved partitions."""
		try:
			batches = LoRATrainingBatchRepository.from_config(
				self._config,
				refresh_cache=refresh_cache,
				profile=PROFILE,
				warmup_only=True,
			)
			if gpu_memory_monitor is not None:
				gpu_memory_monitor.start()
			device = GPUEnvironment(self._config.gpu).configure()
			trainer = LoRATrainer(
				self._model_config,
				self._config.gpu,
				device,
				token_cache_dir=self._config.cache_dir / "token_cache",
				refresh_cache=refresh_cache,
				profile=PROFILE,
				warmup_only=True,
				gpu_memory_monitor=gpu_memory_monitor,
			)
			return trainer.build_warmup(batches)
		except torch.cuda.OutOfMemoryError as error:
			raise ModelExecutionError(
				"CUDA ran out of memory during warmup. Lower "
				"warmup.extraction_budget or warmup.warmup_training_batch_size, "
				"or use fp16/bf16 instead of fp32."
			) from error
		except ModelExecutionError:
			raise
		except Exception as error:
			raise ModelExecutionError(f"Could not execute warmup: {error}") from error
		finally:
			with suppress(Exception):
				gc.collect()
			with suppress(Exception):
				if torch.cuda.is_initialized():
					torch.cuda.empty_cache()
