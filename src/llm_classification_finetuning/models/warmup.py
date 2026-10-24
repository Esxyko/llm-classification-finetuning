"""Build reusable non-quantized head warmup from the saved training partition."""

from __future__ import annotations

import gc

import torch

from ..config import AppConfig
from ..errors import ConfigurationError, ModelExecutionError
from .common import GPUEnvironment, WarmupBuildResult
from .qwen3_4b_lora import LoRATrainer, LoRATrainingBatchRepository


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
		self._config = config
		self._model_config = config.warmup

	def run(self, *, refresh_cache: bool = False) -> WarmupBuildResult:
		"""Prepare train-only tokens and publish completed warmup, or reuse a hit."""
		try:
			batches = LoRATrainingBatchRepository.from_config(
				self._config, refresh_cache=refresh_cache, warmup_only=True
			)
			device = GPUEnvironment(self._config.gpu).configure()
			trainer = LoRATrainer(
				self._model_config,
				self._config.gpu,
				device,
				token_cache_dir=self._config.data.processed_path.parent / "token_cache",
				refresh_cache=refresh_cache,
				warmup_only=True,
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
			gc.collect()
			if torch.cuda.is_initialized():
				torch.cuda.empty_cache()
