"""Validate and build a frozen Skywork backbone with the shared SR2 head."""

from __future__ import annotations

import gc
from contextlib import suppress
from pathlib import Path

import torch

from ...config import AppConfig, SR2ModelConfig
from ...errors import ModelExecutionError
from ..common import (
	GPUEnvironment,
	GPUMemoryMonitor,
	ModelBuildResult,
	ModelExecutionResult,
	ModelMode,
	ModelResultPublisher,
	ModelRunResult,
)
from ..profile import ModelProfile
from ..SR2_Qwen3_4B_lora import LoRATrainer, LoRATrainingBatchRepository
from ..SR2_Qwen3_4B_lora.profile import PROFILE as LORA_PROFILE
from .checkpoint import SR2HeadCheckpointStore


class SR2ModelPipeline:
	"""Train only the SR2 classifier, reusing compatible warmup state and features."""

	def __init__(
		self,
		config: AppConfig,
		project_root: Path,
		profile: ModelProfile,
		checkpoint_tag: str | None = None,
	) -> None:
		self._config = config
		self._profile = profile
		model_config = profile.resolve_config(config)
		if not isinstance(model_config, SR2ModelConfig):
			raise TypeError("Frozen SR2 execution requires an SR2ModelConfig.")
		self._model_config = model_config
		self._head_config = model_config.as_warmup_config()
		checkpoint_name = f"head_{checkpoint_tag}.pt" if checkpoint_tag else "head.pt"
		self._checkpoint_store = SR2HeadCheckpointStore(
			project_root / "models" / profile.artifact_stem / checkpoint_name
		)
		self._result_publisher = ModelResultPublisher(
			config.result_dir, profile.result_prefix
		)

	def run(
		self,
		*,
		mode: ModelMode = ModelMode.VALIDATION,
		refresh_cache: bool = False,
		gpu_memory_monitor: GPUMemoryMonitor | None = None,
	) -> ModelExecutionResult:
		"""Prepare CPU inputs before head training and publish the selected output."""
		if mode not in (ModelMode.VALIDATION, ModelMode.BUILD):
			raise ModelExecutionError(
				f"{self._profile.selector} does not support {mode.value}."
			)
		try:
			batches = LoRATrainingBatchRepository.from_config(
				self._config,
				refresh_cache=refresh_cache,
				profile=self._profile,
				warmup_only=True,
				head_only_config=self._head_config,
			)
			training = None
			if mode is ModelMode.BUILD:
				training = batches.get_training(full_data=True)
				if training.warmup_context is None:
					raise ModelExecutionError(
						"Frozen SR2 builds require an immutable backbone identity."
					)
				self._checkpoint_store.check_writable()
			if gpu_memory_monitor is not None:
				gpu_memory_monitor.start()
			device = GPUEnvironment(self._config.gpu).configure()
			trainer = LoRATrainer(
				self._head_config,
				self._config.gpu,
				device,
				token_cache_dir=self._config.cache_dir / "token_cache",
				refresh_cache=refresh_cache,
				profile=self._profile,
				warmup_only=True,
				config_key=self._profile.config_key,
				warmup_cache_stem=LORA_PROFILE.artifact_stem,
				gpu_memory_monitor=gpu_memory_monitor,
			)
			if mode is ModelMode.VALIDATION:
				result = trainer.validate(batches)
				output_dir, validation_loss, predictions = (
					self._result_publisher.publish(
						result,
						self._config,
						self._profile,
						prediction_aggregation="original_only",
					)
				)
				return ModelRunResult(output_dir, predictions, validation_loss)
			model = trainer.create_model(
				seed=self._model_config.random_state, revision=batches.base_revision
			)
			try:
				history = trainer.fit(
					model,
					training,
					batches.data.canonical_texts,
					seed=self._model_config.random_state,
				)
				if history.warmup_training_loss is None:
					raise ModelExecutionError(
						"Frozen SR2 build did not produce a training loss."
					)
				self._checkpoint_store.save(
					model,
					self._model_config,
					self._config.gpu,
					context=training.warmup_context,
					training_source_sha256=batches.data.fingerprint,
				)
				return ModelBuildResult(
					checkpoint_path=self._checkpoint_store.path,
					training_rows=history.training_rows,
					epochs=len(history.training_losses),
					final_training_loss=history.warmup_training_loss,
				)
			finally:
				del model
		except torch.cuda.OutOfMemoryError as error:
			raise ModelExecutionError(
				f"CUDA ran out of memory during {self._profile.selector}. Lower "
				f"{self._profile.config_key}.extraction_budget or training_batch_size, "
				"or use fp16/bf16 instead of fp32. Inputs are never truncated."
			) from error
		except ModelExecutionError:
			raise
		except Exception as error:
			raise ModelExecutionError(
				f"Could not execute {self._profile.selector}: {error}"
			) from error
		finally:
			with suppress(Exception):
				gc.collect()
			with suppress(Exception):
				if torch.cuda.is_initialized():
					torch.cuda.empty_cache()
