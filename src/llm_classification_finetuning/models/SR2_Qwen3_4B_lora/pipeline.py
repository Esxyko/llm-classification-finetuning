"""Coordinate live LoRA holdout validation and production builds."""

from __future__ import annotations

import gc
from contextlib import suppress
from pathlib import Path

import torch

from ...config import AppConfig, LoRAModelConfig
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
from .batching import LoRATrainingBatchRepository
from .checkpoint import (
	LoRACheckpointCompatibility,
	LoRACheckpointStore,
	MergedBackboneStore,
)
from .trainer import LoRATrainer


class LoRAModelPipeline:
	"""Run single-GPU validation and full-data builds with portable adapters."""

	def __init__(
		self,
		config: AppConfig,
		project_root: Path,
		profile: ModelProfile,
		checkpoint_tag: str | None = None,
	) -> None:
		self._config = config
		self._profile = profile
		self._model_label = "QLoRA" if profile.quantize_training else "LoRA"
		model_config = profile.resolve_config(config)
		if not isinstance(model_config, LoRAModelConfig):
			raise TypeError(
				f"{self._model_label} execution requires a LoRAModelConfig."
			)
		self._model_config = model_config
		project_root = project_root.resolve()
		checkpoint_name = (
			f"adapter_{checkpoint_tag}.pt" if checkpoint_tag else "adapter.pt"
		)
		build_command = f"uv run model {profile.selector} --build"
		if checkpoint_tag:
			build_command += f" --checkpoint-tag {checkpoint_tag}"
		self._checkpoint_store = LoRACheckpointStore(
			project_root / "models" / profile.artifact_stem / checkpoint_name,
			build_command,
			model_label=self._model_label,
		)
		self._merged_backbone_store = MergedBackboneStore(
			self._checkpoint_store.path.parent
			/ (f"merged_{checkpoint_tag}" if checkpoint_tag else "merged")
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
		"""Reuse training tokens, batch plans and head warmup before live joint training."""
		try:
			if mode not in (ModelMode.VALIDATION, ModelMode.BUILD):
				raise ModelExecutionError(f"Unsupported model mode: {mode!r}")
			if mode is ModelMode.BUILD:
				return self._run_build(
					refresh_cache=refresh_cache, gpu_memory_monitor=gpu_memory_monitor
				)
			if mode is ModelMode.VALIDATION and gpu_memory_monitor is not None:
				gpu_memory_monitor.start()
			environment = GPUEnvironment(self._config.gpu)
			device = environment.configure()
			trainer = self._create_trainer(
				device,
				refresh_cache=refresh_cache,
				gpu_memory_monitor=gpu_memory_monitor,
			)
			batches = LoRATrainingBatchRepository.from_config(
				self._config, refresh_cache=refresh_cache, profile=self._profile
			)
			validation_result = trainer.validate(batches)
			output_dir, validation_loss, predictions = self._result_publisher.publish(
				validation_result,
				self._config,
				self._profile,
				prediction_aggregation=(
					"original_only"
					if self._model_config.ab_swap == "none"
					else "ab_swap_average"
				),
			)
			return ModelRunResult(output_dir, predictions, validation_loss)
		except torch.cuda.OutOfMemoryError as error:
			raise ModelExecutionError(
				f"CUDA ran out of memory during {self._model_label} execution. Lower "
				f"{self._profile.config_key}.gradient_checkpointing_threshold when positive, "
				"or lower gradient_checkpointing_budget (0 disables singleton merging), "
				"or lower warmup.extraction_budget for warmup extraction, "
				"warmup.warmup_training_batch_size for warmup head training, "
				f"{self._profile.config_key}.extraction_budget for "
				"joint-validation extraction, or training_batch_size "
				"or head_inference_batch_size for the affected stage. "
				"Keep max_length at least "
				"as large as the longest required input; inputs are never truncated. "
				"Gradient checkpointing is automatic above the threshold, or for all "
				"joint-training batches when the threshold is 0. Singletons above the "
				"checkpointing budget still run alone without truncation. "
				"Use fp16/bf16 instead of fp32. "
				"Increase gradient_accumulation_steps "
				"to preserve the effective training batch when lowering the micro-batch."
			) from error
		except ModelExecutionError:
			raise
		except Exception as error:
			raise ModelExecutionError(
				f"Could not execute {self._profile.selector}: {error}"
			) from error
		finally:
			# Cleanup must not replace an error raised by preparation, training or saving.
			with suppress(Exception):
				gc.collect()
			with suppress(Exception):
				if torch.cuda.is_initialized():
					torch.cuda.empty_cache()

	def _create_trainer(
		self,
		device: torch.device,
		*,
		refresh_cache: bool,
		gpu_memory_monitor: GPUMemoryMonitor | None = None,
	) -> LoRATrainer:
		return LoRATrainer(
			self._model_config,
			self._config.gpu,
			device,
			token_cache_dir=self._config.cache_dir / "token_cache",
			refresh_cache=refresh_cache,
			profile=self._profile,
			gpu_memory_monitor=gpu_memory_monitor,
		)

	def _run_build(
		self,
		*,
		refresh_cache: bool,
		gpu_memory_monitor: GPUMemoryMonitor | None = None,
	) -> ModelBuildResult:
		"""Prepare full-data inputs and both output paths before configuring CUDA."""
		batches = LoRATrainingBatchRepository.from_config(
			self._config, refresh_cache=refresh_cache, profile=self._profile
		)
		training = batches.get_training(full_data=True)
		self._checkpoint_store.check_writable()
		self._merged_backbone_store.check_writable()
		if gpu_memory_monitor is not None:
			gpu_memory_monitor.start()
		device = GPUEnvironment(self._config.gpu).configure()
		trainer = self._create_trainer(
			device,
			refresh_cache=refresh_cache,
			gpu_memory_monitor=gpu_memory_monitor,
		)
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
			self._checkpoint_store.save(
				model,
				self._compatibility(batches.serializer_version),
				training_source_sha256=batches.data.fingerprint,
				gpu=self._config.gpu,
			)
			merged_backbone_path = self._merged_backbone_store.save(
				model,
				batches.token_cache.tokenizer,
				model_name=self._model_config.model_name,
				training_source_sha256=batches.data.fingerprint,
				checkpoint_path=self._checkpoint_store.path,
			)
			return ModelBuildResult(
				checkpoint_path=self._checkpoint_store.path,
				training_rows=history.training_rows,
				epochs=len(history.training_losses),
				final_training_loss=history.training_losses[-1],
				executed_warmup_epochs=history.executed_warmup_epochs,
				executed_joint_epochs=history.executed_joint_epochs,
				merged_backbone_path=merged_backbone_path,
			)
		finally:
			del model

	def _compatibility(self, serializer_version: str) -> LoRACheckpointCompatibility:
		return LoRACheckpointCompatibility.from_config(
			self._model_config,
			serializer_version,
			quantize_training=self._profile.quantize_training,
			head_architecture=self._profile.head_architecture,
		)
