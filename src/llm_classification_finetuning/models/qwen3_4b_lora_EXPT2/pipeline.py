"""Coordinate live LoRA holdout validation, production builds, and inference."""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoConfig

from ...config import AppConfig, LoRAModelConfig
from ...errors import ModelExecutionError
from ...model_storage import load_pretrained
from ..common import (
	GPUEnvironment,
	ModelBuildResult,
	ModelExecutionResult,
	ModelMode,
	ModelResultPublisher,
	ModelRunResult,
	ModelSubmissionPublisher,
	ModelSubmitResult,
	ModelTestDataRepository,
	SubmissionAttention,
)
from ..profile import ModelProfile
from .batching import LoRATrainingBatchRepository
from .checkpoint import LoRACheckpointCompatibility, LoRACheckpointStore
from .trainer import LoRATrainer


class LoRAModelPipeline:
	"""Run single-GPU training or dual-GPU submission with portable adapters."""

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
		self._test_repository = ModelTestDataRepository(config.data.raw_dir)
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
		self._result_publisher = ModelResultPublisher(
			project_root / "results", profile.result_prefix
		)
		self._submission_publisher = ModelSubmissionPublisher(
			project_root / "results" / "submit", profile.result_prefix
		)

	def run(
		self,
		*,
		mode: ModelMode = ModelMode.VALIDATION,
		refresh_cache: bool = False,
	) -> ModelExecutionResult:
		"""Reuse training tokens, batch plans and head warmup before live joint training."""
		try:
			environment = GPUEnvironment(self._config.gpu)
			submit_devices = (
				environment.configure_submit() if mode is ModelMode.SUBMIT else None
			)
			device = submit_devices[0] if submit_devices else environment.configure()
			trainer = LoRATrainer(
				self._model_config,
				self._config.gpu,
				device,
				token_cache_dir=self._config.data.processed_path.parent / "token_cache",
				refresh_cache=refresh_cache,
				profile=self._profile,
			)
			if submit_devices is not None:
				with SubmissionAttention(self._config.gpu).activate():
					return self._run_submit(trainer, submit_devices)
			if mode not in (ModelMode.VALIDATION, ModelMode.BUILD):
				raise ModelExecutionError(f"Unsupported model mode: {mode!r}")
			batches = LoRATrainingBatchRepository.from_config(
				self._config, refresh_cache=refresh_cache, profile=self._profile
			)
			if mode is ModelMode.BUILD:
				training = batches.get_training(full_data=True)
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
					return ModelBuildResult(
						checkpoint_path=self._checkpoint_store.path,
						training_rows=history.training_rows,
						epochs=len(history.training_losses),
						final_training_loss=history.training_losses[-1],
					)
				finally:
					del model
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
				"joint-validation/submission extraction, or training_batch_size "
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
			gc.collect()
			if torch.cuda.is_initialized():
				torch.cuda.empty_cache()

	def _run_submit(
		self,
		trainer: LoRATrainer,
		submit_devices: tuple[torch.device, torch.device],
	) -> ModelSubmitResult:
		data = self._test_repository.load()
		checkpoint = self._checkpoint_store.load(
			self._compatibility(self._test_repository.serializer_version)
		)
		base_config = load_pretrained(
			AutoConfig.from_pretrained,
			self._model_config.model_name,
			revision=checkpoint.base_revision,
		)
		model = trainer.create_model(
			seed=self._model_config.random_state,
			revision=checkpoint.base_revision,
			device_map=GPUEnvironment.submission_device_map(
				base_config, submit_devices
			),
			quantize_backbone=False,
		)
		try:
			self._checkpoint_store.restore(model, checkpoint)
			reference = pd.DataFrame({"id": data.ids, "is_swapped": False})
			_, probabilities = trainer.evaluate(model, reference, data.canonical_texts)
			output_dir, submission_path = self._submission_publisher.publish(
				data.ids, probabilities
			)
			return ModelSubmitResult(output_dir, submission_path, len(data.ids))
		finally:
			del model

	def _compatibility(self, serializer_version: str) -> LoRACheckpointCompatibility:
		return LoRACheckpointCompatibility.from_config(
			self._model_config,
			serializer_version,
			quantize_training=self._profile.quantize_training,
			head_architecture=self._profile.head_architecture,
		)
