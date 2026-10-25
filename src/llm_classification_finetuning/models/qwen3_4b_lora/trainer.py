"""Train independent LoRA adapters and heads, then infer over temporary features."""

from __future__ import annotations

import gc
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import set_seed

from ...config import GPUConfig, LoRAModelConfig, WarmupModelConfig
from ...errors import ModelExecutionError
from ..common import (
	ABSwapAverager,
	ClassificationLoss,
	EpochMetrics,
	ExtractionBatchSampler,
	PairwiseText,
	PairwiseTokenStore,
	TokenizationCache,
	TrainingProgress,
	ValidationTrainingResult,
	WarmupBuildResult,
)
from ..profile import ModelProfile
from .batching import (
	LoRATrainingBatchRepository,
	OptimizerGroupLayout,
	PreparedTrainingSplit,
)
from .data import (
	BranchBatch,
	BranchCollator,
	BranchTokenDataset,
	PairwiseBatch,
	PairwiseFeatureStore,
)
from .extractor import LoRAEmbeddingExtractor
from .heads import PAIRWISE_MLP
from .model import LoRAQwenClassifier
from .warmup_cache import HeadWarmupCache, WarmupCacheContext, WarmupCacheEntry


@dataclass(frozen=True, slots=True)
class LoRATrainingHistory:
	"""Epoch losses and final canonical validation predictions."""

	training_losses: tuple[float, ...]
	epochs: tuple[EpochMetrics, ...]
	validation_probabilities: np.ndarray | None
	training_rows: int


class LoRATrainer:
	"""Run adapter optimization on one configured CUDA device."""

	WARMUP_PROGRESS_WEIGHT = 0.5

	def __init__(
		self,
		config: WarmupModelConfig,
		gpu: GPUConfig,
		device: torch.device,
		*,
		token_cache_dir: Path,
		refresh_cache: bool = False,
		profile: ModelProfile | None = None,
		warmup_only: bool = False,
	) -> None:
		if not warmup_only and not isinstance(config, LoRAModelConfig):
			raise TypeError("Joint training requires a LoRAModelConfig.")
		if warmup_only and config.warmup_epochs <= 0:
			raise ModelExecutionError("Standalone warmup requires warmup_epochs > 0.")
		if warmup_only and profile is not None and profile.quantize_training:
			raise ValueError("Standalone warmup cannot use a quantized profile.")
		self._config = config
		self._criterion = ClassificationLoss(
			label_smoothing=config.label_smoothing, reduction="sum"
		)
		self._warmup_only = warmup_only
		self._gpu = gpu
		self._device = device
		self._token_cache_dir = token_cache_dir
		self._refresh_cache = refresh_cache
		self._config_key = (
			"warmup"
			if warmup_only
			else (profile.config_key if profile else "qwen3_4b_lora")
		)
		self._model_key = (
			"warmup"
			if warmup_only
			else (profile.artifact_stem if profile else "qwen3_4b_lora")
		)
		self._quantize_training = profile.quantize_training if profile else False
		self._head_architecture = profile.head_architecture if profile else PAIRWISE_MLP
		self._model_label = "QLoRA" if self._quantize_training else "LoRA"
		artifact_stem = profile.artifact_stem if profile else "qwen3_4b_lora"
		self._warmup_cache = HeadWarmupCache(
			token_cache_dir.parent / f"{artifact_stem}_warmup",
			config,
			gpu,
			device,
			quantize_training=self._quantize_training,
			model_label=self._model_label,
			head_architecture=self._head_architecture,
		)
		self._lora_warmup_cache = (
			HeadWarmupCache(
				token_cache_dir.parent / "qwen3_4b_lora_warmup",
				config,
				gpu,
				device,
				head_architecture=self._head_architecture,
			)
			if self._quantize_training
			else None
		)
		self._tokens: PairwiseTokenStore | None = None
		self._base_revision: str | None = None
		self._model_initialized = False

	def create_model(
		self,
		*,
		seed: int,
		revision: str | None = None,
		device_map: dict[str, str] | None = None,
		quantize_backbone: bool | None = None,
	) -> LoRAQwenClassifier:
		"""Initialize fresh adapter/head parameters and a matching tokenizer."""
		set_seed(seed)
		model = LoRAQwenClassifier(
			self._config,
			self._gpu,
			self._device,
			revision=revision if revision is not None else self._base_revision,
			device_map=device_map,
			quantize_backbone=(
				self._quantize_training
				if quantize_backbone is None
				else quantize_backbone
			),
			warmup_only=self._warmup_only,
			head_architecture=self._head_architecture,
		)
		self._base_revision = model.base_revision
		self._model_initialized = True
		trainable = sum(
			parameter.numel()
			for parameter in model.parameters()
			if parameter.requires_grad
		)
		print(
			f"Initialized {self._model_label} {self._config.model_name} "
			f"with {trainable:,} trainable parameters.",
			flush=True,
		)
		return model

	def build_warmup(self, batches: LoRATrainingBatchRepository) -> WarmupBuildResult:
		"""Build only train-partition head warmup, avoiding weights on a cache hit."""
		if not self._warmup_only:
			raise ModelExecutionError("Warmup cache builds require warmup_only=True.")
		training = batches.get_training()
		seed = self._config.random_state
		entry = self._prepare_warmup(training.warmup_context, seed=seed)
		if entry is None:
			raise ModelExecutionError("Standalone warmup requires a cache identity.")
		path = self._warmup_cache.path(entry)
		if entry.state is not None:
			print(
				f"Reused LoRA head warmup (Warmup cache build): "
				f"source_precision={entry.state['source_precision']}, "
				f"precision={self._gpu.precision}; {path}",
				flush=True,
			)
			return WarmupBuildResult(path, True, len(training.reference), 0)
		model = self.create_model(seed=seed, revision=batches.base_revision)
		try:
			history = self.fit(
				model,
				training,
				batches.data.canonical_texts,
				seed=seed,
				description="Warmup cache build",
				warmup_cache_entry=entry,
			)
			return WarmupBuildResult(
				path, False, history.training_rows, len(history.training_losses)
			)
		finally:
			del model

	def validate(
		self, batches: LoRATrainingBatchRepository
	) -> ValidationTrainingResult:
		"""Train one fresh model against the saved holdout and align final predictions."""
		if self._warmup_only:
			raise ModelExecutionError(
				"Standalone warmup does not run holdout validation."
			)
		run_started = self._timestamp()
		data = batches.data
		training = batches.get_training()
		validation = data.reference.loc[data.reference["split"].eq("validation")]
		originals = validation.loc[~validation["is_swapped"]]
		if originals.empty:
			raise ModelExecutionError("The holdout validation set is empty.")
		seed = self._config.random_state
		model = self.create_model(seed=seed, revision=batches.base_revision)
		try:
			history = self.fit(
				model,
				training,
				data.canonical_texts,
				seed=seed,
				validation=originals,
				description="Validation run",
			)
			if history.validation_probabilities is None:
				raise ModelExecutionError(
					"The holdout run did not produce validation predictions."
				)
			if self._config.ab_swap == "none":
				ids = originals["id"].to_numpy(dtype=np.int64, copy=True)
				probabilities = history.validation_probabilities
			else:
				ids = validation["id"].to_numpy(dtype=np.int64, copy=True)
				probabilities = ABSwapAverager.align_rows(
					ids,
					validation["is_swapped"].to_numpy(dtype=np.bool_),
					originals["id"].to_numpy(dtype=np.int64),
					history.validation_probabilities,
				)
		finally:
			del model
			gc.collect()
			torch.cuda.empty_cache()
		return ValidationTrainingResult(
			ids=ids,
			probabilities=probabilities,
			epochs=history.epochs,
			total_time_seconds=self._timestamp() - run_started,
			training_rows=history.training_rows,
			source_training_rows=int(
				data.reference.loc[data.reference["split"].eq("train"), "id"].nunique()
			),
		)

	def fit(
		self,
		model: LoRAQwenClassifier,
		training: PreparedTrainingSplit,
		records: tuple[PairwiseText, ...],
		*,
		seed: int,
		validation: pd.DataFrame | None = None,
		description: str = "Build",
		progress: TrainingProgress | None = None,
		warmup_cache_entry: WarmupCacheEntry | None = None,
	) -> LoRATrainingHistory:
		"""Warm up the head, then jointly optimize adapters and head at fixed rates."""
		if self._warmup_only != model.warmup_only:
			raise ModelExecutionError("Model and trainer warmup_only modes must match.")
		if self._warmup_only and validation is not None:
			raise ModelExecutionError("Standalone warmup does not run validation.")
		reference = training.reference
		if reference.empty:
			raise ModelExecutionError(f"{self._model_label} training received no rows.")
		if validation is not None and validation.empty:
			raise ModelExecutionError(
				f"{self._model_label} inference received no rows."
			)
		if not training.tokens.matches(records, model.base_revision):
			raise ModelExecutionError(
				f"Prepared {self._model_label} tokens do not match the model's base revision or data."
			)
		self._tokens = training.tokens
		# Cached tokens have already been checked, including every validation branch.
		self._tokens_for(records)
		context = training.warmup_context
		if context is not None and (
			context.base_revision != model.base_revision
			or context.backbone_hidden_size != model.backbone_hidden_size
		):
			raise ModelExecutionError(
				"Warmup cache context does not match the model backbone."
			)
		entry = None
		if self._config.warmup_epochs and context is not None:
			entry = warmup_cache_entry or self._prepare_warmup(context, seed=seed)
		warmup_cache_entry = None
		reused_warmup = entry is not None and entry.state is not None
		mode = "validation" if validation is not None else "build"
		if progress is None:
			progress = self._create_progress(
				len(reference),
				mode=mode,
				validation_rows=0 if validation is None else len(validation),
				warmup_required=not reused_warmup,
			)
		loader: DataLoader[PairwiseBatch] | DataLoader[list[torch.Tensor]] | None = None
		warmup_validation: DataLoader[list[torch.Tensor]] | None = None
		parameter_groups = [
			{
				"params": list(model.classifier.parameters()),
				"lr": self._config.warmup_learning_rate
				if self._config.warmup_epochs
				else self._config.head_learning_rate,
			}
		]
		if not self._warmup_only:
			parameter_groups.insert(
				0,
				{
					"params": [
						parameter
						for parameter in model.backbone.parameters()
						if parameter.requires_grad
					],
					"lr": self._config.learning_rate,
				},
			)
		optimizer = AdamW(parameter_groups, weight_decay=self._config.weight_decay)
		scaler = torch.amp.GradScaler("cuda", enabled=self._gpu.precision == "fp16")
		if reused_warmup:
			self._warmup_cache.restore(entry, model.classifier, optimizer, scaler)
			print(
				f"Reused {self._model_label} head warmup ({description}): "
				f"skipping {self._config.warmup_epochs} warmup epochs; "
				f"source_precision={entry.state['source_precision']}, "
				f"source_quantization={entry.metadata['training_quantization']}, "
				f"precision={self._gpu.precision}; {self._warmup_cache.path(entry)}",
				flush=True,
			)
			# Release CPU head/moment copies before joint training starts.
			entry = None
		elif entry is not None:
			print(
				f"Rebuilding {self._model_label} head warmup ({description}): "
				f"precision={self._gpu.precision}; {self._warmup_cache.path(entry)}",
				flush=True,
			)
		losses: list[float] = []
		metrics: list[EpochMetrics] = []
		probabilities: np.ndarray | None = None
		total_epochs = self._config.warmup_epochs + (
			0 if self._warmup_only else self._config.epochs
		)
		first_epoch = self._config.warmup_epochs + 1 if reused_warmup else 1
		if self._config.warmup_epochs and not reused_warmup:
			# Feature extraction belongs to run setup, outside every epoch timer.
			loader, warmup_validation = self._create_warmup_loaders(
				model, reference, records, validation=validation, seed=seed
			)
		for epoch in range(first_epoch, total_epochs + 1):
			epoch_started = self._timestamp()
			head_only = epoch <= self._config.warmup_epochs
			stage = "head warmup" if head_only else "joint training"
			stage_epoch = epoch if head_only else epoch - self._config.warmup_epochs
			stage_epochs = (
				self._config.warmup_epochs if head_only else self._config.epochs
			)
			epoch_description = (
				f"{description} epoch {stage_epoch}/{stage_epochs} "
				f"(total {epoch}/{total_epochs})"
			)
			if epoch == self._config.warmup_epochs + 1:
				# Drop both views of the shared CPU feature store before joint training.
				loader = None
				warmup_validation = None
				loader = training.loader
				for group, learning_rate in zip(
					optimizer.param_groups,
					(self._config.learning_rate, self._config.head_learning_rate),
					strict=True,
				):
					group["lr"] = learning_rate
			training_started = self._timestamp()
			if not head_only:
				training.set_epoch(epoch - self._config.warmup_epochs - 1)
			loss = self._train_epoch(
				model,
				loader,
				optimizer,
				scaler,
				head_only=head_only,
				optimizer_group_rows=None
				if head_only
				else training.optimizer_group_rows,
				progress=progress,
				description=epoch_description,
			)
			training_finished = self._timestamp()
			if validation is not None:
				validation_loss, probabilities = self.evaluate(
					model,
					validation,
					records,
					feature_loader=warmup_validation,
					head_only=head_only,
					progress=progress,
					description=epoch_description,
				)
				validation_finished = self._timestamp()
				metrics.append(
					EpochMetrics(
						epoch=epoch,
						training_loss=loss,
						validation_loss=validation_loss,
						training_time_seconds=training_finished - training_started,
						validation_time_seconds=validation_finished - training_finished,
						total_time_seconds=validation_finished - epoch_started,
					)
				)
			losses.append(loss)
			message = (
				f"{description} {stage} epoch {stage_epoch}/{stage_epochs} "
				f"(total {epoch}/{total_epochs}): train_loss={loss:.6f}"
			)
			if validation is not None:
				message += f" validation_loss={validation_loss:.6f}"
			print(message, flush=True)
			if head_only and epoch == self._config.warmup_epochs and entry is not None:
				self._warmup_cache.save(entry, model.classifier, optimizer, scaler)
				print(
					f"Saved {self._model_label} head warmup ({description}): "
					f"source_precision={self._gpu.precision}; {self._warmup_cache.path(entry)}",
					flush=True,
				)
		return LoRATrainingHistory(
			training_losses=tuple(losses),
			epochs=tuple(metrics),
			validation_probabilities=probabilities,
			training_rows=len(reference),
		)

	def _prepare_warmup(
		self, context: WarmupCacheContext | None, *, seed: int
	) -> WarmupCacheEntry | None:
		"""Leave manually constructed splits and disabled warmup uncached."""
		if not self._config.warmup_epochs or context is None:
			return None
		if self._lora_warmup_cache is not None and not self._refresh_cache:
			entry = self._lora_warmup_cache.prepare(context, seed=seed)
			if entry.state is not None:
				return entry
		return self._warmup_cache.prepare(
			context, seed=seed, refresh=self._refresh_cache
		)

	def _create_progress(
		self,
		training_rows: int,
		*,
		mode: str,
		validation_rows: int = 0,
		warmup_required: bool = True,
	) -> TrainingProgress:
		"""Count one run's training and validation, weighting warmup work by half."""
		warmup_training_steps = OptimizerGroupLayout(
			training_rows, self._config.warmup_training_batch_size
		).update_count
		if self._warmup_only:
			return TrainingProgress(
				model=self._model_key,
				mode=mode,
				total_steps=(
					self.WARMUP_PROGRESS_WEIGHT
					* self._config.warmup_epochs
					* warmup_training_steps
					if warmup_required
					else 0
				),
			)
		joint_batch_size = (
			self._config.training_batch_size * self._config.gradient_accumulation_steps
		)
		validation_steps = math.ceil(
			validation_rows / self._config.head_inference_batch_size
		)
		warmup_steps = warmup_training_steps + validation_steps
		joint_steps = OptimizerGroupLayout(training_rows, joint_batch_size).update_count
		return TrainingProgress(
			model=self._model_key,
			mode=mode,
			total_steps=(
				(
					self.WARMUP_PROGRESS_WEIGHT
					* self._config.warmup_epochs
					* warmup_steps
					if warmup_required
					else 0
				)
				+ self._config.epochs * (joint_steps + validation_steps)
			),
		)

	def _timestamp(self) -> float:
		"""Wait for this device's GPU work before reading the monotonic clock."""
		if self._device.type == "cuda":
			torch.cuda.synchronize(self._device)
		return perf_counter()

	def _train_epoch(
		self,
		model: LoRAQwenClassifier,
		loader: DataLoader[PairwiseBatch] | DataLoader[list[torch.Tensor]],
		optimizer: AdamW,
		scaler: torch.amp.GradScaler,
		*,
		head_only: bool = False,
		optimizer_group_rows: tuple[int, ...] | None = None,
		progress: TrainingProgress | None = None,
		description: str = "Build",
	) -> float:
		"""Optimize one epoch, optionally retaining gradients only for the head."""
		model.train()
		if head_only:
			model.backbone.eval()
		optimizer.zero_grad(set_to_none=True)
		effective_batch_size = (
			self._config.warmup_training_batch_size
			if head_only
			else self._config.training_batch_size
			* self._config.gradient_accumulation_steps
		)
		total_rows = len(loader.dataset)
		loss_total = torch.zeros((), device=self._device, dtype=torch.float64)
		finite_losses = torch.ones((), device=self._device, dtype=torch.bool)
		trainable_parameters = [
			parameter for parameter in model.parameters() if parameter.requires_grad
		]
		rows_total = 0
		group_rows = 0
		group_processed = 0
		layout = OptimizerGroupLayout(total_rows, effective_batch_size)
		expected_groups = layout.row_counts
		if optimizer_group_rows is None:
			optimizer_group_rows = expected_groups
		elif optimizer_group_rows != expected_groups:
			raise RuntimeError(
				f"A {self._model_label} batch plan has invalid optimizer-group sizes."
			)
		planned_groups = iter(optimizer_group_rows)
		for batch in loader:
			if group_processed == 0:
				group_rows = next(planned_groups, 0)
				if not 0 < group_rows <= total_rows - rows_total:
					raise RuntimeError(
						f"A {self._model_label} batch plan has an invalid optimizer group."
					)
			if head_only:
				features, labels = batch
				features = features.to(self._device, non_blocking=True)
				labels = labels.to(self._device, non_blocking=True)
				encodings = (features[:, 0], features[:, 1])
			else:
				labels = batch.labels.to(self._device, non_blocking=True)
				encodings = model.encode(batch)
			if self._config.ab_swap == "inf":
				log_probabilities = ABSwapAverager.log_probabilities(
					model.classifier, *encodings
				)
				loss_sum = self._criterion.from_log_probabilities(
					log_probabilities, labels
				)
			else:
				logits = model.classifier(*encodings)
				loss_sum = self._criterion(logits, labels)
			finite_losses.logical_and_(torch.isfinite(loss_sum.detach()))
			scaler.scale(loss_sum / group_rows).backward()
			loss_total.add_(loss_sum.detach())
			rows_total += int(labels.shape[0])
			group_processed += int(labels.shape[0])
			if group_processed > group_rows:
				raise RuntimeError(
					f"A {self._model_label} micro-batch crossed an optimizer update boundary."
				)
			if group_processed == group_rows:
				if not bool(finite_losses):
					raise ModelExecutionError(
						f"{self._model_label} training produced a non-finite loss."
					)
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(
					trainable_parameters,
					self._config.max_grad_norm,
				)
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad(set_to_none=True)
				group_processed = 0
				if progress is not None:
					# A consumed group advances progress even if FP16 skips its update.
					progress.advance(
						description,
						stage="head warmup" if head_only else "joint training",
						weight=self.WARMUP_PROGRESS_WEIGHT if head_only else 1.0,
					)
		if (
			group_processed
			or rows_total != total_rows
			or next(planned_groups, None) is not None
		):
			raise RuntimeError(
				f"{self._model_label} training did not consume the complete optimizer-group plan."
			)
		return loss_total.item() / rows_total

	def _create_warmup_loaders(
		self,
		model: LoRAQwenClassifier,
		reference: pd.DataFrame,
		records: tuple[PairwiseText, ...],
		*,
		validation: pd.DataFrame | None,
		seed: int,
	) -> tuple[DataLoader[list[torch.Tensor]], DataLoader[list[torch.Tensor]] | None]:
		"""Extract each needed ID once, then share oriented features for head warmup."""
		ids = reference["id"]
		if validation is not None:
			ids = pd.concat((ids, validation["id"]), ignore_index=True)
		with nullcontext() if model.warmup_only else model.backbone.disable_adapter():
			embeddings = self._extract_features(
				model,
				ids,
				records,
				extraction_budget=(
					self._config.extraction_budget
					if self._warmup_only
					else self._config.warmup_extraction_budget
				),
				seed=seed,
				stage="warmup",
			)
		training_loader = DataLoader(
			embeddings.dataset(reference),
			batch_size=self._config.warmup_training_batch_size,
			shuffle=True,
			pin_memory=True,
			generator=torch.Generator().manual_seed(seed),
		)
		validation_loader = (
			self._create_feature_loader(embeddings, validation)
			if validation is not None
			else None
		)
		return training_loader, validation_loader

	def _extract_features(
		self,
		model: LoRAQwenClassifier,
		row_ids: pd.Series,
		records: tuple[PairwiseText, ...],
		*,
		extraction_budget: int,
		seed: int,
		stage: str,
	) -> PairwiseFeatureStore:
		"""Encode each requested ID once in canonical orientation with current weights."""
		canonical_ids = row_ids.drop_duplicates()
		loader = self._create_extraction_loader(
			canonical_ids,
			records,
			seed=seed,
			extraction_budget=extraction_budget,
		)
		return LoRAEmbeddingExtractor(self._model_label).extract(
			model,
			loader,
			tuple(int(row_id) for row_id in canonical_ids),
			stage=stage,
		)

	def _create_feature_loader(
		self, embeddings: PairwiseFeatureStore, reference: pd.DataFrame
	) -> DataLoader[list[torch.Tensor]]:
		"""Select reference-ordered feature views for head-only inference."""
		return DataLoader(
			embeddings.dataset(reference),
			batch_size=self._config.head_inference_batch_size,
			shuffle=False,
			pin_memory=True,
			generator=torch.Generator().manual_seed(self._config.random_state),
		)

	@torch.no_grad()
	def evaluate(
		self,
		model: LoRAQwenClassifier,
		reference: pd.DataFrame,
		records: tuple[PairwiseText, ...],
		*,
		feature_loader: DataLoader[list[torch.Tensor]] | None = None,
		head_only: bool = False,
		progress: TrainingProgress | None = None,
		description: str = "",
	) -> tuple[float, np.ndarray]:
		"""Extract current features or reuse warmup views, then run head inference."""
		if reference.empty:
			raise ModelExecutionError(
				f"{self._model_label} inference received no rows."
			)
		model.eval()
		progress_stage = (
			"head warmup validation"
			if head_only
			else "joint validation"
			if "label" in reference
			else "submission"
		)
		if feature_loader is None:
			embeddings = self._extract_features(
				model,
				reference["id"],
				records,
				extraction_budget=self._config.extraction_budget,
				seed=self._config.random_state,
				stage=progress_stage,
			)
			feature_loader = self._create_feature_loader(embeddings, reference)
		probabilities = torch.empty(
			(len(reference), 3), device=self._device, dtype=torch.float32
		)
		loss_total = torch.zeros((), device=self._device, dtype=torch.float64)
		finite_probabilities = torch.ones((), device=self._device, dtype=torch.bool)
		progress_weight = self.WARMUP_PROGRESS_WEIGHT if head_only else 1.0
		start = 0
		for features, labels in feature_loader:
			features = features.to(self._device, non_blocking=True)
			if self._config.ab_swap == "none":
				log_probabilities = F.log_softmax(
					model.classifier(features[:, 0], features[:, 1]), dim=-1
				)
			else:
				log_probabilities = ABSwapAverager.log_probabilities(
					model.classifier, features[:, 0], features[:, 1]
				)
			finite_probabilities.logical_and_(torch.isfinite(log_probabilities).all())
			if "label" in reference:
				labels = labels.to(self._device, non_blocking=True)
				loss_total.add_(F.nll_loss(log_probabilities, labels, reduction="sum"))
			stop = start + log_probabilities.shape[0]
			probabilities[start:stop].copy_(log_probabilities.exp())
			start = stop
			if progress is not None and stop < len(reference):
				progress.advance(
					description,
					stage=progress_stage,
					weight=progress_weight,
				)
		if not bool(finite_probabilities):
			raise ModelExecutionError(
				f"{self._model_label} inference produced non-finite probabilities."
			)
		result = loss_total.item() / len(reference), probabilities.cpu().numpy()
		if progress is not None and start == len(reference):
			# Complete the last batch only after the existing result checks succeed.
			progress.advance(description, stage=progress_stage, weight=progress_weight)
		return result

	def _tokens_for(self, records: tuple[PairwiseText, ...]) -> PairwiseTokenStore:
		"""Reuse token storage across stages while keeping tokenizer revisions aligned."""
		if not self._model_initialized:
			raise RuntimeError(
				f"Initialize a {self._model_label} model before creating its loaders."
			)
		if self._tokens is None or not self._tokens.matches(
			records, self._base_revision
		):
			cache = TokenizationCache.from_pretrained(
				self._token_cache_dir,
				self._config.model_name,
				revision=self._base_revision,
			)
			self._tokens = PairwiseTokenStore(
				records,
				cache,
				revision=self._base_revision,
				max_length=self._config.max_length,
				refresh_cache=self._refresh_cache,
				max_length_field=f"{self._config_key}.max_length",
			)
		return self._tokens

	def _create_extraction_loader(
		self,
		row_ids: pd.Series,
		records: tuple[PairwiseText, ...],
		*,
		seed: int,
		extraction_budget: int,
	) -> DataLoader[BranchBatch]:
		"""Batch individual A/B branches with the frozen model's extraction semantics."""
		tokens = self._tokens_for(records)
		dataset = BranchTokenDataset(row_ids, tokens)
		sampler = ExtractionBatchSampler(
			[dataset.branch_length(index) for index in range(len(dataset))],
			extraction_budget,
		)
		workers = self._config.dataloader_workers
		generator = torch.Generator().manual_seed(seed)
		options = {
			"collate_fn": BranchCollator(tokens.pad_token_id),
			"num_workers": workers,
			"pin_memory": True,
			"persistent_workers": workers > 0,
			"generator": generator,
		}
		return DataLoader(
			dataset,
			batch_sampler=sampler,
			**options,
		)
