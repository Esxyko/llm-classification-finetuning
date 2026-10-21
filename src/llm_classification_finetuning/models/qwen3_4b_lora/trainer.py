"""Train independent LoRA adapters and heads, then infer over temporary features."""

from __future__ import annotations

import gc
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, set_seed

from ...config import GPUConfig, LoRAModelConfig
from ...errors import ModelExecutionError
from ..common import (
	ABSwapAverager,
	EpochMetrics,
	FoldTrainingResult,
	PairwiseText,
	PairwiseTokenStore,
	TokenizationCache,
	TrainingProgress,
)
from .batching import LoRATrainingBatchRepository, PreparedTrainingSplit
from .data import (
	PairwiseBatch,
	PairwiseCollator,
	PairwiseFeatureStore,
	PairwiseTokenDataset,
)
from .extractor import LoRAEmbeddingExtractor
from .model import LoRAQwenClassifier


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
		config: LoRAModelConfig,
		gpu: GPUConfig,
		device: torch.device,
		*,
		token_cache_dir: Path,
		refresh_cache: bool = False,
	) -> None:
		self._config = config
		self._gpu = gpu
		self._device = device
		self._token_cache_dir = token_cache_dir
		self._refresh_cache = refresh_cache
		self._tokens: PairwiseTokenStore | None = None
		self._base_revision: str | None = None
		self._model_initialized = False

	def create_model(
		self,
		*,
		seed: int,
		revision: str | None = None,
		device_map: dict[str, str] | None = None,
	) -> LoRAQwenClassifier:
		"""Initialize fresh adapter/head parameters and a matching tokenizer."""
		set_seed(seed)
		model = LoRAQwenClassifier(
			self._config,
			self._gpu,
			self._device,
			revision=revision if revision is not None else self._base_revision,
			device_map=device_map,
		)
		self._base_revision = model.base_revision
		self._model_initialized = True
		trainable = sum(
			parameter.numel()
			for parameter in model.parameters()
			if parameter.requires_grad
		)
		print(
			f"Initialized {self._config.model_name} with {trainable:,} trainable parameters.",
			flush=True,
		)
		return model

	def cross_validate(
		self, batches: LoRATrainingBatchRepository
	) -> Iterator[FoldTrainingResult]:
		"""Yield each fresh model's fold predictions after alignment and cleanup."""
		data = batches.data
		selected = data.reference
		if self._config.ab_swap == "inf":
			selected = selected.loc[~selected["is_swapped"]]
		training_rows = tuple(
			int(selected["fold"].ne(fold).sum()) for fold in range(batches.n_splits)
		)
		originals = data.reference.loc[~data.reference["is_swapped"]]
		validation_rows = tuple(
			int(originals["fold"].eq(fold).sum()) for fold in range(batches.n_splits)
		)
		progress = self._create_progress(
			training_rows, mode="cross-validation", validation_rows=validation_rows
		)
		for fold in range(batches.n_splits):
			fold_started = self._timestamp()
			training = batches.get_training(fold)
			validation = data.reference.loc[data.reference["fold"].eq(fold)]
			originals = validation.loc[~validation["is_swapped"]]
			if originals.empty:
				raise ModelExecutionError(
					f"Fold {fold} has an empty training or validation set."
				)
			model = self.create_model(
				seed=self._config.random_state + fold, revision=batches.base_revision
			)
			try:
				history = self.fit(
					model,
					training,
					data.canonical_texts,
					seed=self._config.random_state + fold,
					validation=originals,
					description=f"Fold {fold}",
					progress=progress,
				)
				if history.validation_probabilities is None:
					raise ModelExecutionError(
						f"Fold {fold} did not produce validation predictions."
					)
				ids = validation["id"].to_numpy(dtype=np.int64, copy=True)
				probabilities = ABSwapAverager.align_rows(
					validation["id"].to_numpy(dtype=np.int64),
					validation["is_swapped"].to_numpy(dtype=np.bool_),
					originals["id"].to_numpy(dtype=np.int64),
					history.validation_probabilities,
				)
			finally:
				del model
				gc.collect()
				torch.cuda.empty_cache()
			yield FoldTrainingResult(
				fold=fold,
				ids=ids,
				probabilities=probabilities,
				epochs=history.epochs,
				total_time_seconds=self._timestamp() - fold_started,
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
	) -> LoRATrainingHistory:
		"""Warm up the head, then jointly optimize adapters and head with linear decay."""
		reference = training.reference
		if reference.empty:
			raise ModelExecutionError("LoRA training received no rows.")
		if validation is not None and validation.empty:
			raise ModelExecutionError("LoRA inference received no rows.")
		if not training.tokens.matches(records, model.base_revision):
			raise ModelExecutionError(
				"Prepared LoRA tokens do not match the model's base revision or data."
			)
		self._tokens = training.tokens
		# Cached tokens have already been checked, including every validation branch.
		self._tokens_for(records)
		mode = "cross-validation" if validation is not None else "build"
		if progress is None:
			progress = self._create_progress(
				(len(reference),),
				mode=mode,
				validation_rows=() if validation is None else (len(validation),),
			)
		loader: DataLoader[PairwiseBatch] | DataLoader[list[torch.Tensor]] | None = None
		warmup_validation: DataLoader[list[torch.Tensor]] | None = None
		optimizer = AdamW(
			[
				{
					"params": [
						parameter
						for parameter in model.backbone.parameters()
						if parameter.requires_grad
					],
					"lr": self._config.learning_rate,
				},
				{
					"params": model.classifier.parameters(),
					"lr": (
						self._config.warmup_learning_rate
						if self._config.warmup_epochs
						else self._config.head_learning_rate
					),
				},
			],
			weight_decay=self._config.weight_decay,
		)
		scheduler: LambdaLR | None = None
		scaler = torch.amp.GradScaler("cuda", enabled=self._gpu.precision == "fp16")
		losses: list[float] = []
		metrics: list[EpochMetrics] = []
		probabilities: np.ndarray | None = None
		total_epochs = self._config.warmup_epochs + self._config.epochs
		for epoch in range(1, total_epochs + 1):
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
				steps = (
					math.ceil(
						len(reference)
						/ (
							self._config.training_batch_size
							* self._config.gradient_accumulation_steps
						)
					)
					* self._config.epochs
				)
				scheduler = get_linear_schedule_with_warmup(
					optimizer,
					num_warmup_steps=0,
					num_training_steps=steps,
				)
			training_started = self._timestamp()
			if loader is None:
				loader, warmup_validation = self._create_warmup_loaders(
					model, reference, records, validation=validation, seed=seed
				)
			if not head_only:
				training.set_epoch(epoch - self._config.warmup_epochs - 1)
			loss = self._train_epoch(
				model,
				loader,
				optimizer,
				scheduler,
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
		return LoRATrainingHistory(
			training_losses=tuple(losses),
			epochs=tuple(metrics),
			validation_probabilities=probabilities,
			training_rows=len(reference),
		)

	def _create_progress(
		self,
		training_rows: tuple[int, ...],
		*,
		mode: str,
		validation_rows: tuple[int, ...] = (),
	) -> TrainingProgress:
		"""Combine training and validation work, weighting all warmup work by half."""
		warmup_batch_size = self._config.warmup_training_batch_size
		joint_batch_size = (
			self._config.training_batch_size * self._config.gradient_accumulation_steps
		)
		warmup_steps = sum(
			(rows + warmup_batch_size - 1) // warmup_batch_size
			for rows in training_rows
		)
		joint_steps = sum(
			(rows + joint_batch_size - 1) // joint_batch_size for rows in training_rows
		)
		validation_batches = sum(
			(rows + self._config.head_inference_batch_size - 1)
			// self._config.head_inference_batch_size
			for rows in validation_rows
		)
		return TrainingProgress(
			model="qwen3_4b_lora",
			mode=mode,
			total_steps=(
				self.WARMUP_PROGRESS_WEIGHT
				* self._config.warmup_epochs
				* (warmup_steps + validation_batches)
				+ self._config.epochs * (joint_steps + validation_batches)
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
		scheduler: LambdaLR | None,
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
		planned_groups = (
			None if optimizer_group_rows is None else iter(optimizer_group_rows)
		)
		for batch in loader:
			if group_processed == 0:
				# A cached short group may occur anywhere after shuffling update groups.
				if planned_groups is None:
					group_rows = min(effective_batch_size, total_rows - rows_total)
				else:
					group_rows = next(planned_groups, 0)
					if not 0 < group_rows <= effective_batch_size:
						raise RuntimeError(
							"A LoRA batch plan has an invalid optimizer group."
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
				loss_sum = F.nll_loss(log_probabilities, labels, reduction="sum")
			else:
				logits = model.classifier(*encodings)
				loss_sum = F.cross_entropy(logits, labels, reduction="sum")
			finite_losses.logical_and_(torch.isfinite(loss_sum.detach()))
			scaler.scale(loss_sum / group_rows).backward()
			loss_total.add_(loss_sum.detach())
			rows_total += int(labels.shape[0])
			group_processed += int(labels.shape[0])
			if group_processed > group_rows:
				raise RuntimeError(
					"A LoRA micro-batch crossed an optimizer update boundary."
				)
			if group_processed == group_rows:
				if not bool(finite_losses):
					raise ModelExecutionError(
						"LoRA training produced a non-finite loss."
					)
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(
					trainable_parameters,
					self._config.max_grad_norm,
				)
				old_scale = scaler.get_scale()
				scaler.step(optimizer)
				scaler.update()
				if scheduler is not None and scaler.get_scale() >= old_scale:
					scheduler.step()
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
			or (planned_groups is not None and next(planned_groups, None) is not None)
		):
			raise RuntimeError(
				"LoRA training did not consume the complete optimizer-group plan."
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
		embeddings = self._extract_features(
			model,
			ids,
			records,
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
		seed: int,
		stage: str,
	) -> PairwiseFeatureStore:
		"""Encode each requested ID once in canonical orientation with current weights."""
		canonical = pd.DataFrame({"id": row_ids.drop_duplicates(), "is_swapped": False})
		loader = self._create_loader(
			canonical,
			records,
			seed=seed,
			batch_size=self._config.warmup_extraction_batch_size,
		)
		return LoRAEmbeddingExtractor().extract(
			model,
			loader,
			tuple(int(row_id) for row_id in canonical["id"]),
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
			raise ModelExecutionError("LoRA inference received no rows.")
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
				"LoRA inference produced non-finite probabilities."
			)
		result = loss_total.item() / len(reference), probabilities.cpu().numpy()
		if progress is not None and start == len(reference):
			# Complete the last batch only after the existing result checks succeed.
			progress.advance(description, stage=progress_stage, weight=progress_weight)
		return result

	def _tokens_for(self, records: tuple[PairwiseText, ...]) -> PairwiseTokenStore:
		"""Reuse token storage across folds while keeping tokenizer revisions aligned."""
		if not self._model_initialized:
			raise RuntimeError("Initialize a LoRA model before creating its loaders.")
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
			)
		return self._tokens

	def _create_loader(
		self,
		reference: pd.DataFrame,
		records: tuple[PairwiseText, ...],
		*,
		seed: int,
		batch_size: int,
	) -> DataLoader[PairwiseBatch]:
		tokens = self._tokens_for(records)
		dataset = PairwiseTokenDataset(reference, tokens)
		workers = self._config.dataloader_workers
		generator = torch.Generator().manual_seed(seed)
		options = {
			"collate_fn": PairwiseCollator(tokens.pad_token_id),
			"num_workers": workers,
			"pin_memory": True,
			"persistent_workers": workers > 0,
			"generator": generator,
		}
		return DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=False,
			**options,
		)
