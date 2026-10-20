"""Train independent LoRA adapters and heads, and predict with live encodings."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
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
	ModelData,
	PairwiseText,
)
from .data import (
	OptimizerGroupBatchSampler,
	PairwiseBatch,
	PairwiseCollator,
	PairwiseTokenDataset,
	PairwiseTokenStore,
)
from .extractor import WarmupEmbeddingExtractor
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

	def __init__(
		self, config: LoRAModelConfig, gpu: GPUConfig, device: torch.device
	) -> None:
		self._config = config
		self._gpu = gpu
		self._device = device
		self._tokens: PairwiseTokenStore | None = None
		self._base_revision: str | None = None
		self._model_initialized = False

	def create_model(
		self, *, seed: int, revision: str | None = None
	) -> LoRAQwenClassifier:
		"""Initialize fresh adapter/head parameters and a matching tokenizer."""
		set_seed(seed)
		model = LoRAQwenClassifier(
			self._config,
			self._gpu,
			self._device,
			revision=revision if revision is not None else self._base_revision,
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
		self, data: ModelData, n_splits: int
	) -> tuple[FoldTrainingResult, ...]:
		"""Train a fresh model per fold and restore augmented prediction row order."""
		results: list[FoldTrainingResult] = []
		for fold in range(n_splits):
			fold_started = self._timestamp()
			training = data.reference.loc[data.reference["fold"].ne(fold)]
			validation = data.reference.loc[data.reference["fold"].eq(fold)]
			originals = validation.loc[~validation["is_swapped"]]
			if training.empty or originals.empty:
				raise ModelExecutionError(
					f"Fold {fold} has an empty training or validation set."
				)
			model = self.create_model(seed=self._config.random_state + fold)
			try:
				history = self.fit(
					model,
					training,
					data.canonical_texts,
					seed=self._config.random_state + fold,
					validation=originals,
					description=f"Fold {fold}",
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
			results.append(
				FoldTrainingResult(
					fold=fold,
					ids=ids,
					probabilities=probabilities,
					epochs=history.epochs,
					total_time_seconds=self._timestamp() - fold_started,
				)
			)
		return tuple(results)

	def fit(
		self,
		model: LoRAQwenClassifier,
		training: pd.DataFrame,
		records: tuple[PairwiseText, ...],
		*,
		seed: int,
		validation: pd.DataFrame | None = None,
		description: str = "Build",
	) -> LoRATrainingHistory:
		"""Warm up the head, then jointly optimize adapters and head with linear decay."""
		if self._config.ab_swap == "inf":
			training = training.loc[~training["is_swapped"]]
		if training.empty:
			raise ModelExecutionError("LoRA training received no rows.")
		if validation is not None and validation.empty:
			raise ModelExecutionError("LoRA inference received no rows.")
		# Validate every canonical branch before any optimizer update, including validation.
		self._tokens_for(records)
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
			if epoch == self._config.warmup_epochs + 1:
				# Drop both views of the shared CPU feature store before joint training.
				loader = None
				warmup_validation = None
				loader = self._create_loader(
					training, records, training=True, seed=seed
				)
				for group, learning_rate in zip(
					optimizer.param_groups,
					(self._config.learning_rate, self._config.head_learning_rate),
					strict=True,
				):
					group["lr"] = learning_rate
				steps = (
					math.ceil(
						len(training)
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
					model, training, records, validation=validation, seed=seed
				)
			loss = self._train_epoch(
				model, loader, optimizer, scheduler, scaler, head_only=head_only
			)
			training_finished = self._timestamp()
			if validation is not None:
				validation_loss, probabilities = self.evaluate(
					model, validation, records, feature_loader=warmup_validation
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
			stage = "head warmup" if head_only else "joint training"
			stage_epoch = epoch if head_only else epoch - self._config.warmup_epochs
			stage_epochs = (
				self._config.warmup_epochs if head_only else self._config.epochs
			)
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
			training_rows=len(training),
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
	) -> float:
		"""Optimize one epoch, optionally retaining gradients only for the head."""
		model.train()
		if head_only:
			model.backbone.eval()
		optimizer.zero_grad(set_to_none=True)
		batch_size = (
			self._config.warmup_training_batch_size
			if head_only
			else self._config.training_batch_size
		)
		effective_batch_size = batch_size * self._config.gradient_accumulation_steps
		total_rows = len(loader.dataset)
		loss_total = torch.zeros((), device=self._device, dtype=torch.float64)
		finite_losses = torch.ones((), device=self._device, dtype=torch.bool)
		trainable_parameters = [
			parameter for parameter in model.parameters() if parameter.requires_grad
		]
		rows_total = 0
		group_rows = 0
		group_processed = 0
		for index, batch in enumerate(loader):
			if group_processed == 0:
				# Use actual row counts, including a short final micro-batch or group.
				group_rows = min(effective_batch_size, total_rows - rows_total)
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
			if index == 0 or (index + 1) % 100 == 0:
				stage = "head warmup" if head_only else "joint training"
				print(
					f"LoRA {stage} batch {index + 1:,} "
					f"(rows {rows_total:,}/{total_rows:,}): "
					f"train_loss={loss_total.item() / rows_total:.6f}",
					flush=True,
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
		canonical = pd.DataFrame({"id": ids.drop_duplicates(), "is_swapped": False})
		extraction_loader = self._create_loader(
			canonical,
			records,
			training=False,
			seed=seed,
			batch_size=self._config.warmup_extraction_batch_size,
		)
		embeddings = WarmupEmbeddingExtractor().extract(
			model, extraction_loader, tuple(int(row_id) for row_id in canonical["id"])
		)
		training_loader = DataLoader(
			embeddings.dataset(reference),
			batch_size=self._config.warmup_training_batch_size,
			shuffle=True,
			pin_memory=True,
			generator=torch.Generator().manual_seed(seed),
		)
		validation_loader = (
			DataLoader(
				embeddings.dataset(validation),
				batch_size=self._config.inference_batch_size,
				shuffle=False,
				pin_memory=True,
				generator=torch.Generator().manual_seed(self._config.random_state),
			)
			if validation is not None
			else None
		)
		return training_loader, validation_loader

	@torch.no_grad()
	def evaluate(
		self,
		model: LoRAQwenClassifier,
		reference: pd.DataFrame,
		records: tuple[PairwiseText, ...],
		*,
		feature_loader: DataLoader[list[torch.Tensor]] | None = None,
	) -> tuple[float, np.ndarray]:
		"""Evaluate live encodings or temporary frozen features in reference order."""
		if reference.empty:
			raise ModelExecutionError("LoRA inference received no rows.")
		model.eval()
		loader = (
			feature_loader
			if feature_loader is not None
			else self._create_loader(
				reference, records, training=False, seed=self._config.random_state
			)
		)
		probabilities = torch.empty(
			(len(reference), 3), device=self._device, dtype=torch.float32
		)
		loss_total = torch.zeros((), device=self._device, dtype=torch.float64)
		finite_probabilities = torch.ones((), device=self._device, dtype=torch.bool)
		start = 0
		for batch in loader:
			if feature_loader is not None:
				features, labels = batch
				features = features.to(self._device, non_blocking=True)
				log_probabilities = ABSwapAverager.log_probabilities(
					model.classifier, features[:, 0], features[:, 1]
				)
			else:
				labels = batch.labels
				log_probabilities = model.log_probabilities(batch)
			finite_probabilities.logical_and_(torch.isfinite(log_probabilities).all())
			if "label" in reference:
				labels = labels.to(self._device, non_blocking=True)
				loss_total.add_(F.nll_loss(log_probabilities, labels, reduction="sum"))
			stop = start + log_probabilities.shape[0]
			probabilities[start:stop].copy_(log_probabilities.exp())
			start = stop
		if not bool(finite_probabilities):
			raise ModelExecutionError(
				"LoRA inference produced non-finite probabilities."
			)
		return loss_total.item() / len(reference), probabilities.cpu().numpy()

	def _tokens_for(self, records: tuple[PairwiseText, ...]) -> PairwiseTokenStore:
		"""Reuse token storage across folds while keeping tokenizer revisions aligned."""
		if not self._model_initialized:
			raise RuntimeError("Initialize a LoRA model before creating its loaders.")
		if self._tokens is None or not self._tokens.matches(
			records, self._base_revision
		):
			self._tokens = PairwiseTokenStore(
				records,
				self._config.model_name,
				self._config.max_length,
				self._base_revision,
			)
		return self._tokens

	def _create_loader(
		self,
		reference: pd.DataFrame,
		records: tuple[PairwiseText, ...],
		*,
		training: bool,
		seed: int,
		batch_size: int | None = None,
	) -> DataLoader[PairwiseBatch]:
		tokens = self._tokens_for(records)
		dataset = PairwiseTokenDataset(reference, tokens)
		if batch_size is None:
			batch_size = (
				self._config.training_batch_size
				if training
				else self._config.inference_batch_size
			)
		workers = self._config.dataloader_workers
		generator = torch.Generator().manual_seed(seed)
		options = {
			"collate_fn": PairwiseCollator(tokens.pad_token_id),
			"num_workers": workers,
			"pin_memory": True,
			"persistent_workers": workers > 0,
			"generator": generator,
		}
		if training and self._config.training_token_budget > 0:
			return DataLoader(
				dataset,
				batch_sampler=OptimizerGroupBatchSampler(
					dataset,
					batch_size,
					self._config.gradient_accumulation_steps,
					self._config.training_token_budget,
					generator,
				),
				**options,
			)
		return DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=training,
			**options,
		)
