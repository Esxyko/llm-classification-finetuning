"""Train independent LoRA adapters and heads, and predict with live encodings."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass

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
from .data import PairwiseBatch, PairwiseCollator, PairwiseTextDataset
from .model import LoRAQwenClassifier


@dataclass(frozen=True, slots=True)
class LoRATrainingHistory:
	"""Epoch losses and final canonical validation predictions."""

	training_losses: tuple[float, ...]
	epochs: tuple[EpochMetrics, ...]
	validation_probabilities: np.ndarray | None


class LoRATrainer:
	"""Run adapter optimization on one configured CUDA device."""

	def __init__(
		self, config: LoRAModelConfig, gpu: GPUConfig, device: torch.device
	) -> None:
		self._config = config
		self._gpu = gpu
		self._device = device
		self._collator: PairwiseCollator | None = None

	def create_model(
		self, *, seed: int, revision: str | None = None
	) -> LoRAQwenClassifier:
		"""Initialize fresh adapter/head parameters and a matching tokenizer."""
		set_seed(seed)
		model = LoRAQwenClassifier(
			self._config, self._gpu, self._device, revision=revision
		)
		self._collator = PairwiseCollator(
			self._config.model_name, self._config.max_length, model.base_revision
		)
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
				results.append(
					FoldTrainingResult(
						fold=fold,
						ids=validation["id"].to_numpy(dtype=np.int64, copy=True),
						probabilities=ABSwapAverager.align_rows(
							validation["id"].to_numpy(dtype=np.int64),
							validation["is_swapped"].to_numpy(dtype=np.bool_),
							originals["id"].to_numpy(dtype=np.int64),
							history.validation_probabilities,
						),
						epochs=history.epochs,
					)
				)
			finally:
				del model
				gc.collect()
				torch.cuda.empty_cache()
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
		"""Optimize both augmented orientations with an effective-batch schedule."""
		if training.empty:
			raise ModelExecutionError("LoRA training received no rows.")
		loader = self._create_loader(training, records, training=True, seed=seed)
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
					"lr": self._config.head_learning_rate,
				},
			],
			weight_decay=self._config.weight_decay,
		)
		steps = (
			math.ceil(len(loader) / self._config.gradient_accumulation_steps)
			* self._config.epochs
		)
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=math.ceil(steps * self._config.warmup_ratio),
			num_training_steps=steps,
		)
		scaler = torch.amp.GradScaler("cuda", enabled=self._gpu.precision == "fp16")
		losses: list[float] = []
		metrics: list[EpochMetrics] = []
		probabilities: np.ndarray | None = None
		for epoch in range(1, self._config.epochs + 1):
			loss = self._train_epoch(model, loader, optimizer, scheduler, scaler)
			losses.append(loss)
			message = f"{description} epoch {epoch}/{self._config.epochs}: train_loss={loss:.6f}"
			if validation is not None:
				validation_loss, probabilities = self.evaluate(
					model, validation, records
				)
				metrics.append(EpochMetrics(epoch, loss, validation_loss))
				message += f" validation_loss={validation_loss:.6f}"
			print(message, flush=True)
		return LoRATrainingHistory(tuple(losses), tuple(metrics), probabilities)

	def _train_epoch(
		self,
		model: LoRAQwenClassifier,
		loader: DataLoader[PairwiseBatch],
		optimizer: AdamW,
		scheduler: LambdaLR,
		scaler: torch.amp.GradScaler,
	) -> float:
		model.train()
		optimizer.zero_grad(set_to_none=True)
		accumulation = self._config.gradient_accumulation_steps
		total_rows = len(loader.dataset)
		loss_total = 0.0
		rows_total = 0
		group_rows = 0
		for index, batch in enumerate(loader):
			if index % accumulation == 0:
				# Use actual row counts, including a short final micro-batch or group.
				group_rows = min(
					accumulation * self._config.training_batch_size,
					total_rows - rows_total,
				)
			labels = batch.labels.to(self._device, non_blocking=True)
			loss_sum = F.cross_entropy(model(batch), labels, reduction="sum")
			if not torch.isfinite(loss_sum):
				raise ModelExecutionError("LoRA training produced a non-finite loss.")
			scaler.scale(loss_sum / group_rows).backward()
			loss_total += float(loss_sum.detach())
			rows_total += int(labels.shape[0])
			if (index + 1) % accumulation == 0 or index + 1 == len(loader):
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(
					[
						parameter
						for parameter in model.parameters()
						if parameter.requires_grad
					],
					self._config.max_grad_norm,
				)
				old_scale = scaler.get_scale()
				scaler.step(optimizer)
				scaler.update()
				if scaler.get_scale() >= old_scale:
					scheduler.step()
				optimizer.zero_grad(set_to_none=True)
			if index == 0 or (index + 1) % 100 == 0:
				print(
					f"LoRA training batch {index + 1:,}/{len(loader):,}: "
					f"train_loss={loss_total / rows_total:.6f}",
					flush=True,
				)
		return loss_total / rows_total

	@torch.no_grad()
	def evaluate(
		self,
		model: LoRAQwenClassifier,
		reference: pd.DataFrame,
		records: tuple[PairwiseText, ...],
	) -> tuple[float, np.ndarray]:
		"""Return averaged probabilities in reference order and optional label loss."""
		if reference.empty:
			raise ModelExecutionError("LoRA inference received no rows.")
		model.eval()
		loader = self._create_loader(
			reference, records, training=False, seed=self._config.random_state
		)
		probabilities: list[np.ndarray] = []
		loss_total = 0.0
		for batch in loader:
			log_probabilities = model.log_probabilities(batch)
			if not torch.isfinite(log_probabilities).all():
				raise ModelExecutionError(
					"LoRA inference produced non-finite probabilities."
				)
			if "label" in reference:
				labels = batch.labels.to(self._device, non_blocking=True)
				loss_total += float(
					F.nll_loss(log_probabilities, labels, reduction="sum")
				)
			probabilities.append(log_probabilities.exp().cpu().numpy())
		return loss_total / len(reference), np.concatenate(probabilities, axis=0)

	def _create_loader(
		self,
		reference: pd.DataFrame,
		records: tuple[PairwiseText, ...],
		*,
		training: bool,
		seed: int,
	) -> DataLoader[PairwiseBatch]:
		if self._collator is None:
			raise RuntimeError("Initialize a LoRA model before creating its loaders.")
		workers = self._config.dataloader_workers
		return DataLoader(
			PairwiseTextDataset(reference, records),
			batch_size=self._config.training_batch_size
			if training
			else self._config.inference_batch_size,
			shuffle=training,
			collate_fn=self._collator,
			num_workers=workers,
			pin_memory=True,
			persistent_workers=workers > 0,
			generator=torch.Generator().manual_seed(seed),
		)
