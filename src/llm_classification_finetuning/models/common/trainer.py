"""Train pairwise MLP heads for validation folds or a full-data build."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from ...config import ModelConfig
from ...errors import ModelExecutionError
from .cache import CachedEmbeddings
from .model import PairwiseClassificationHead


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Training and validation loss for one completed epoch."""

    epoch: int
    training_loss: float
    validation_loss: float


@dataclass(frozen=True, slots=True)
class FoldTrainingResult:
    """Final probabilities and metrics for one validation fold."""

    fold: int
    ids: np.ndarray
    probabilities: np.ndarray
    epochs: tuple[EpochMetrics, ...]

    @property
    def final_loss(self) -> float:
        """Return the final epoch validation loss."""
        return self.epochs[-1].validation_loss


@dataclass(frozen=True, slots=True)
class FullTrainingResult:
    """A classifier head trained on every row and its epoch losses."""

    head: PairwiseClassificationHead
    rows: int
    training_losses: tuple[float, ...]


class PairwiseEmbeddingDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Orient canonical cached embeddings to match augmented processed rows."""

    def __init__(
        self,
        reference: pd.DataFrame,
        embeddings: CachedEmbeddings,
    ) -> None:
        id_to_position = {
            int(row_id): position
            for position, row_id in enumerate(embeddings.ids.tolist())
        }
        try:
            positions = [id_to_position[int(row_id)] for row_id in reference["id"]]
        except KeyError as error:
            raise ModelExecutionError(
                f"No cached embedding exists for processed ID {error.args[0]}."
            ) from error

        self._h_a = embeddings.h_a
        self._h_b = embeddings.h_b
        self._positions = torch.tensor(positions, dtype=torch.int64)
        self._swapped = torch.tensor(
            reference["is_swapped"].to_numpy(dtype=np.bool_),
            dtype=torch.bool,
        )
        self._labels = torch.tensor(
            reference["label"].to_numpy(dtype=np.int64),
            dtype=torch.int64,
        )

    def __len__(self) -> int:
        return int(self._positions.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        position = int(self._positions[index])
        if bool(self._swapped[index]):
            h_a, h_b = self._h_b[position], self._h_a[position]
        else:
            h_a, h_b = self._h_a[position], self._h_b[position]
        return h_a, h_b, self._labels[index]


class ModelCrossValidator:
    """Train pairwise heads for cross-validation or a full-data build."""

    def __init__(
        self,
        config: ModelConfig,
        n_splits: int,
        device: torch.device,
        config_key: str,
    ) -> None:
        self._config = config
        self._n_splits = n_splits
        self._device = device
        self._config_key = config_key
        self._criterion = CrossEntropyLoss()

    def train(
        self,
        reference: pd.DataFrame,
        embeddings: CachedEmbeddings,
    ) -> tuple[FoldTrainingResult, ...]:
        """Train all folds in order and return final validation probabilities."""
        results: list[FoldTrainingResult] = []
        for fold in range(self._n_splits):
            training_reference = reference.loc[reference["fold"].ne(fold)]
            validation_reference = reference.loc[reference["fold"].eq(fold)]
            if training_reference.empty or validation_reference.empty:
                raise ModelExecutionError(
                    f"Fold {fold} has an empty training or validation set."
                )
            results.append(
                self._train_fold(
                    fold,
                    training_reference,
                    validation_reference,
                    embeddings,
                )
            )
        return tuple(results)

    def train_all(
        self,
        reference: pd.DataFrame,
        embeddings: CachedEmbeddings,
    ) -> FullTrainingResult:
        """Train one head on every original and augmented fold row."""
        if reference.empty:
            raise ModelExecutionError("Full-data model training received no rows.")
        seed = self._config.random_state
        self._seed_everything(seed)
        training_loader = self._create_loader(
            reference,
            embeddings,
            shuffle=True,
            seed=seed,
        )
        head = self._create_head(embeddings.hidden_size)
        optimizer = self._create_optimizer(head)
        training_losses: list[float] = []

        try:
            for epoch in range(1, self._config.epochs + 1):
                training_loss = self._train_epoch(head, optimizer, training_loader)
                training_losses.append(training_loss)
                print(
                    f"Build epoch {epoch}/{self._config.epochs}: "
                    f"train_loss={training_loss:.6f}"
                )
        except torch.cuda.OutOfMemoryError as error:
            raise ModelExecutionError(
                "GPU memory was exhausted while training the model head. Lower "
                f"{self._config_key}.training_batch_size in config.yaml."
            ) from error

        return FullTrainingResult(
            head=head,
            rows=len(reference),
            training_losses=tuple(training_losses),
        )

    def _train_fold(
        self,
        fold: int,
        training_reference: pd.DataFrame,
        validation_reference: pd.DataFrame,
        embeddings: CachedEmbeddings,
    ) -> FoldTrainingResult:
        seed = self._config.random_state + fold
        self._seed_everything(seed)
        training_loader = self._create_loader(
            training_reference,
            embeddings,
            shuffle=True,
            seed=seed,
        )
        validation_loader = self._create_loader(
            validation_reference,
            embeddings,
            shuffle=False,
            seed=seed,
        )

        head = self._create_head(embeddings.hidden_size)
        optimizer = self._create_optimizer(head)
        epoch_metrics: list[EpochMetrics] = []
        final_probabilities: np.ndarray | None = None

        try:
            for epoch in range(1, self._config.epochs + 1):
                training_loss = self._train_epoch(
                    head,
                    optimizer,
                    training_loader,
                )
                validation_loss, probabilities = self._evaluate(head, validation_loader)
                final_probabilities = probabilities
                metrics = EpochMetrics(
                    epoch=epoch,
                    training_loss=training_loss,
                    validation_loss=validation_loss,
                )
                epoch_metrics.append(metrics)
                print(
                    f"Fold {fold} epoch {epoch}/{self._config.epochs}: "
                    f"train_loss={metrics.training_loss:.6f} "
                    f"validation_loss={metrics.validation_loss:.6f}"
                )
        except torch.cuda.OutOfMemoryError as error:
            raise ModelExecutionError(
                "GPU memory was exhausted while training the model head. Lower "
                f"{self._config_key}.training_batch_size in config.yaml."
            ) from error

        if final_probabilities is None:
            raise ModelExecutionError(
                f"Fold {fold} did not produce validation predictions."
            )
        return FoldTrainingResult(
            fold=fold,
            ids=validation_reference["id"].to_numpy(dtype=np.int64, copy=True),
            probabilities=final_probabilities,
            epochs=tuple(epoch_metrics),
        )

    def _create_head(self, backbone_hidden_size: int) -> PairwiseClassificationHead:
        return PairwiseClassificationHead(
            backbone_hidden_size=backbone_hidden_size,
            hidden_size=self._config.hidden_size,
            dropout=self._config.dropout,
        ).to(device=self._device, dtype=torch.float32)

    def _create_optimizer(self, head: PairwiseClassificationHead) -> AdamW:
        return AdamW(
            head.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

    def _train_epoch(
        self,
        head: PairwiseClassificationHead,
        optimizer: AdamW,
        loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    ) -> float:
        head.train()
        loss_total = 0.0
        rows_total = 0
        for h_a, h_b, labels in loader:
            h_a = h_a.to(
                device=self._device,
                dtype=torch.float32,
                non_blocking=True,
            )
            h_b = h_b.to(
                device=self._device,
                dtype=torch.float32,
                non_blocking=True,
            )
            labels = labels.to(self._device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = head(h_a, h_b)
            loss = self._criterion(logits, labels)
            loss.backward()
            optimizer.step()

            rows = int(labels.shape[0])
            loss_total += float(loss.detach()) * rows
            rows_total += rows
        if rows_total == 0:
            raise ModelExecutionError("Model training loader produced no rows.")
        return loss_total / rows_total

    def _create_loader(
        self,
        reference: pd.DataFrame,
        embeddings: CachedEmbeddings,
        *,
        shuffle: bool,
        seed: int,
    ) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        generator = torch.Generator()
        generator.manual_seed(seed)
        workers = self._config.dataloader_workers
        return DataLoader(
            PairwiseEmbeddingDataset(reference, embeddings),
            batch_size=self._config.training_batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            generator=generator,
        )

    def _evaluate(
        self,
        head: PairwiseClassificationHead,
        loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[float, np.ndarray]:
        head.eval()
        loss_total = 0.0
        rows_total = 0
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for h_a, h_b, labels in loader:
                h_a = h_a.to(
                    device=self._device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                h_b = h_b.to(
                    device=self._device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                labels = labels.to(self._device, non_blocking=True)
                logits = head(h_a, h_b)
                loss = self._criterion(logits, labels)

                rows = int(labels.shape[0])
                loss_total += float(loss) * rows
                rows_total += rows
                probabilities.append(
                    torch.softmax(logits, dim=-1).to(device="cpu").numpy()
                )
        return loss_total / rows_total, np.concatenate(probabilities, axis=0)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
