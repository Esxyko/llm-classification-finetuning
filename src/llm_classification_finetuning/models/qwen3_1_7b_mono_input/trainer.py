"""Train mono-input MLP heads for validation folds or a full-data build."""

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
from ..common.trainer import EpochMetrics, FoldTrainingResult
from .cache import MonoInputCachedEmbeddings
from .model import MonoInputClassificationHead
from .orientation import MonoInputOrientationAverager


@dataclass(frozen=True, slots=True)
class MonoInputFullTrainingResult:
    """A mono-input head trained on every original row."""

    head: MonoInputClassificationHead
    rows: int
    training_losses: tuple[float, ...]


class MonoInputEmbeddingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Select cached states by the stable augmented-row position."""

    def __init__(
        self,
        reference: pd.DataFrame,
        embeddings: MonoInputCachedEmbeddings,
    ) -> None:
        positions = torch.tensor(reference.index.to_numpy(), dtype=torch.int64)
        if positions.numel() and (
            int(positions.min()) < 0 or int(positions.max()) >= embeddings.ids.shape[0]
        ):
            raise ModelExecutionError(
                "Processed row positions do not match the mono-input embedding cache."
            )
        expected_ids = torch.tensor(
            reference["id"].to_numpy(dtype=np.int64),
            dtype=torch.int64,
        )
        expected_swapped = torch.tensor(
            reference["is_swapped"].to_numpy(dtype=np.bool_),
            dtype=torch.bool,
        )
        if not torch.equal(embeddings.ids[positions], expected_ids) or not torch.equal(
            embeddings.is_swapped[positions], expected_swapped
        ):
            raise ModelExecutionError(
                "Cached mono-input rows do not match processed IDs and orientations."
            )
        self._features = embeddings.features
        self._positions = positions
        self._labels = torch.tensor(
            reference["label"].to_numpy(dtype=np.int64),
            dtype=torch.int64,
        )

    def __len__(self) -> int:
        return int(self._positions.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._features[int(self._positions[index])], self._labels[index]


class MonoInputCrossValidator:
    """Train the mono-input head for cross-validation or production builds."""

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
        self._averager = MonoInputOrientationAverager()

    def train(
        self,
        reference: pd.DataFrame,
        embeddings: MonoInputCachedEmbeddings,
    ) -> tuple[FoldTrainingResult, ...]:
        results: list[FoldTrainingResult] = []
        for fold in range(self._n_splits):
            training_reference = reference.loc[
                reference["fold"].ne(fold) & ~reference["is_swapped"]
            ]
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
        embeddings: MonoInputCachedEmbeddings,
    ) -> MonoInputFullTrainingResult:
        reference = reference.loc[~reference["is_swapped"]]
        if reference.empty:
            raise ModelExecutionError("Full-data model training received no rows.")
        seed = self._config.random_state
        self._seed_everything(seed)
        loader = self._create_loader(reference, embeddings, shuffle=True, seed=seed)
        head = self._create_head(embeddings.hidden_size)
        optimizer = self._create_optimizer(head)
        losses: list[float] = []
        try:
            for epoch in range(1, self._config.epochs + 1):
                loss = self._train_epoch(head, optimizer, loader)
                losses.append(loss)
                print(
                    f"Build epoch {epoch}/{self._config.epochs}: train_loss={loss:.6f}"
                )
        except torch.cuda.OutOfMemoryError as error:
            raise ModelExecutionError(
                "GPU memory was exhausted while training the model head. Lower "
                f"{self._config_key}.training_batch_size in config.yaml."
            ) from error
        return MonoInputFullTrainingResult(
            head=head,
            rows=len(reference),
            training_losses=tuple(losses),
        )

    def _train_fold(
        self,
        fold: int,
        training_reference: pd.DataFrame,
        validation_reference: pd.DataFrame,
        embeddings: MonoInputCachedEmbeddings,
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
                training_loss = self._train_epoch(head, optimizer, training_loader)
                validation_loss, probabilities = self._evaluate(
                    head,
                    validation_loader,
                    validation_reference,
                )
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

    def _create_head(self, backbone_hidden_size: int) -> MonoInputClassificationHead:
        return MonoInputClassificationHead(
            backbone_hidden_size=backbone_hidden_size,
            hidden_size=self._config.hidden_size,
            dropout=self._config.dropout,
        ).to(device=self._device, dtype=torch.float32)

    def _create_optimizer(self, head: MonoInputClassificationHead) -> AdamW:
        return AdamW(
            head.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

    def _train_epoch(
        self,
        head: MonoInputClassificationHead,
        optimizer: AdamW,
        loader: DataLoader[tuple[Tensor, Tensor]],
    ) -> float:
        head.train()
        loss_total = 0.0
        rows_total = 0
        for features, labels in loader:
            features = features.to(
                device=self._device,
                dtype=torch.float32,
                non_blocking=True,
            )
            labels = labels.to(self._device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = self._criterion(head(features), labels)
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
        embeddings: MonoInputCachedEmbeddings,
        *,
        shuffle: bool,
        seed: int,
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        generator = torch.Generator()
        generator.manual_seed(seed)
        workers = self._config.dataloader_workers
        return DataLoader(
            MonoInputEmbeddingDataset(reference, embeddings),
            batch_size=self._config.training_batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            generator=generator,
        )

    def _evaluate(
        self,
        head: MonoInputClassificationHead,
        loader: DataLoader[tuple[Tensor, Tensor]],
        reference: pd.DataFrame,
    ) -> tuple[float, np.ndarray]:
        head.eval()
        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for features, _labels in loader:
                features = features.to(
                    device=self._device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                logits = head(features)
                probabilities.append(
                    torch.softmax(logits, dim=-1).to(device="cpu").numpy()
                )
        if not probabilities:
            raise ModelExecutionError("Model validation loader produced no rows.")
        averaged = self._averager.average(
            reference["id"].to_numpy(dtype=np.int64),
            reference["is_swapped"].to_numpy(dtype=np.bool_),
            np.concatenate(probabilities, axis=0),
        )
        original_reference = reference.loc[~reference["is_swapped"]]
        if not np.array_equal(
            averaged.ids, original_reference["id"].to_numpy(dtype=np.int64)
        ):
            raise ModelExecutionError("Averaged validation IDs are out of order.")
        labels = original_reference["label"].to_numpy(dtype=np.int64)
        selected = averaged.canonical[np.arange(len(labels)), labels]
        validation_loss = float(
            -np.log(np.clip(selected, np.finfo(np.float64).tiny, 1.0)).mean()
        )
        return validation_loss, averaged.row_aligned

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


__all__ = (
    "MonoInputCrossValidator",
    "MonoInputEmbeddingDataset",
    "MonoInputFullTrainingResult",
)
