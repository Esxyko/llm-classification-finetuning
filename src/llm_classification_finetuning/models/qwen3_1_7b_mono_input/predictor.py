"""Run test inference with a trained mono-input classifier head."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ...config import ModelConfig
from ...errors import ModelExecutionError
from ..common.checkpoint import LoadedHeadCheckpoint
from .cache import MonoInputCachedEmbeddings
from .model import MonoInputClassificationHead
from .orientation import MonoInputOrientationAverager


class MonoInputPredictor:
    """Load a compatible head and average paired test predictions."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        selector: str,
        config_key: str,
    ) -> None:
        self._config = config
        self._device = device
        self._selector = selector
        self._config_key = config_key

    def predict(
        self,
        checkpoint: LoadedHeadCheckpoint,
        embeddings: MonoInputCachedEmbeddings,
        expected_ids: tuple[int, ...],
    ) -> np.ndarray:
        if embeddings.hidden_size != checkpoint.backbone_hidden_size:
            raise ModelExecutionError(
                "Test embedding size does not match the saved model checkpoint. "
                f"Rebuild it with 'uv run model {self._selector} --build'."
            )
        head = MonoInputClassificationHead(
            backbone_hidden_size=checkpoint.backbone_hidden_size,
            hidden_size=self._config.hidden_size,
            dropout=self._config.dropout,
        ).to(device=self._device, dtype=torch.float32)
        try:
            head.load_state_dict(checkpoint.state_dict, strict=True)
        except RuntimeError as error:
            raise ModelExecutionError(
                f"Saved model head parameters are invalid: {error}"
            ) from error
        loader = DataLoader(
            TensorDataset(embeddings.features),
            batch_size=self._config.training_batch_size,
            shuffle=False,
            num_workers=self._config.dataloader_workers,
            pin_memory=True,
            persistent_workers=self._config.dataloader_workers > 0,
        )
        probabilities: list[np.ndarray] = []
        head.eval()
        try:
            with torch.no_grad():
                for (features,) in loader:
                    logits = head(
                        features.to(
                            device=self._device,
                            dtype=torch.float32,
                            non_blocking=True,
                        )
                    )
                    probabilities.append(
                        torch.softmax(logits, dim=-1).to(device="cpu").numpy()
                    )
        except torch.cuda.OutOfMemoryError as error:
            raise ModelExecutionError(
                "GPU memory was exhausted during model test inference. Lower "
                f"{self._config_key}.training_batch_size in config.yaml."
            ) from error
        if not probabilities:
            raise ModelExecutionError("Model test inference produced no predictions.")
        predictions = np.concatenate(probabilities, axis=0)
        if not np.isfinite(predictions).all():
            raise ModelExecutionError(
                "Model test inference produced non-finite values."
            )
        averaged = MonoInputOrientationAverager().average(
            embeddings.ids.numpy(),
            embeddings.is_swapped.numpy(),
            predictions,
        )
        if not np.array_equal(averaged.ids, np.asarray(expected_ids, dtype=np.int64)):
            raise ModelExecutionError("Averaged test IDs do not match test.csv order.")
        return averaged.canonical


__all__ = ("MonoInputPredictor",)
