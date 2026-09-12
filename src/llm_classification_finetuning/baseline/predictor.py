"""Run submission inference with a trained pairwise classifier head."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import BaselineConfig
from ..errors import BaselineError
from .cache import CachedEmbeddings
from .checkpoint import LoadedHeadCheckpoint
from .model import PairwiseClassificationHead


class BaselinePredictor:
    """Load a compatible classifier head and predict cached test embeddings."""

    def __init__(
        self,
        config: BaselineConfig,
        device: torch.device,
    ) -> None:
        self._config = config
        self._device = device

    def predict(
        self,
        checkpoint: LoadedHeadCheckpoint,
        embeddings: CachedEmbeddings,
    ) -> np.ndarray:
        """Return A-win, B-win, and tie probabilities in embedding order."""
        if embeddings.hidden_size != checkpoint.backbone_hidden_size:
            raise BaselineError(
                "Test embedding size does not match the saved baseline checkpoint. "
                "Rebuild the model with 'uv run baseline --build'."
            )
        head = PairwiseClassificationHead(
            backbone_hidden_size=checkpoint.backbone_hidden_size,
            hidden_size=self._config.hidden_size,
            dropout=self._config.dropout,
        ).to(device=self._device, dtype=torch.float32)
        try:
            head.load_state_dict(checkpoint.state_dict, strict=True)
        except RuntimeError as error:
            raise BaselineError(
                f"Saved baseline head parameters are invalid: {error}"
            ) from error

        loader = DataLoader(
            TensorDataset(embeddings.h_a, embeddings.h_b),
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
                for h_a, h_b in loader:
                    logits = head(
                        h_a.to(
                            device=self._device,
                            dtype=torch.float32,
                            non_blocking=True,
                        ),
                        h_b.to(
                            device=self._device,
                            dtype=torch.float32,
                            non_blocking=True,
                        ),
                    )
                    probabilities.append(
                        torch.softmax(logits, dim=-1).to(device="cpu").numpy()
                    )
        except torch.cuda.OutOfMemoryError as error:
            raise BaselineError(
                "GPU memory was exhausted during baseline test inference. Lower "
                "baseline.training_batch_size in config.yaml."
            ) from error

        if not probabilities:
            raise BaselineError("Baseline test inference produced no predictions.")
        predictions = np.concatenate(probabilities, axis=0)
        if not np.isfinite(predictions).all():
            raise BaselineError("Baseline test inference produced non-finite values.")
        return predictions
