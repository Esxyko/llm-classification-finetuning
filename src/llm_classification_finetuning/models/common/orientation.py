"""Average pairwise predictions across original and swapped A/B order."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ...errors import ModelExecutionError
from .model import PairwiseClassificationHead


class ABSwapAverager:
    """Return canonical probabilities and align them with processed fold rows."""

    SWAPPED_ORDER = (1, 0, 2)

    @classmethod
    def log_probabilities(
        cls, head: PairwiseClassificationHead, h_a: Tensor, h_b: Tensor
    ) -> Tensor:
        """Average both orientations in probability space, with stable logs."""
        original = F.log_softmax(head(h_a, h_b), dim=-1)
        swapped = F.log_softmax(head(h_b, h_a), dim=-1)
        return torch.logaddexp(
            original, swapped[:, list(cls.SWAPPED_ORDER)]
        ) - math.log(2.0)

    @classmethod
    def align_rows(
        cls,
        reference_ids: np.ndarray,
        is_swapped: np.ndarray,
        canonical_ids: np.ndarray,
        canonical_probabilities: np.ndarray,
    ) -> np.ndarray:
        """Repeat each canonical distribution in the corresponding row order."""
        positions_by_id = {
            int(row_id): position for position, row_id in enumerate(canonical_ids)
        }
        if (
            len(positions_by_id) != len(canonical_ids)
            or canonical_probabilities.shape != (len(canonical_ids), 3)
            or len(reference_ids) != 2 * len(canonical_ids)
            or is_swapped.shape != reference_ids.shape
        ):
            raise ModelExecutionError("Validation orientation pairs are invalid.")
        try:
            positions = [positions_by_id[int(row_id)] for row_id in reference_ids]
        except KeyError as error:
            raise ModelExecutionError(
                f"No averaged validation prediction exists for ID {error.args[0]}."
            ) from error
        row_aligned = canonical_probabilities[positions].copy()
        row_aligned[is_swapped] = row_aligned[is_swapped][
            :, list(cls.SWAPPED_ORDER)
        ]
        return row_aligned
