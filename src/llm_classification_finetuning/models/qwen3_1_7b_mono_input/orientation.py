"""Pair mono-input orientations and average their class probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import ModelExecutionError


@dataclass(frozen=True, slots=True)
class AveragedMonoInputPredictions:
    """Canonical predictions in original-row order and augmented-row order."""

    ids: np.ndarray
    canonical: np.ndarray
    row_aligned: np.ndarray


class MonoInputOrientationAverager:
    """Restore swapped outputs to the original A/B orientation before averaging."""

    SWAPPED_ORDER = (1, 0, 2)

    def average(
        self,
        ids: np.ndarray,
        is_swapped: np.ndarray,
        probabilities: np.ndarray,
    ) -> AveragedMonoInputPredictions:
        ids = np.asarray(ids)
        is_swapped = np.asarray(is_swapped)
        probabilities = np.asarray(probabilities)
        if (
            ids.ndim != 1
            or is_swapped.shape != ids.shape
            or is_swapped.dtype != np.bool_
            or probabilities.shape != (len(ids), 3)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or (probabilities > 1.0).any()
            or not np.allclose(
                probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6
            )
        ):
            raise ModelExecutionError("Mono-input orientation predictions are invalid.")

        original_positions = np.flatnonzero(~is_swapped)
        swapped_positions = np.flatnonzero(is_swapped)
        original_ids = ids[original_positions]
        swapped_ids = ids[swapped_positions]
        if (
            not len(original_positions)
            or len(original_positions) != len(swapped_positions)
            or len(set(original_ids.tolist())) != len(original_positions)
            or len(set(swapped_ids.tolist())) != len(swapped_positions)
            or set(original_ids.tolist()) != set(swapped_ids.tolist())
        ):
            raise ModelExecutionError(
                "Each mono-input ID must have one original and one swapped prediction."
            )

        swapped_by_id = {
            row_id: position
            for row_id, position in zip(
                swapped_ids.tolist(), swapped_positions.tolist(), strict=True
            )
        }
        paired_swapped_positions = np.array(
            [swapped_by_id[row_id] for row_id in original_ids.tolist()],
            dtype=np.int64,
        )
        canonical = (
            probabilities[original_positions]
            + probabilities[paired_swapped_positions][:, self.SWAPPED_ORDER]
        ) / 2.0
        row_aligned = np.empty_like(probabilities)
        row_aligned[original_positions] = canonical
        row_aligned[paired_swapped_positions] = canonical[:, self.SWAPPED_ORDER]
        return AveragedMonoInputPredictions(
            ids=original_ids,
            canonical=canonical,
            row_aligned=row_aligned,
        )


__all__ = ("AveragedMonoInputPredictions", "MonoInputOrientationAverager")
