"""Create symmetric model A/B training examples."""

from __future__ import annotations

import pandas as pd


class ABAugmenter:
    """Append a swapped counterpart for every folded training row."""

    SWAP_COLUMN_PAIRS = (
        ("model_a", "model_b"),
        ("response_a", "response_b"),
        ("winner_model_a", "winner_model_b"),
    )
    LABEL_MAPPING = {0: 1, 1: 0, 2: 2}

    def augment(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return original rows followed by their A/B-swapped counterparts."""
        original = data.copy()
        original["is_swapped"] = False

        swapped = data.copy()
        for left_column, right_column in self.SWAP_COLUMN_PAIRS:
            left_values = swapped[left_column].copy()
            swapped[left_column] = swapped[right_column]
            swapped[right_column] = left_values

        swapped["label"] = (
            swapped["label"]
            .map(self.LABEL_MAPPING)
            .astype(data["label"].dtype)
        )
        swapped["is_swapped"] = True

        return pd.concat((original, swapped), ignore_index=True)
