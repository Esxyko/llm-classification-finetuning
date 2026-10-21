"""Balance fold token volumes while preserving grouped class distributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig

from ..config import CrossValidationConfig
from ..errors import DataPreparationError
from ..models.common import ConversationSerializer, TokenizationCache


@dataclass(frozen=True, slots=True)
class TokenBalanceResult:
	"""Balanced assignments and canonical A+B token counts in source-row order."""

	folds: pd.Series
	pair_tokens: np.ndarray
	swaps: int


class FoldTokenBalancer:
	"""Swap whole prompt groups with identical three-class count vectors."""

	MAX_SWAPS = 1_000

	def __init__(self, config: CrossValidationConfig) -> None:
		self._config = config

	def balance(self, data: pd.DataFrame, cache_dir: Path) -> TokenBalanceResult:
		"""Count real tokens, then improve the initial stratified group assignment."""
		if data.empty:
			raise DataPreparationError("Cannot balance folds for empty training data.")
		pair_tokens = self._count_tokens(data, cache_dir)
		groups = (
			data.assign(_tokens=pair_tokens)
			.groupby("group_id", sort=True)
			.agg(fold=("fold", "first"), tokens=("_tokens", "sum"))
		)
		class_counts = pd.crosstab(data["group_id"], data["label"]).reindex(
			index=groups.index, columns=(0, 1, 2), fill_value=0
		)
		folds = groups["fold"].to_numpy(dtype=np.int16, copy=True)
		tokens = groups["tokens"].to_numpy(dtype=np.int64)
		buckets: dict[tuple[int, ...], list[int]] = {}
		order = np.random.default_rng(self._config.random_state).permutation(
			len(groups)
		)
		counts = class_counts.to_numpy()
		for index in order:
			buckets.setdefault(tuple(int(value) for value in counts[index]), []).append(
				int(index)
			)
		indices_by_class = tuple(np.asarray(indices) for indices in buckets.values())
		totals = np.zeros(self._config.n_splits, dtype=np.int64)
		np.add.at(totals, folds, tokens)
		mean_tokens = float(totals.sum()) / self._config.n_splits
		if mean_tokens <= 0:
			raise DataPreparationError(
				"Training inputs must contain tokens to balance folds."
			)

		for swaps in range(self.MAX_SWAPS + 1):
			deviation = float(np.abs(totals / mean_tokens - 1.0).max())
			if deviation <= self._config.max_token_deviation:
				assignments = pd.Series(folds, index=groups.index)
				print(
					f"Token balancing: {swaps:,} prompt-group swaps; max deviation {deviation:.4%}."
				)
				return TokenBalanceResult(
					folds=data["group_id"].map(assignments).astype("int16"),
					pair_tokens=pair_tokens,
					swaps=swaps,
				)
			if swaps == self.MAX_SWAPS:
				break
			high, low = int(totals.argmax()), int(totals.argmin())
			swap = self._best_swap(
				folds,
				tokens,
				indices_by_class,
				high,
				low,
				int(totals[high] - totals[low]),
			)
			if swap is None:
				break
			left, right, delta = swap
			folds[left], folds[right] = low, high
			totals[high] -= delta
			totals[low] += delta

		raise DataPreparationError(
			f"Could not balance fold tokens within {self._config.max_token_deviation:.2%} "
			f"of their mean after {swaps:,} swaps (max deviation {deviation:.4%}) "
			"without changing class counts or splitting prompt groups. "
			"The processed dataset was not replaced."
		)

	def _count_tokens(self, data: pd.DataFrame, cache_dir: Path) -> np.ndarray:
		serializer = ConversationSerializer()
		records = tuple(
			serializer.serialize_pair(
				int(row.id), row.prompt, row.response_a, row.response_b
			)
			for row in data.itertuples(index=False)
		)
		try:
			base_config = AutoConfig.from_pretrained(
				self._config.tokenizer_name, revision=self._config.tokenizer_revision
			)
			revision = (
				getattr(base_config, "_commit_hash", None)
				or self._config.tokenizer_revision
			)
			cache = TokenizationCache.from_pretrained(
				cache_dir, self._config.tokenizer_name, revision=revision
			)
		except Exception as error:
			raise DataPreparationError(
				f"Could not prepare fold tokenizer {self._config.tokenizer_name}: {error}"
			) from error
		print(
			f"Counting canonical A/B tokens: {self._config.tokenizer_name}; revision: {revision}",
			flush=True,
		)
		encoded = cache.encode(
			[text for row in records for text in (row.response_a, row.response_b)],
			add_special_tokens=True,
			truncation=False,
		)
		return np.diff(encoded.offsets).reshape(-1, 2).sum(axis=1)

	@staticmethod
	def _best_swap(
		folds: np.ndarray,
		tokens: np.ndarray,
		buckets: tuple[np.ndarray, ...],
		high: int,
		low: int,
		gap: int,
	) -> tuple[int, int, int] | None:
		"""Find the equal-class group swap closest to halving the token gap."""
		best_distance = gap
		best: tuple[int, int, int] | None = None
		for indices in buckets:
			left = indices[folds[indices] == high]
			right = indices[folds[indices] == low]
			if not len(left) or not len(right):
				continue
			right = right[np.argsort(tokens[right], kind="stable")]
			positions = np.searchsorted(tokens[right], tokens[left] - gap / 2)
			for candidates in (positions, positions - 1):
				partners = right[np.clip(candidates, 0, len(right) - 1)]
				deltas = tokens[left] - tokens[partners]
				distances = np.abs(gap - 2 * deltas)
				index = int(distances.argmin())
				distance = int(distances[index])
				if distance < best_distance:
					best_distance = distance
					best = int(left[index]), int(partners[index]), int(deltas[index])
		return best
