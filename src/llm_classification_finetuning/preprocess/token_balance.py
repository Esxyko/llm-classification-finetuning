"""Balance holdout token volume while preserving grouped class distributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig

from ..config import ValidationConfig
from ..errors import DataPreparationError
from ..models.common import ConversationSerializer, TokenizationCache


@dataclass(frozen=True, slots=True)
class TokenBalanceResult:
	"""Balanced assignments and canonical A+B token counts in source-row order."""

	splits: pd.Series
	pair_tokens: np.ndarray
	swaps: int


class SplitTokenBalancer:
	"""Swap whole prompt groups with identical three-class count vectors."""

	MAX_SWAPS = 1_000

	def __init__(self, config: ValidationConfig) -> None:
		self._config = config

	def balance(self, data: pd.DataFrame, cache_dir: Path) -> TokenBalanceResult:
		"""Count real tokens, then improve the initial stratified group assignment."""
		if data.empty:
			raise DataPreparationError(
				"Cannot balance a split for empty training data."
			)
		pair_tokens = self._count_tokens(data, cache_dir)
		groups = (
			data.assign(_tokens=pair_tokens)
			.groupby("group_id", sort=True)
			.agg(split=("split", "first"), tokens=("_tokens", "sum"))
		)
		class_counts = pd.crosstab(data["group_id"], data["label"]).reindex(
			index=groups.index, columns=(0, 1, 2), fill_value=0
		)
		validation = groups["split"].eq("validation").to_numpy(copy=True)
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
		validation_tokens = int(tokens[validation].sum())
		target_tokens = float(tokens.sum()) * self._config.validation_fraction
		if target_tokens <= 0:
			raise DataPreparationError(
				"Training inputs must contain tokens to balance the validation split."
			)

		for swaps in range(self.MAX_SWAPS + 1):
			deviation = abs(validation_tokens / target_tokens - 1.0)
			if deviation < self._config.max_token_deviation:
				assignments = pd.Series(
					np.where(validation, "validation", "train"), index=groups.index
				)
				print(
					f"Token balancing: {swaps:,} prompt-group swaps; validation deviation {deviation:.4%}."
				)
				return TokenBalanceResult(
					splits=data["group_id"].map(assignments).astype("string"),
					pair_tokens=pair_tokens,
					swaps=swaps,
				)
			if swaps == self.MAX_SWAPS:
				break
			from_validation = validation_tokens > target_tokens
			swap = self._best_swap(
				validation,
				tokens,
				indices_by_class,
				from_validation,
				abs(validation_tokens - target_tokens),
			)
			if swap is None:
				break
			left, right, delta = swap
			validation[left], validation[right] = not from_validation, from_validation
			validation_tokens += -delta if from_validation else delta

		raise DataPreparationError(
			f"Could not balance validation tokens below {self._config.max_token_deviation:.2%} "
			f"of their target after {swaps:,} swaps (deviation {deviation:.4%}) "
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
				f"Could not prepare split tokenizer {self._config.tokenizer_name}: {error}"
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
		validation: np.ndarray,
		tokens: np.ndarray,
		buckets: tuple[np.ndarray, ...],
		from_validation: bool,
		gap: float,
	) -> tuple[int, int, int] | None:
		"""Find an equal-class group swap closest to the holdout token deficit."""
		best_distance = gap
		best: tuple[int, int, int] | None = None
		for indices in buckets:
			left = indices[validation[indices] == from_validation]
			right = indices[validation[indices] != from_validation]
			if not len(left) or not len(right):
				continue
			right = right[np.argsort(tokens[right], kind="stable")]
			positions = np.searchsorted(tokens[right], tokens[left] - gap)
			for candidates in (positions, positions - 1):
				partners = right[np.clip(candidates, 0, len(right) - 1)]
				deltas = tokens[left] - tokens[partners]
				distances = np.abs(gap - deltas)
				index = int(distances.argmin())
				distance = float(distances[index])
				if distance < best_distance:
					best_distance = distance
					best = int(left[index]), int(partners[index]), int(deltas[index])
		return best
