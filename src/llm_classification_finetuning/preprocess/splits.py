"""Build and persist a leakage-safe, stratified train/validation split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MAX_VALIDATION_ROW_DEVIATION, ValidationConfig
from ..errors import DataPreparationError
from ..models.common import CacheStorage, TrainingInputLoader
from .augmentation import ABAugmenter
from .token_balance import SplitTokenBalancer


@dataclass(frozen=True, slots=True)
class SplitSummary:
	"""Summary statistics for one train/validation partition."""

	split: str
	rows: int
	groups: int
	class_counts: tuple[int, int, int]
	source_rows: int
	source_class_counts: tuple[int, int, int]
	tokens_total: int
	tokens_augmented: int
	row_share_pct: float
	row_deviation_pct: float
	token_share_pct: float
	token_deviation_pct: float


@dataclass(frozen=True, slots=True)
class SplitPreparationResult:
	"""Metadata describing a completed processed dataset."""

	output_path: Path
	rows: int
	groups: int
	splits: tuple[SplitSummary, ...]


class SplitPreprocessor:
	"""Stratify complete prompts, balance tokens, and create symmetric A/B rows."""

	MAX_ROW_DEVIATION = MAX_VALIDATION_ROW_DEVIATION

	TARGET_COLUMNS = ("winner_model_a", "winner_model_b", "winner_tie")
	LABEL_MAPPING = {
		"winner_model_a": 0,
		"winner_model_b": 1,
		"winner_tie": 2,
	}

	def __init__(self, config: ValidationConfig, *, cache_dir: Path) -> None:
		self._config = config
		self._cache_dir = cache_dir
		self._augmenter = ABAugmenter()
		self._token_balancer = SplitTokenBalancer(config)

	def prepare(self, train_path: Path, output_path: Path) -> SplitPreparationResult:
		"""Publish only after grouped splitting and token balancing succeed."""
		data = self._read_training_data(train_path)

		target_values = data.loc[:, list(self.TARGET_COLUMNS)]
		if (
			data.empty
			or data["id"].isna().any()
			or data["id"].duplicated().any()
			or not target_values.isin((0, 1)).all().all()
			or not target_values.sum(axis=1).eq(1).all()
		):
			raise DataPreparationError(
				"Training data needs unique, nonmissing IDs and exactly one winner per row."
			)
		data["label"] = (
			target_values.idxmax(axis="columns").map(self.LABEL_MAPPING).astype("int8")
		)
		data["group_id"] = pd.Series(self._create_group_ids(data), dtype="string")

		unique_groups = int(data["group_id"].nunique())
		data["split"] = self._assign_split(data)
		balanced = self._token_balancer.balance(data, self._cache_dir / "token_cache")
		data["split"] = balanced.splits
		data = self._augmenter.augment(data)
		TrainingInputLoader.validate(
			data, validation_fraction=self._config.validation_fraction
		)
		summaries = self._summarize_splits(data, balanced.pair_tokens)
		self._write_parquet(data, output_path)

		return SplitPreparationResult(
			output_path=output_path,
			rows=len(data),
			groups=unique_groups,
			splits=summaries,
		)

	@staticmethod
	def _read_training_data(train_path: Path) -> pd.DataFrame:
		try:
			return pd.read_csv(train_path)
		except Exception as error:
			raise DataPreparationError(
				f"Could not read training data from {train_path}: {error}"
			) from error

	def _create_group_ids(self, data: pd.DataFrame) -> list[str]:
		return [self._hash_prompt(prompt) for prompt in data["prompt"]]

	@staticmethod
	def _hash_prompt(prompt: str) -> str:
		canonical_prompt = json.dumps(
			json.loads(prompt),
			ensure_ascii=False,
			sort_keys=True,
			separators=(",", ":"),
			allow_nan=False,
		)

		return hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest()

	def _assign_split(self, data: pd.DataFrame) -> pd.Series:
		"""Reserve whole multirow prompts and fill class quotas with singletons."""
		counts = pd.crosstab(data["group_id"], data["label"]).reindex(
			columns=(0, 1, 2), fill_value=0
		)
		vectors = counts.to_numpy(dtype=np.int64)
		sizes = vectors.sum(axis=1)
		fraction = self._config.validation_fraction
		targets = vectors.sum(axis=0) * fraction
		quotas = np.floor(targets).astype(np.int64)
		target_rows = int(np.floor(len(data) * fraction + 0.5))
		remainders = np.argsort(-(targets - quotas), kind="stable")
		quotas[remainders[: target_rows - int(quotas.sum())]] += 1
		remaining = quotas.copy()
		selected = np.zeros(len(counts), dtype=np.bool_)
		rng = np.random.default_rng(self._config.random_state)
		multi = rng.permutation(np.flatnonzero(sizes > 1))
		available_singletons = vectors[sizes == 1].sum(axis=0)
		remaining_multi = vectors[multi].sum(axis=0)
		for index in multi:
			vector = vectors[index]
			remaining_multi -= vector
			needed = bool((remaining > available_singletons + remaining_multi).any())
			if (needed or rng.random() < fraction) and (vector <= remaining).all():
				selected[index] = True
				remaining -= vector
		for label in range(3):
			pool = rng.permutation(
				np.flatnonzero((sizes == 1) & (vectors[:, label] == 1))
			)
			if remaining[label] > len(pool):
				raise DataPreparationError(
					"Could not fill stratified validation quotas without splitting prompt groups. "
					"The processed dataset was not replaced."
				)
			selected[pool[: int(remaining[label])]] = True
		validation_rows = int(sizes[selected].sum())
		deviation = abs(validation_rows / (len(data) * fraction) - 1.0)
		if not 0 < validation_rows < len(data) or deviation >= self.MAX_ROW_DEVIATION:
			raise DataPreparationError(
				f"Validation row deviation {deviation:.4%} must be below {self.MAX_ROW_DEVIATION:.2%}. "
				"The processed dataset was not replaced."
			)
		assignments = pd.Series(
			np.where(selected, "validation", "train"), index=counts.index
		)
		return data["group_id"].map(assignments).astype("string")

	def _summarize_splits(
		self, data: pd.DataFrame, pair_tokens: np.ndarray
	) -> tuple[SplitSummary, ...]:
		summaries: list[SplitSummary] = []
		source = data.loc[~data["is_swapped"]].assign(_tokens=pair_tokens)
		total_tokens = int(pair_tokens.sum())
		for split, fraction in (
			("train", 1 - self._config.validation_fraction),
			("validation", self._config.validation_fraction),
		):
			split_data = data.loc[data["split"].eq(split)]
			counts = split_data["label"].value_counts().reindex((0, 1, 2), fill_value=0)
			split_source = source.loc[source["split"].eq(split)]
			source_counts = (
				split_source["label"].value_counts().reindex((0, 1, 2), fill_value=0)
			)
			tokens = int(split_source["_tokens"].sum())
			summaries.append(
				SplitSummary(
					split=split,
					rows=len(split_data),
					groups=int(split_data["group_id"].nunique()),
					class_counts=(
						int(counts.loc[0]),
						int(counts.loc[1]),
						int(counts.loc[2]),
					),
					source_rows=len(split_source),
					source_class_counts=tuple(int(value) for value in source_counts),
					tokens_total=tokens,
					tokens_augmented=tokens * 2,
					row_share_pct=len(split_source) / len(source) * 100,
					row_deviation_pct=(len(split_source) / (len(source) * fraction) - 1)
					* 100,
					token_share_pct=tokens / total_tokens * 100,
					token_deviation_pct=(tokens / (total_tokens * fraction) - 1) * 100,
				)
			)
		return tuple(summaries)

	@staticmethod
	def _write_parquet(data: pd.DataFrame, output_path: Path) -> None:
		try:
			with CacheStorage(output_path).open("w") as session:
				data.to_parquet(
					session.path,
					engine="pyarrow",
					compression="zstd",
					index=False,
				)
		except Exception as error:
			raise DataPreparationError(
				f"Could not write processed Parquet file {output_path}: {error}"
			) from error
