"""Build and persist leakage-safe stratified group folds."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..config import CrossValidationConfig
from ..errors import DataPreparationError
from .augmentation import ABAugmenter
from .token_balance import FoldTokenBalancer


@dataclass(frozen=True, slots=True)
class FoldSummary:
	"""Summary statistics for one validation fold."""

	fold: int
	rows: int
	groups: int
	class_counts: tuple[int, int, int]
	source_rows: int
	source_class_counts: tuple[int, int, int]
	tokens_total: int
	tokens_augmented: int
	token_deviation_pct: float


@dataclass(frozen=True, slots=True)
class FoldPreparationResult:
	"""Metadata describing a completed processed dataset."""

	output_path: Path
	rows: int
	groups: int
	folds: tuple[FoldSummary, ...]


class FoldPreprocessor:
	"""Add labels and folds, then create symmetric A/B training examples."""

	TARGET_COLUMNS = ("winner_model_a", "winner_model_b", "winner_tie")
	LABEL_MAPPING = {
		"winner_model_a": 0,
		"winner_model_b": 1,
		"winner_tie": 2,
	}

	def __init__(self, config: CrossValidationConfig) -> None:
		self._config = config
		self._augmenter = ABAugmenter()
		self._token_balancer = FoldTokenBalancer(config)

	def prepare(self, train_path: Path, output_path: Path) -> FoldPreparationResult:
		"""Create folds, augment A/B order, and atomically write the artifact."""
		data = self._read_training_data(train_path)

		target_values = data.loc[:, list(self.TARGET_COLUMNS)]
		data["label"] = (
			target_values.idxmax(axis="columns").map(self.LABEL_MAPPING).astype("int8")
		)
		data["group_id"] = pd.Series(self._create_group_ids(data), dtype="string")

		unique_groups = int(data["group_id"].nunique())
		data["fold"] = self._assign_folds(data)
		balanced = self._token_balancer.balance(
			data, output_path.parent / "token_cache"
		)
		data["fold"] = balanced.folds
		data = self._augmenter.augment(data)
		summaries = self._summarize_folds(data, balanced.pair_tokens)
		self._write_parquet(data, output_path)

		return FoldPreparationResult(
			output_path=output_path,
			rows=len(data),
			groups=unique_groups,
			folds=summaries,
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

	def _assign_folds(self, data: pd.DataFrame) -> pd.Series:
		splitter = StratifiedGroupKFold(
			n_splits=self._config.n_splits,
			shuffle=True,
			random_state=self._config.random_state,
		)
		fold_assignments = np.full(len(data), -1, dtype=np.int16)

		splits = splitter.split(
			X=np.zeros((len(data), 1), dtype=np.uint8),
			y=data["label"],
			groups=data["group_id"],
		)
		for fold, (_, validation_indices) in enumerate(splits):
			fold_assignments[validation_indices] = fold

		return pd.Series(fold_assignments, index=data.index, dtype="int16")

	def _summarize_folds(
		self, data: pd.DataFrame, pair_tokens: np.ndarray
	) -> tuple[FoldSummary, ...]:
		summaries: list[FoldSummary] = []
		source = data.loc[~data["is_swapped"]].assign(_tokens=pair_tokens)
		mean_tokens = float(pair_tokens.sum()) / self._config.n_splits
		for fold in range(self._config.n_splits):
			fold_data = data.loc[data["fold"].eq(fold)]
			counts = fold_data["label"].value_counts().reindex((0, 1, 2), fill_value=0)
			fold_source = source.loc[source["fold"].eq(fold)]
			source_counts = (
				fold_source["label"].value_counts().reindex((0, 1, 2), fill_value=0)
			)
			tokens = int(fold_source["_tokens"].sum())
			summaries.append(
				FoldSummary(
					fold=fold,
					rows=len(fold_data),
					groups=int(fold_data["group_id"].nunique()),
					class_counts=(
						int(counts.loc[0]),
						int(counts.loc[1]),
						int(counts.loc[2]),
					),
					source_rows=len(fold_source),
					source_class_counts=tuple(int(value) for value in source_counts),
					tokens_total=tokens,
					tokens_augmented=tokens * 2,
					token_deviation_pct=(tokens / mean_tokens - 1.0) * 100,
				)
			)
		return tuple(summaries)

	@staticmethod
	def _write_parquet(data: pd.DataFrame, output_path: Path) -> None:
		try:
			output_path.parent.mkdir(parents=True, exist_ok=True)
		except OSError as error:
			raise DataPreparationError(
				f"Could not create processed data directory {output_path.parent}: {error}"
			) from error

		temporary_path: Path | None = None
		try:
			with tempfile.NamedTemporaryFile(
				dir=output_path.parent,
				prefix=f".{output_path.stem}-",
				suffix=".tmp.parquet",
				delete=False,
			) as temporary_file:
				temporary_path = Path(temporary_file.name)

			data.to_parquet(
				temporary_path,
				engine="pyarrow",
				compression="zstd",
				index=False,
			)
			os.replace(temporary_path, output_path)
		except Exception as error:
			raise DataPreparationError(
				f"Could not write processed Parquet file {output_path}: {error}"
			) from error
		finally:
			if temporary_path is not None:
				try:
					temporary_path.unlink(missing_ok=True)
				except OSError:
					pass
