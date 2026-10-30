"""Orient canonical token/feature views and independently pad conversation branches."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ...errors import ModelExecutionError
from ..common import PairwiseTokenStore


def _row_indices(row_ids: Iterable[int], positions: dict[int, int]) -> np.ndarray:
	"""Resolve reference rows without changing their order or duplicating storage."""
	try:
		return np.fromiter(
			(positions[int(row_id)] for row_id in row_ids), dtype=np.int64
		)
	except KeyError as error:
		raise ModelExecutionError(
			f"No canonical conversation exists for ID {error.args[0]}."
		) from error


def _labels(reference: pd.DataFrame) -> np.ndarray:
	return reference["label"].to_numpy(dtype=np.int64)


@dataclass(frozen=True, slots=True)
class BranchBatch:
	"""Right-padded tokens with CPU pooling positions for causal encoding."""

	input_ids: Tensor
	last_positions: Tensor

	def pin_memory(self) -> BranchBatch:
		return BranchBatch(
			self.input_ids.pin_memory(),
			self.last_positions.pin_memory(),
		)


@dataclass(frozen=True, slots=True)
class PairwiseBatch:
	"""Independently padded A/B branches and their orientation-specific labels."""

	branch_a: BranchBatch
	branch_b: BranchBatch
	labels: Tensor

	def pin_memory(self) -> PairwiseBatch:
		"""Allow DataLoader to pin the tensors inside this custom batch."""
		return PairwiseBatch(
			self.branch_a.pin_memory(),
			self.branch_b.pin_memory(),
			self.labels.pin_memory(),
		)


class PairwiseTokenDataset(Dataset[tuple[np.ndarray, np.ndarray, int]]):
	"""Orient cached token views in the exact processed-reference row order."""

	def __init__(self, reference: pd.DataFrame, tokens: PairwiseTokenStore) -> None:
		self._tokens = tokens.tokens
		self._offsets = tokens.offsets
		self._lengths = tokens.lengths
		self._indices = _row_indices(reference["id"], tokens.positions)
		self._swapped = reference["is_swapped"].to_numpy(dtype=np.bool_)
		self._labels = _labels(reference)

	def __len__(self) -> int:
		return len(self._indices)

	def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, int]:
		position = 2 * int(self._indices[index])
		a_start, b_start, stop = self._offsets[position : position + 3]
		a, b = self._tokens[a_start:b_start], self._tokens[b_start:stop]
		if self._swapped[index]:
			a, b = b, a
		return a, b, int(self._labels[index])

	def branch_lengths(self, index: int) -> tuple[int, int]:
		a, b = self._lengths[self._indices[index]]
		return (int(b), int(a)) if self._swapped[index] else (int(a), int(b))


class BranchTokenDataset(Dataset[np.ndarray]):
	"""Flatten canonical IDs into A/B branch views in A0, B0, A1, B1 order."""

	def __init__(self, row_ids: Iterable[int], tokens: PairwiseTokenStore) -> None:
		self._tokens = tokens.tokens
		self._offsets = tokens.offsets
		self._indices = _row_indices(row_ids, tokens.positions)

	def __len__(self) -> int:
		return 2 * len(self._indices)

	def __getitem__(self, index: int) -> np.ndarray:
		position = 2 * int(self._indices[index // 2]) + index % 2
		start, stop = self._offsets[position : position + 2]
		return self._tokens[start:stop]

	def branch_length(self, index: int) -> int:
		"""Read a canonical branch length without materializing its token view."""
		position = 2 * int(self._indices[index // 2]) + index % 2
		return int(self._offsets[position + 1] - self._offsets[position])


class BranchCollator:
	"""Right-pad any A/B branch mix and compute CPU pooling positions."""

	def __init__(self, pad_token_id: int) -> None:
		self._pad_token_id = pad_token_id

	def __call__(self, rows: list[np.ndarray]) -> BranchBatch:
		lengths = [len(row) for row in rows]
		input_ids = torch.full(
			(len(rows), max(lengths)), self._pad_token_id, dtype=torch.long
		)
		for row_index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
			input_ids[row_index, :length].copy_(torch.from_numpy(row))
		return BranchBatch(input_ids, torch.tensor(lengths, dtype=torch.long) - 1)


class PairwiseCollator:
	"""Right-pad A/B tokens without masks and compute their last real positions."""

	def __init__(self, pad_token_id: int) -> None:
		self._branch_collator = BranchCollator(pad_token_id)

	def __call__(self, rows: list[tuple[np.ndarray, np.ndarray, int]]) -> PairwiseBatch:
		return PairwiseBatch(
			self._branch_collator([row[0] for row in rows]),
			self._branch_collator([row[1] for row in rows]),
			torch.tensor([row[2] for row in rows], dtype=torch.long),
		)


class PairwiseFeatureDataset(Dataset[tuple[Tensor, int]]):
	"""Select and orient frozen features without materializing augmented copies."""

	def __init__(
		self, reference: pd.DataFrame, features: Tensor, positions: dict[int, int]
	) -> None:
		self._features = features
		self._indices = _row_indices(reference["id"], positions)
		self._swapped = reference["is_swapped"].to_numpy(dtype=np.bool_)
		self._labels = _labels(reference)

	def __len__(self) -> int:
		return len(self._indices)

	def __getitem__(self, index: int) -> tuple[Tensor, int]:
		features = self._features[int(self._indices[index])]
		if self._swapped[index]:
			features = features.flip(0)
		return features, int(self._labels[index])


class PairwiseFeatureStore:
	"""Share one CPU feature pair per ID across oriented head-training/inference views."""

	def __init__(self, row_ids: tuple[int, ...], features: Tensor) -> None:
		self._positions = {row_id: index for index, row_id in enumerate(row_ids)}
		self._features = features

	def dataset(self, reference: pd.DataFrame) -> PairwiseFeatureDataset:
		return PairwiseFeatureDataset(reference, self._features, self._positions)
