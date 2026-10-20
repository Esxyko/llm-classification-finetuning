"""Reuse canonical tokens and features, orient rows, and batch within update groups."""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, RandomSampler, Sampler
from transformers import AutoTokenizer

from ...errors import ModelExecutionError
from ..common import PairwiseText


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
	return (
		reference["label"].to_numpy(dtype=np.int64)
		if "label" in reference
		else np.full(len(reference), -1, dtype=np.int64)
	)


@dataclass(frozen=True, slots=True)
class BranchBatch:
	"""Right-padded tokens with CPU-computed pooling positions and an optional mask."""

	input_ids: Tensor
	attention_mask: Tensor | None
	last_positions: Tensor

	def pin_memory(self) -> BranchBatch:
		return BranchBatch(
			self.input_ids.pin_memory(),
			self.attention_mask.pin_memory()
			if self.attention_mask is not None
			else None,
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


class PairwiseTokenStore:
	"""Tokenize every canonical branch once into a temporary compact CPU buffer."""

	def __init__(
		self,
		records: tuple[PairwiseText, ...],
		model_name: str,
		max_length: int,
		revision: str | None,
	) -> None:
		self._records = records
		self.revision = revision
		self.positions = {record.row_id: index for index, record in enumerate(records)}
		if len(self.positions) != len(records):
			raise ModelExecutionError(
				"Canonical LoRA conversations must have unique IDs."
			)
		tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
		self.pad_token_id = tokenizer.pad_token_id
		if self.pad_token_id is None:
			self.pad_token_id = tokenizer.eos_token_id
		if self.pad_token_id is None:
			raise ModelExecutionError(
				"The configured tokenizer has neither a pad token nor an EOS token."
			)
		tokens = array("i")
		self.offsets = np.empty(2 * len(records) + 1, dtype=np.int64)
		self.offsets[0] = 0
		longest = 0
		for start in range(0, len(records), 256):
			group = records[start : start + 256]
			encoded = tokenizer(
				[
					text
					for record in group
					for text in (record.response_a, record.response_b)
				],
				padding=False,
				truncation=False,
				return_attention_mask=False,
				return_token_type_ids=False,
				verbose=False,
			)
			for offset, input_ids in enumerate(encoded["input_ids"]):
				branch_index = 2 * start + offset
				length = len(input_ids)
				row_id = records[branch_index // 2].row_id
				branch_name = "AB"[branch_index % 2]
				if length == 0:
					raise ModelExecutionError(
						f"Row {row_id} branch {branch_name} contains no tokens."
					)
				if length > max_length:
					raise ModelExecutionError(
						f"Row {row_id} branch {branch_name} has {length:,} tokens, exceeding "
						f"qwen3_4b_lora.max_length={max_length:,}. Increase max_length to "
						f"at least {length:,}; LoRA inputs are never truncated."
					)
				longest = max(longest, length)
				tokens.extend(input_ids)
				self.offsets[branch_index + 1] = len(tokens)
		self.tokens = np.frombuffer(tokens, dtype=np.int32)
		self.lengths = np.diff(self.offsets).reshape(-1, 2)
		print(
			f"Tokenized {len(records):,} unique LoRA pairs once; "
			f"longest branch={longest:,} tokens (no truncation).",
			flush=True,
		)

	def matches(self, records: tuple[PairwiseText, ...], revision: str | None) -> bool:
		return self._records is records and self.revision == revision


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


class PairwiseCollator:
	"""Pad cached A/B token views independently and pool without GPU mask scans."""

	def __init__(self, pad_token_id: int) -> None:
		self._pad_token_id = pad_token_id

	def __call__(self, rows: list[tuple[np.ndarray, np.ndarray, int]]) -> PairwiseBatch:
		branches: list[BranchBatch] = []
		for index in (0, 1):
			lengths = [len(row[index]) for row in rows]
			longest = max(lengths)
			input_ids = torch.full(
				(len(rows), longest), self._pad_token_id, dtype=torch.long
			)
			mask = torch.zeros_like(input_ids) if min(lengths) != longest else None
			for row_index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
				input_ids[row_index, :length].copy_(torch.from_numpy(row[index]))
				if mask is not None:
					mask[row_index, :length] = 1
			branches.append(
				BranchBatch(
					input_ids, mask, torch.tensor(lengths, dtype=torch.long) - 1
				)
			)
		return PairwiseBatch(
			branches[0],
			branches[1],
			torch.tensor([row[2] for row in rows], dtype=torch.long),
		)


class OptimizerGroupBatchSampler(Sampler[list[int]]):
	"""Split the original shuffled update groups into fixed or token-budget batches."""

	def __init__(
		self,
		dataset: PairwiseTokenDataset,
		batch_size: int,
		accumulation_steps: int,
		token_budget: int,
		generator: torch.Generator,
	) -> None:
		self._dataset = dataset
		self._sampler = RandomSampler(dataset, generator=generator)
		self._batch_size = batch_size
		self._group_size = batch_size * accumulation_steps
		self._token_budget = token_budget

	def __iter__(self) -> Iterator[list[int]]:
		indices = iter(self._sampler)
		while group := list(islice(indices, self._group_size)):
			if self._token_budget == 0:
				for start in range(0, len(group), self._batch_size):
					yield group[start : start + self._batch_size]
				continue
			batch: list[int] = []
			max_a = max_b = 0
			for index in group:
				a, b = self._dataset.branch_lengths(index)
				if (
					batch
					and (len(batch) + 1) * (max(max_a, a) + max(max_b, b))
					> self._token_budget
				):
					yield batch
					batch = []
					max_a = max_b = 0
				batch.append(index)
				max_a, max_b = max(max_a, a), max(max_b, b)
			if batch:
				yield batch


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
	"""Share one CPU feature pair per ID across warmup training and validation."""

	def __init__(self, row_ids: tuple[int, ...], features: Tensor) -> None:
		self._positions = {row_id: index for index, row_id in enumerate(row_ids)}
		self._features = features

	def dataset(self, reference: pd.DataFrame) -> PairwiseFeatureDataset:
		return PairwiseFeatureDataset(reference, self._features, self._positions)
