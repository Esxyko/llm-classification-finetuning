"""Tokenize canonical conversation branches in their training orientation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ...errors import ModelExecutionError
from ..common import PairwiseText


@dataclass(frozen=True, slots=True)
class PairwiseBatch:
	"""Independently padded A/B branches and their orientation-specific labels."""

	branch_a: dict[str, Tensor]
	branch_b: dict[str, Tensor]
	labels: Tensor

	def pin_memory(self) -> PairwiseBatch:
		"""Allow DataLoader to pin the tensors inside this custom batch."""
		return PairwiseBatch(
			{name: value.pin_memory() for name, value in self.branch_a.items()},
			{name: value.pin_memory() for name, value in self.branch_b.items()},
			self.labels.pin_memory(),
		)


class PairwiseTextDataset(Dataset[tuple[str, str, int]]):
	"""Expose canonical texts in the exact processed-reference row order."""

	def __init__(
		self, reference: pd.DataFrame, records: tuple[PairwiseText, ...]
	) -> None:
		by_id = {record.row_id: record for record in records}
		try:
			self._records = tuple(by_id[int(row_id)] for row_id in reference["id"])
		except KeyError as error:
			raise ModelExecutionError(
				f"No canonical conversation exists for ID {error.args[0]}."
			) from error
		self._swapped = reference["is_swapped"].to_numpy(dtype=np.bool_)
		self._labels = (
			reference["label"].to_numpy(dtype=np.int64)
			if "label" in reference
			else np.full(len(reference), -1, dtype=np.int64)
		)

	def __len__(self) -> int:
		return len(self._records)

	def __getitem__(self, index: int) -> tuple[str, str, int]:
		record = self._records[index]
		a, b = record.response_a, record.response_b
		if self._swapped[index]:
			a, b = b, a
		return a, b, int(self._labels[index])


class PairwiseCollator:
	"""Dynamically pad each branch and apply the existing right truncation."""

	def __init__(
		self, model_name: str, max_length: int, revision: str | None = None
	) -> None:
		self._max_length = max_length
		self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
		self._tokenizer.padding_side = "right"
		self._tokenizer.truncation_side = "right"
		if self._tokenizer.pad_token_id is None:
			if self._tokenizer.eos_token_id is None:
				raise ModelExecutionError(
					"The configured tokenizer has neither a pad token nor an EOS token."
				)
			self._tokenizer.pad_token = self._tokenizer.eos_token

	def __call__(self, rows: list[tuple[str, str, int]]) -> PairwiseBatch:
		branches: list[dict[str, Tensor]] = []
		for index in (0, 1):
			tokens = self._tokenizer(
				[row[index] for row in rows],
				padding=True,
				truncation=True,
				max_length=self._max_length,
				return_tensors="pt",
			)
			branches.append(
				{name: tokens[name] for name in ("input_ids", "attention_mask")}
			)
		return PairwiseBatch(
			branches[0],
			branches[1],
			torch.tensor([row[2] for row in rows], dtype=torch.long),
		)
