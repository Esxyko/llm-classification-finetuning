"""Plan deterministic extraction micro-batches by padded branch-token cost."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class ExtractionBatchSampler(Sampler[list[int]]):
	"""Choose contiguous DP or best-fit, retaining fixed original branch indices."""

	def __init__(
		self,
		lengths: Sequence[int],
		budget: int,
		*,
		indices: Sequence[int] | None = None,
	) -> None:
		if budget <= 0:
			raise ValueError("Extraction token budget must be positive.")
		self._lengths = lengths
		self._budget = budget
		ordered = sorted(
			range(len(lengths)) if indices is None else indices,
			key=lambda index: (-int(lengths[index]), index),
		)
		eligible: list[int] = []
		oversized: list[tuple[int, ...]] = []
		for index in ordered:
			length = int(lengths[index])
			if length <= 0:
				raise ValueError("Extraction branches must contain tokens.")
			if length > budget:
				oversized.append((index,))
			else:
				eligible.append(index)
		if not eligible:
			self._batches = tuple(oversized)
		elif len(eligible) * int(lengths[eligible[0]]) <= budget:
			# This is also the exact best-fit result, including its branch order.
			self._batches = (*oversized, tuple(eligible))
		else:
			# Oversized singletons add the same fixed count and cost to both choices.
			best_fit, padded_tokens = self._best_fit(eligible)
			candidate = self._pack_contiguous(eligible, (len(best_fit), padded_tokens))
			self._batches = tuple(oversized) + (
				best_fit if candidate is None else candidate
			)

	def _best_fit(self, indices: list[int]) -> tuple[tuple[tuple[int, ...], ...], int]:
		"""Retain descending best-fit packing and its total padded branch tokens."""
		batches: list[list[int]] = []
		maximum = 0
		padded_tokens = 0
		for index in indices:
			length = int(self._lengths[index])
			# Descending lengths never increase a batch's maximum. Once a batch
			# cannot accept one more branch, no later branch can fit it either.
			# Thus best-fit has at most one open batch and needs no batch scan.
			if not batches or (len(batches[-1]) + 1) * maximum > self._budget:
				batches.append([index])
				maximum = length
			else:
				batches[-1].append(index)
			padded_tokens += maximum
		return tuple(tuple(batch) for batch in batches), padded_tokens

	def _pack_contiguous(
		self, indices: list[int], incumbent: tuple[int, int]
	) -> tuple[tuple[int, ...], ...] | None:
		"""Reconstruct optimal descending suffix cuts only when they beat best-fit."""
		lengths = [int(self._lengths[index]) for index in indices]
		rows = len(indices)
		batch_counts = [0] * (rows + 1)
		padded_tokens = [0] * (rows + 1)
		next_cuts = [0] * rows
		budget = self._budget
		for start in range(rows - 1, -1, -1):
			# Descending lengths make every interval's maximum its first length.
			maximum = lengths[start]
			best_count = 1 + batch_counts[start + 1]
			best_tokens = maximum + padded_tokens[start + 1]
			best_cut = start + 1
			last_cut = min(rows, start + budget // maximum)
			for cut in range(start + 2, last_cut + 1):
				count = 1 + batch_counts[cut]
				tokens = (cut - start) * maximum + padded_tokens[cut]
				if count < best_count or (count == best_count and tokens < best_tokens):
					best_count, best_tokens, best_cut = count, tokens, cut
			batch_counts[start] = best_count
			padded_tokens[start] = best_tokens
			next_cuts[start] = best_cut
		if (batch_counts[0], padded_tokens[0]) >= incumbent:
			return None
		batches: list[tuple[int, ...]] = []
		start = 0
		while start < rows:
			stop = next_cuts[start]
			batches.append(tuple(indices[start:stop]))
			start = stop
		return tuple(batches)

	def __iter__(self) -> Iterator[list[int]]:
		for batch in self._batches:
			yield list(batch)

	def __len__(self) -> int:
		return len(self._batches)
