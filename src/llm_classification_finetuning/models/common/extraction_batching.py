"""Plan deterministic extraction micro-batches by padded branch-token cost."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class ExtractionBatchSampler(Sampler[list[int]]):
	"""Best-fit decreasing packing with original branch indices and fixed batches."""

	def __init__(
		self,
		lengths: Sequence[int],
		budget: int,
		*,
		indices: Sequence[int] | None = None,
	) -> None:
		if budget <= 0:
			raise ValueError("Extraction token budget must be positive.")
		ordered = sorted(
			range(len(lengths)) if indices is None else indices,
			key=lambda index: (-int(lengths[index]), index),
		)
		batches: list[list[int]] = []
		maximum = 0
		for index in ordered:
			length = int(lengths[index])
			if length <= 0:
				raise ValueError("Extraction branches must contain tokens.")
			# Descending lengths never increase a batch's maximum. Once a batch
			# cannot accept one more branch, no later branch can fit it either.
			# Thus best-fit has at most one open batch and needs no batch scan.
			if not batches or (len(batches[-1]) + 1) * maximum > budget:
				batches.append([index])
				maximum = length
			else:
				batches[-1].append(index)
		self._batches = tuple(tuple(batch) for batch in batches)

	def __iter__(self) -> Iterator[list[int]]:
		for batch in self._batches:
			yield list(batch)

	def __len__(self) -> int:
		return len(self._batches)
