"""Scope submission SDPA compatibility to the model execution process."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from torch.nn.attention import SDPBackend, sdpa_kernel

from ...config import GPUConfig


class SubmissionAttention:
	"""Expand K/V heads for fused SDPA submission on GPUs such as the T4."""

	def __init__(self, config: GPUConfig) -> None:
		self._config = config

	@contextmanager
	def activate(self) -> Iterator[None]:
		"""Disable native GQA and math only while an SDPA submission is running."""
		if self._config.attention_implementation != "sdpa":
			yield
			return

		from transformers.integrations import sdpa_attention

		original_use_gqa = sdpa_attention.use_gqa_in_sdpa
		try:
			# Native GQA excludes memory-efficient SDPA on PyTorch 2.11/T4.
			# Transformers instead repeats K/V heads when this predicate is false.
			sdpa_attention.use_gqa_in_sdpa = self._disable_native_gqa
			with sdpa_kernel(
				[
					SDPBackend.FLASH_ATTENTION,
					SDPBackend.EFFICIENT_ATTENTION,
					SDPBackend.CUDNN_ATTENTION,
				]
			):
				yield
		finally:
			sdpa_attention.use_gqa_in_sdpa = original_use_gqa

	@staticmethod
	def _disable_native_gqa(*args: object, **kwargs: object) -> bool:
		return False
