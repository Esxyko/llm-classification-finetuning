"""Scope submission SDPA compatibility to the model execution process."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from torch.nn.attention import SDPBackend, sdpa_kernel

from ...config import GPUConfig


class SubmissionAttention:
	"""Expand K/V heads and require Flash Attention for SDPA submission."""

	def __init__(self, config: GPUConfig) -> None:
		self._config = config

	@contextmanager
	def activate(self) -> Iterator[None]:
		"""Disable native GQA and select only Flash during an SDPA submission."""
		if self._config.attention_implementation != "sdpa":
			yield
			return

		from transformers.integrations import sdpa_attention

		original_use_gqa = sdpa_attention.use_gqa_in_sdpa
		try:
			# Preserve explicit K/V-head expansion for submission.
			sdpa_attention.use_gqa_in_sdpa = self._disable_native_gqa
			with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
				yield
		finally:
			sdpa_attention.use_gqa_in_sdpa = original_use_gqa

	@staticmethod
	def _disable_native_gqa(*args: object, **kwargs: object) -> bool:
		return False
