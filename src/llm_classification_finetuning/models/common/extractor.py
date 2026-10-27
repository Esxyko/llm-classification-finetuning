"""Prepare token identities on CPU and lazily encode missing frozen branches."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from ...config import GPUConfig, ModelConfig
from ...errors import ModelExecutionError
from ...model_storage import load_pretrained
from .backbone import LocalBackboneFingerprint
from .data import PairwiseText
from .extraction_batching import ExtractionBatchSampler
from .hardware import GPUEnvironment
from .model import PairwiseQwenClassifier
from .token_cache import TokenizationCache, TokenizedTexts


class QwenEmbeddingExtractor:
	"""Resolve immutable inputs without loading backbone weights on cache hits."""

	def __init__(
		self,
		gpu_config: GPUConfig,
		model_config: ModelConfig,
		device: torch.device,
		*,
		token_cache_dir: Path,
		token_cache: TokenizationCache | None = None,
		revision: str | None = None,
		refresh_cache: bool = False,
		submit_devices: tuple[torch.device, torch.device] | None = None,
	) -> None:
		self._config = model_config
		self._gpu = gpu_config
		self._device = device
		self._refresh_cache = refresh_cache
		self._model: PairwiseQwenClassifier | None = None
		try:
			base_config = load_pretrained(
				AutoConfig.from_pretrained, model_config.model_name, revision=revision
			)
			self._device_map = (
				GPUEnvironment.submission_device_map(base_config, submit_devices)
				if submit_devices is not None
				else None
			)
			self.base_revision = getattr(base_config, "_commit_hash", None)
			self.hidden_size = int(base_config.hidden_size)
			local_path = Path(model_config.model_name)
			self.local_weights_sha256 = (
				LocalBackboneFingerprint.compute(local_path)
				if local_path.is_dir()
				else None
			)
			if self.local_weights_sha256 is None and self.base_revision is None:
				raise ModelExecutionError(
					"Could not resolve an immutable backbone revision for embedding caching."
				)
			self._token_cache = token_cache or TokenizationCache.from_pretrained(
				token_cache_dir, model_config.model_name, revision=self.base_revision
			)
			self._token_cache.tokenizer.truncation_side = "right"
		except ModelExecutionError:
			raise
		except Exception as error:
			raise ModelExecutionError(
				f"Could not prepare {model_config.model_name}: {error}"
			) from error

	def prepare(self, records: Sequence[PairwiseText]) -> TokenizedTexts:
		"""Resolve exact truncated inputs independently of IDs and dataset metadata."""
		if not records:
			raise ModelExecutionError(
				"No canonical records are available for extraction."
			)
		tokens = self._token_cache.encode(
			[
				text
				for record in records
				for text in (record.response_a, record.response_b)
			],
			add_special_tokens=False,
			truncation=True,
			max_length=self._config.max_length,
			refresh_cache=self._refresh_cache,
		)
		empty = np.flatnonzero(np.diff(tokens.offsets) <= 0)
		if len(empty):
			index = int(empty[0])
			raise ModelExecutionError(
				f"Row {records[index // 2].row_id} branch {'AB'[index % 2]} contains no tokens."
			)
		return tokens

	@staticmethod
	def input_keys(tokens: TokenizedTexts) -> list[str]:
		"""Use effective token IDs rather than tokenizer versions or configured caps."""
		return [
			hashlib.sha256(
				memoryview(tokens[index].astype("<i4", copy=False))
			).hexdigest()
			for index in range(len(tokens))
		]

	def _load_model(self) -> PairwiseQwenClassifier:
		if self._model is None:
			self._model = PairwiseQwenClassifier(
				model_name=self._config.model_name,
				classifier_hidden_size=self._config.hidden_size,
				dropout=self._config.dropout,
				device=self._device,
				dtype=GPUEnvironment.PRECISION_DTYPES[self._gpu.precision],
				attention_implementation=self._gpu.attention_implementation,
				revision=self.base_revision,
				device_map=self._device_map,
			)
			if self._model.backbone_hidden_size != self.hidden_size or (
				self.local_weights_sha256 is None
				and self._model.base_revision != self.base_revision
			):
				raise ModelExecutionError(
					"Loaded backbone does not match the embedding cache identity."
				)
		return self._model

	@torch.no_grad()
	def extract(
		self, tokens: TokenizedTexts, indices: Sequence[int], target: torch.Tensor
	) -> None:
		"""Fill one representative position per missing input, retaining CPU half storage."""
		if not indices:
			return
		try:
			batches = ExtractionBatchSampler(
				np.diff(tokens.offsets).tolist(),
				self._config.extraction_budget,
				indices=indices,
			)
			total_batches = len(batches)
			model = self._load_model()
			for batch_number, batch in enumerate(batches, start=1):
				input_ids, attention_mask = self._pad_batch(tokens, batch)
				pooled = model.encode_branch(input_ids, attention_mask).to(
					device="cpu", dtype=torch.float16
				)
				if not torch.isfinite(pooled).all():
					raise ModelExecutionError(
						"Frozen backbone produced non-finite embeddings."
					)
				target[batch] = pooled
				if batch_number == 1 or batch_number % 100 == 0:
					print(
						f"Encoded missing embedding branch batch {batch_number:,}/{total_batches:,}.",
						flush=True,
					)
		except torch.cuda.OutOfMemoryError as error:
			raise ModelExecutionError(
				"CUDA ran out of memory during embedding extraction. Lower the selected model's "
				"extraction_budget or max_length, or select a lower-precision gpu.precision."
			) from error
		except ModelExecutionError:
			raise
		except Exception as error:
			raise ModelExecutionError(
				f"Could not extract Qwen embeddings: {error}"
			) from error

	def _pad_batch(
		self, tokens: TokenizedTexts, indices: Sequence[int]
	) -> tuple[torch.Tensor, torch.Tensor]:
		length = max(len(tokens[index]) for index in indices)
		input_ids = torch.full(
			(len(indices), length), self._token_cache.pad_token_id, dtype=torch.long
		)
		mask = torch.zeros_like(input_ids)
		for row, index in enumerate(indices):
			sequence = tokens[index]
			input_ids[row, : len(sequence)].copy_(torch.from_numpy(sequence))
			mask[row, : len(sequence)] = 1
		return input_ids, mask
