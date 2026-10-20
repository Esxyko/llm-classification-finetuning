"""Prepare token identities on CPU and lazily encode missing frozen branches."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from ...config import GPUConfig, ModelConfig
from ...errors import ModelExecutionError
from .data import PairwiseText
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
		refresh_cache: bool = False,
	) -> None:
		self._config = model_config
		self._gpu = gpu_config
		self._device = device
		self._refresh_cache = refresh_cache
		self._model: PairwiseQwenClassifier | None = None
		try:
			base_config = AutoConfig.from_pretrained(model_config.model_name)
			self.base_revision = getattr(base_config, "_commit_hash", None)
			self.hidden_size = int(base_config.hidden_size)
			local_path = Path(model_config.model_name)
			self.local_weights_sha256 = (
				self._local_weights_key(local_path) if local_path.is_dir() else None
			)
			if self.local_weights_sha256 is None and self.base_revision is None:
				raise ModelExecutionError(
					"Could not resolve an immutable backbone revision for embedding caching."
				)
			self._token_cache = TokenizationCache.from_pretrained(
				token_cache_dir, model_config.model_name, revision=self.base_revision
			)
			self._token_cache.tokenizer.truncation_side = "right"
		except ModelExecutionError:
			raise
		except Exception as error:
			raise ModelExecutionError(
				f"Could not prepare {model_config.model_name}: {error}"
			) from error

	@staticmethod
	def _local_weights_key(path: Path) -> str:
		"""Hash the actual local checkpoint and configuration, excluding its location."""
		root = path.resolve()
		files = [root / "config.json"]
		for name in (
			"model.safetensors",
			"model.safetensors.index.json",
			"pytorch_model.bin",
			"pytorch_model.bin.index.json",
		):
			checkpoint = root / name
			if not checkpoint.is_file():
				continue
			files.append(checkpoint)
			if name.endswith(".index.json"):
				index = json.loads(checkpoint.read_text(encoding="utf-8"))
				files.extend(
					root / name for name in sorted(set(index["weight_map"].values()))
				)
			break
		else:
			raise ModelExecutionError(
				f"No supported local backbone checkpoint found in {root}."
			)
		digest = hashlib.sha256()
		for file_path in files:
			if (
				not file_path.is_relative_to(root)
				or ".." in file_path.relative_to(root).parts
			):
				raise ModelExecutionError(
					"Local backbone checkpoint shards must remain inside the model directory."
				)
			# Keep snapshot-relative names; Hugging Face snapshots may symlink to blobs.
			name = file_path.relative_to(root).as_posix().encode("utf-8")
			digest.update(len(name).to_bytes(8, "big"))
			digest.update(name)
			digest.update(file_path.stat().st_size.to_bytes(8, "big"))
			with file_path.open("rb") as stream:
				for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
					digest.update(chunk)
		return digest.hexdigest()

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
		batch_size = self._config.extraction_batch_size
		total_batches = math.ceil(len(indices) / batch_size)
		try:
			model = self._load_model()
			for batch_number, start in enumerate(
				range(0, len(indices), batch_size), start=1
			):
				batch = list(indices[start : start + batch_size])
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
				"extraction_batch_size or max_length, or select a lower-precision gpu.precision."
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
