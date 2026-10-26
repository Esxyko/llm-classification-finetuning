"""Prepare reusable global LoRA batches over independently cached text encodings."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from transformers import AutoConfig

from ...config import AppConfig, LoRAModelConfig, WarmupModelConfig
from ...errors import ModelExecutionError
from ...model_storage import load_pretrained
from ..common import (
	CacheStorage,
	LocalBackboneFingerprint,
	ModelData,
	ModelDataRepository,
	PairwiseTokenStore,
	TokenizationCache,
)
from ..profile import ModelProfile
from .data import (
	PairwiseBatch,
	PairwiseCollator,
	PairwiseTokenDataset,
)
from .profile import PROFILE
from .warmup_cache import WarmupCacheContext


@dataclass(frozen=True, slots=True)
class OptimizerGroupLayout:
	"""Merge a short tail into the last full update, retaining all training rows."""

	rows: int
	group_size: int

	def __post_init__(self) -> None:
		if self.rows <= 0 or self.group_size <= 0:
			raise ValueError("Optimizer groups require positive row and group sizes.")

	@property
	def update_count(self) -> int:
		return max(1, self.rows // self.group_size)

	@property
	def row_counts(self) -> tuple[int, ...]:
		full_groups = self.update_count - 1
		return (self.group_size,) * full_groups + (
			self.rows - full_groups * self.group_size,
		)


@dataclass(frozen=True, slots=True)
class TrainingBatchPlan:
	"""Cached packing groups containing independently padded micro-batches."""

	groups: tuple[tuple[tuple[int, ...], ...], ...]
	padded_tokens: int

	@property
	def batch_count(self) -> int:
		return sum(len(group) for group in self.groups)

	def validate(
		self, dataset: PairwiseTokenDataset, group_size: int, threshold: int
	) -> None:
		"""Reject missing, repeated, out-of-range, oversized, or misgrouped rows."""
		rows = len(dataset)
		if not rows or len(self.groups) != (rows + group_size - 1) // group_size:
			raise ValueError("Adapter batch plan has an invalid packing-group count.")
		seen = np.zeros(rows, dtype=np.bool_)
		padded_tokens = 0
		for group_index, group in enumerate(self.groups):
			if not group:
				raise ValueError("Adapter packing groups cannot be empty.")
			group_rows = 0
			for batch in group:
				if not batch:
					raise ValueError("Adapter micro-batches cannot be empty.")
				max_a = max_b = 0
				for index in batch:
					if (
						isinstance(index, bool)
						or not isinstance(index, int)
						or not 0 <= index < rows
						or seen[index]
					):
						raise ValueError(
							"Adapter batch plan has invalid or repeated rows."
						)
					seen[index] = True
					a, b = dataset.branch_lengths(index)
					max_a, max_b = max(max_a, a), max(max_b, b)
				cost = len(batch) * (max_a + max_b)
				if len(batch) > 1 and cost > threshold:
					raise ValueError("Adapter micro-batch exceeds the token threshold.")
				group_rows += len(batch)
				padded_tokens += cost
			if group_rows != min(group_size, rows - group_index * group_size):
				raise ValueError(
					"Adapter batch plan has an invalid packing-group size."
				)
		if (
			not bool(seen.all())
			or isinstance(self.padded_tokens, bool)
			or not isinstance(self.padded_tokens, int)
			or self.padded_tokens != padded_tokens
		):
			raise ValueError(
				"Adapter batch plan coverage or padded token count is invalid."
			)


class _TokenBudgetPacker:
	"""Best-fit pack rows by padded A/B cost, retaining oversized singletons."""

	def __init__(
		self,
		lengths: list[tuple[int, int]],
		identities: list[tuple[int, bool]],
		budget: int,
	) -> None:
		self._lengths = lengths
		self._identities = identities
		self._budget = budget

	def pack(self, indices: list[int]) -> tuple[tuple[int, ...], ...]:
		batches: list[list[int]] = []
		maxima: list[tuple[int, int]] = []
		for index in sorted(
			indices,
			key=lambda value: (-sum(self._lengths[value]), *self._identities[value]),
		):
			a, b = self._lengths[index]
			candidates = []
			for batch_index, (batch, (max_a, max_b)) in enumerate(
				zip(batches, maxima, strict=True)
			):
				cost = (len(batch) + 1) * (max(a, max_a) + max(b, max_b))
				if cost <= self._budget:
					candidates.append((self._budget - cost, batch_index))
			if candidates:
				_, batch_index = min(candidates)
				batches[batch_index].append(index)
				max_a, max_b = maxima[batch_index]
				maxima[batch_index] = max(a, max_a), max(b, max_b)
			else:
				batches.append([index])
				maxima.append((a, b))
		return tuple(tuple(batch) for batch in batches)


class _GlobalBatchPlanner:
	"""Compare three stable global sorts using best-fit packing inside each group."""

	VERSION = 1

	def __init__(
		self,
		dataset: PairwiseTokenDataset,
		reference: pd.DataFrame,
		group_size: int,
		threshold: int,
	) -> None:
		self._lengths = [dataset.branch_lengths(index) for index in range(len(dataset))]
		self._identities = [
			(int(row.id), bool(row.is_swapped))
			for row in reference.itertuples(index=False)
		]
		self._group_size = group_size
		self._packer = _TokenBudgetPacker(self._lengths, self._identities, threshold)

	def build(self) -> TrainingBatchPlan:
		best: TrainingBatchPlan | None = None
		for order in range(3):
			indices = sorted(
				range(len(self._lengths)),
				key=lambda index: self._sort_key(index, order),
			)
			groups = tuple(
				self._packer.pack(indices[start : start + self._group_size])
				for start in range(0, len(indices), self._group_size)
			)
			candidate = TrainingBatchPlan(
				groups,
				sum(
					len(batch)
					* sum(
						max(self._lengths[index][branch] for index in batch)
						for branch in (0, 1)
					)
					for group in groups
					for batch in group
				),
			)
			if best is None or (candidate.batch_count, candidate.padded_tokens) < (
				best.batch_count,
				best.padded_tokens,
			):
				best = candidate
		if best is None:
			raise ModelExecutionError("Cannot plan empty adapter training data.")
		return best

	def _sort_key(self, index: int, order: int) -> tuple[int, ...]:
		a, b = self._lengths[index]
		length_key = (a, b) if order == 0 else (b, a) if order == 1 else (a + b, a, b)
		return (*length_key, *self._identities[index])


class _TrainingBatchSampler(Sampler[list[int]]):
	"""Form final optimizer groups before repacking checkpointed singletons."""

	def __init__(
		self,
		plan: TrainingBatchPlan | None,
		dataset: PairwiseTokenDataset,
		reference: pd.DataFrame,
		config: LoRAModelConfig,
	) -> None:
		self._plan = plan
		self._layout = OptimizerGroupLayout(
			len(dataset),
			config.training_batch_size * config.gradient_accumulation_steps,
		)
		self._batch_size = config.training_batch_size
		self._threshold = config.gradient_checkpointing_threshold
		self._lengths = (
			[dataset.branch_lengths(index) for index in range(len(dataset))]
			if config.gradient_checkpointing_budget > 0
			else []
		)
		self._packer = (
			_TokenBudgetPacker(
				self._lengths,
				[
					(int(row.id), bool(row.is_swapped))
					for row in reference.itertuples(index=False)
				],
				config.gradient_checkpointing_budget,
			)
			if config.gradient_checkpointing_budget > 0
			else None
		)
		self._seed = config.random_state
		self._epoch = 0
		self._batches: tuple[tuple[int, ...], ...] | None = None

	def set_epoch(self, epoch: int) -> None:
		if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
			raise ValueError("The joint-training epoch must be a nonnegative integer.")
		self._epoch = epoch
		self._batches = None

	@property
	def group_row_counts(self) -> tuple[int, ...]:
		return self._layout.row_counts

	def _groups(
		self, generator: torch.Generator
	) -> Iterator[tuple[tuple[int, ...], ...]]:
		if self._plan is None:
			indices = torch.randperm(self._layout.rows, generator=generator).tolist()
			start = 0
			for rows in self._layout.row_counts:
				group = indices[start : start + rows]
				yield tuple(
					tuple(group[offset : offset + self._batch_size])
					for offset in range(0, rows, self._batch_size)
				)
				start += rows
			return
		order = torch.randperm(self._layout.update_count, generator=generator).tolist()
		for position, index in enumerate(order):
			group = self._plan.groups[index]
			if position == len(order) - 1 and len(self._plan.groups) > len(order):
				# Finalize step membership before merging any checkpointed singletons.
				group += self._plan.groups[-1]
			yield group

	def _repack_checkpointing(
		self, group: tuple[tuple[int, ...], ...]
	) -> tuple[tuple[int, ...], ...]:
		if self._packer is None:
			return group
		singletons = [
			batch[0]
			for batch in group
			if len(batch) == 1
			and (self._threshold == 0 or sum(self._lengths[batch[0]]) > self._threshold)
		]
		if len(singletons) < 2:
			return group
		selected = set(singletons)
		retained = tuple(
			batch for batch in group if len(batch) != 1 or batch[0] not in selected
		)
		return retained + self._packer.pack(singletons)

	def _epoch_batches(self) -> tuple[tuple[int, ...], ...]:
		if self._batches is None:
			generator = torch.Generator().manual_seed(self._seed + self._epoch)
			groups: list[tuple[tuple[int, ...], ...]] = []
			for group in self._groups(generator):
				group = self._repack_checkpointing(group)
				if self._plan is not None:
					group = tuple(
						group[index]
						for index in torch.randperm(
							len(group), generator=generator
						).tolist()
					)
				groups.append(group)
			self._batches = tuple(batch for group in groups for batch in group)
		return self._batches

	def __iter__(self) -> Iterator[list[int]]:
		for batch in self._epoch_batches():
			yield list(batch)

	def __len__(self) -> int:
		return len(self._epoch_batches())


@dataclass(frozen=True, slots=True)
class PreparedTrainingSplit:
	"""One selected training reference, shared tokens, and a ready-to-use loader."""

	reference: pd.DataFrame
	tokens: PairwiseTokenStore
	loader: DataLoader[PairwiseBatch]
	plan: TrainingBatchPlan | None
	_sampler: _TrainingBatchSampler | None = field(repr=False)
	warmup_context: WarmupCacheContext | None = None

	def set_epoch(self, epoch: int) -> None:
		"""Set the joint epoch; unmodified fixed batching keeps its original RNG stream."""
		if self._sampler is not None:
			self._sampler.set_epoch(epoch)

	@property
	def optimizer_group_rows(self) -> tuple[int, ...] | None:
		"""Report effective update sizes, with any merged group last."""
		return None if self._sampler is None else self._sampler.group_row_counts


class _BatchCache:
	"""Read optional cache entries and atomically publish complete CPU payloads."""

	def __init__(self, root: Path, model_label: str = "LoRA") -> None:
		self.root = root
		self._model_label = model_label

	@staticmethod
	def key(metadata: dict[str, Any]) -> str:
		return hashlib.sha256(
			json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
		).hexdigest()

	def path(self, kind: str, metadata: dict[str, Any]) -> Path:
		return self.root / kind / f"{self.key(metadata)}.pt"

	def load(self, kind: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
		path = self.path(kind, metadata)
		storage = CacheStorage(path)
		if not storage.exists():
			return None
		try:
			with storage.open("r") as session:
				payload = torch.load(
					session.path, map_location="cpu", weights_only=True
				)
			if isinstance(payload, dict) and payload.get("metadata") == metadata:
				return payload
		except (
			OSError,
			ValueError,
			TypeError,
			RuntimeError,
			EOFError,
			pickle.UnpicklingError,
			KeyError,
			IndexError,
			OverflowError,
		):
			# Optional cache files may be absent, stale, truncated, or invalid archives.
			return None
		return None

	def save(self, kind: str, metadata: dict[str, Any], values: dict[str, Any]) -> None:
		path = self.path(kind, metadata)
		try:
			with CacheStorage(path).open("w") as session:
				torch.save({"metadata": metadata, **values}, session.path)
		except Exception as error:
			raise ModelExecutionError(
				f"Could not write {self._model_label} batch cache {path}: {error}"
			) from error


class LoRATrainingBatchRepository:
	"""Prepare holdout/build loaders without model weights or CUDA setup."""

	PLAN_SCHEMA_VERSION = 4

	def __init__(
		self,
		*,
		config: WarmupModelConfig,
		data: ModelData,
		serializer_version: str,
		base_revision: str | None,
		cache: _BatchCache,
		token_cache: TokenizationCache,
		refresh_cache: bool,
		model_key: str = "qwen3_4b_lora_EXPT2",
		model_label: str = "LoRA",
		backbone_hidden_size: int | None = None,
		local_weights_sha256: str | None = None,
		warmup_only: bool = False,
	) -> None:
		if not warmup_only and not isinstance(config, LoRAModelConfig):
			raise TypeError("Joint training batches require a LoRAModelConfig.")
		self._config = config
		self._warmup_only = warmup_only
		self._model_label = model_label
		self.data = data
		self.serializer_version = serializer_version
		self.base_revision = base_revision
		self._backbone_hidden_size = backbone_hidden_size
		self._local_weights_sha256 = local_weights_sha256
		self._warmup_ids: pd.DataFrame | None = None
		if config.warmup_epochs and backbone_hidden_size is not None:
			self._warmup_ids = data.reference.loc[
				~data.reference["is_swapped"], ["id", "split"]
			].sort_values("id")
		self._cache = cache
		self.token_cache = token_cache
		self._refresh_cache = refresh_cache
		self._plans: dict[str, dict[str, Any]] = {}
		self.tokens = PairwiseTokenStore(
			data.canonical_texts,
			token_cache,
			revision=base_revision,
			max_length=config.max_length,
			refresh_cache=refresh_cache,
			max_length_field=f"{model_key}.max_length",
		)

	@classmethod
	def from_config(
		cls,
		config: AppConfig,
		refresh_cache: bool = False,
		*,
		profile: ModelProfile | None = None,
		warmup_only: bool = False,
	) -> LoRATrainingBatchRepository:
		"""Load data and canonical tokens once, preparing split plans on demand."""
		profile = profile or PROFILE
		model_key = "warmup" if warmup_only else profile.config_key
		model_label = (
			"Warmup"
			if warmup_only
			else ("QLoRA" if profile.quantize_training else "LoRA")
		)
		model = (
			profile.resolve_config(config)
			if not warmup_only
			else getattr(config, model_key, None)
		)
		if model is None:
			raise ModelExecutionError(
				f"{model_label} batch preparation requires {model_key} configuration."
			)
		expected_type = WarmupModelConfig if warmup_only else LoRAModelConfig
		if not isinstance(model, expected_type):
			raise TypeError(f"{model_key} must be a {expected_type.__name__}.")
		if profile.ab_swap_override is not None:
			model = replace(model, ab_swap=profile.ab_swap_override)
		try:
			repository = ModelDataRepository(
				config.data.processed_path,
				validation_fraction=config.validation.validation_fraction,
			)
			data = repository.load()
			if warmup_only:
				reference = data.reference.loc[
					data.reference["split"].eq("train")
				].copy()
				training_ids = set(reference["id"])
				data = ModelData(
					reference=reference.reset_index(drop=True),
					canonical_texts=tuple(
						record
						for record in data.canonical_texts
						if record.row_id in training_ids
					),
					fingerprint=data.fingerprint,
				)
			base_config = load_pretrained(AutoConfig.from_pretrained, model.model_name)
			revision = getattr(base_config, "_commit_hash", None)
			if revision is None and not Path(model.model_name).is_dir():
				raise ModelExecutionError(
					f"Could not resolve a fixed Hub revision for {model_label} token caching."
				)
			return cls(
				config=model,
				data=data,
				serializer_version=repository.serializer_version,
				base_revision=revision,
				backbone_hidden_size=int(base_config.hidden_size),
				local_weights_sha256=(
					LocalBackboneFingerprint.compute(Path(model.model_name))
					if model.warmup_epochs and Path(model.model_name).is_dir()
					else None
				),
				cache=_BatchCache(
					config.cache_dir / "adapter_batches",
					model_label,
				),
				token_cache=TokenizationCache.from_pretrained(
					config.cache_dir / "token_cache",
					model.model_name,
					revision=revision,
				),
				refresh_cache=refresh_cache,
				model_key=model_key,
				model_label=model_label,
				warmup_only=warmup_only,
			)
		except ModelExecutionError:
			raise
		except Exception as error:
			raise ModelExecutionError(
				f"Could not prepare {model_label} training batches: {error}"
			) from error

	def _load_plan(
		self, reference: pd.DataFrame, dataset: PairwiseTokenDataset, *, full_data: bool
	) -> TrainingBatchPlan:
		config = self._config
		group_size = config.training_batch_size * config.gradient_accumulation_steps
		identities = [
			(int(row.id), bool(row.is_swapped))
			for row in reference.itertuples(index=False)
		]
		positions = {identity: index for index, identity in enumerate(identities)}
		row_data = np.ascontiguousarray(
			np.array(
				sorted(
					(*identity, *dataset.branch_lengths(index))
					for index, identity in enumerate(identities)
				),
				dtype="<i8",
			)
		)
		metadata = {
			"schema_version": self.PLAN_SCHEMA_VERSION,
			"algorithm_version": _GlobalBatchPlanner.VERSION,
			"rows_sha256": hashlib.sha256(memoryview(row_data)).hexdigest(),
			"threshold": config.gradient_checkpointing_threshold,
			"group_size": group_size,
		}
		key = self._cache.key(metadata)
		payload = self._plans.get(key)
		if payload is None and not self._refresh_cache:
			payload = self._cache.load("plans", metadata)
		plan: TrainingBatchPlan | None = None
		if payload is not None:
			try:
				plan = TrainingBatchPlan(
					tuple(
						tuple(
							tuple(positions[tuple(identity)] for identity in batch)
							for batch in group
						)
						for group in payload["groups"]
					),
					payload["padded_tokens"],
				)
				plan.validate(
					dataset, group_size, config.gradient_checkpointing_threshold
				)
			except KeyError, TypeError, ValueError, OverflowError:
				plan = None
		if plan is None:
			plan = _GlobalBatchPlanner(
				dataset, reference, group_size, config.gradient_checkpointing_threshold
			).build()
			plan.validate(dataset, group_size, config.gradient_checkpointing_threshold)
			payload = {
				"groups": tuple(
					tuple(
						tuple(identities[index] for index in batch) for batch in group
					)
					for group in plan.groups
				),
				"padded_tokens": plan.padded_tokens,
			}
			self._cache.save("plans", metadata, payload)
			status = "Created"
		else:
			status = "Reused"
		self._plans[key] = payload
		updates = OptimizerGroupLayout(len(reference), group_size).update_count
		print(
			f"{status} {self._model_label} batch plan ({'build' if full_data else 'holdout'}): "
			f"{len(reference):,} rows, {updates:,} updates, {plan.batch_count:,} base micro-batches; "
			f"{self._cache.path('plans', metadata)}",
			flush=True,
		)
		return plan

	def warmup_context(self, *, full_data: bool = False) -> WarmupCacheContext | None:
		"""Identify actual training IDs, independently of validation membership."""
		if (
			self._warmup_ids is None
			or self._backbone_hidden_size is None
			or (self.base_revision is None and self._local_weights_sha256 is None)
		):
			return None
		originals = self._warmup_ids
		if full_data:
			training_ids = originals["id"]
		else:
			training_ids = originals.loc[originals["split"].eq("train"), "id"]
		return WarmupCacheContext(
			training_ids_sha256=self._ids_key(training_ids),
			serializer_version=self.serializer_version,
			base_revision=self.base_revision,
			local_weights_sha256=self._local_weights_sha256,
			backbone_hidden_size=self._backbone_hidden_size,
			tokenizer_key=self.token_cache.tokenizer_key,
		)

	@staticmethod
	def _ids_key(ids: pd.Series) -> str:
		"""Hash an already sorted original-ID set in a stable binary representation."""
		values = np.ascontiguousarray(ids.to_numpy(dtype="<i8"))
		return hashlib.sha256(memoryview(values)).hexdigest()

	def get_training(self, *, full_data: bool = False) -> PreparedTrainingSplit:
		"""Select holdout-training or full-data orientations and their cached batches."""
		if self._warmup_only and full_data:
			raise ModelExecutionError(
				"Standalone warmup only supports the train partition."
			)
		reference = self.data.reference
		if not full_data:
			reference = reference.loc[reference["split"].eq("train")]
		if self._config.ab_swap in {"inf", "none"}:
			reference = reference.loc[~reference["is_swapped"]]
		reference = reference.reset_index(drop=True).copy()
		if reference.empty or reference.duplicated(["id", "is_swapped"]).any():
			raise ModelExecutionError(
				f"{self._model_label} training must contain unique, nonempty oriented rows."
			)
		dataset = PairwiseTokenDataset(reference, self.tokens)
		seed = self._config.random_state
		options = {
			"collate_fn": PairwiseCollator(self.tokens.pad_token_id),
			"num_workers": self._config.dataloader_workers,
			"pin_memory": True,
			"persistent_workers": self._config.dataloader_workers > 0,
			"generator": torch.Generator().manual_seed(seed),
		}
		plan = None
		sampler = None
		if not self._warmup_only:
			if self._config.gradient_checkpointing_threshold > 0:
				plan = self._load_plan(reference, dataset, full_data=full_data)
			if plan is not None or self._config.gradient_checkpointing_budget > 0:
				sampler = _TrainingBatchSampler(plan, dataset, reference, self._config)
		if sampler is not None:
			loader = DataLoader(dataset, batch_sampler=sampler, **options)
		elif self._warmup_only:
			loader = DataLoader(
				dataset,
				batch_size=self._config.warmup_training_batch_size,
				shuffle=True,
				**options,
			)
		else:
			loader = DataLoader(
				dataset,
				batch_size=self._config.training_batch_size,
				shuffle=True,
				**options,
			)
		return PreparedTrainingSplit(
			reference,
			self.tokens,
			loader,
			plan,
			sampler,
			self.warmup_context(full_data=full_data),
		)
