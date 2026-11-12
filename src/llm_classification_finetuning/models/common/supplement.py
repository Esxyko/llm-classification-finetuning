"""Validate independent synthetic ties before appending them to build inputs."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ...errors import ModelExecutionError
from .cache_storage import CacheStorage
from .inputs import TrainingInputs, TrainingInputLoader, _sha256


class BuildSupplementLoader:
	"""Keep build-only rows outside the saved holdout and its split validation."""

	COUNT = 3822
	SCHEMA_VERSION = 1
	METADATA_KEY = b"llm_classification_finetuning.synthetic_tie"
	PROVENANCE_COLUMNS = (
		"source_original_id",
		"source_side",
		"source_split",
		"historical_wins",
		"historical_occurrences",
	)

	def __init__(self, path: Path) -> None:
		self.path = path.resolve()

	@staticmethod
	def canonical(value: str) -> str:
		"""Normalize JSON formatting without altering conversation content."""
		return json.dumps(
			json.loads(value),
			ensure_ascii=False,
			sort_keys=True,
			separators=(",", ":"),
			allow_nan=False,
		)

	@staticmethod
	def combined_fingerprint(original: str, supplement: str) -> str:
		payload = json.dumps(
			{
				"version": 1,
				"original_sha256": original,
				"supplement_sha256": supplement,
			},
			sort_keys=True,
			separators=(",", ":"),
		)
		return hashlib.sha256(payload.encode("utf-8")).hexdigest()

	def load(self, original: TrainingInputs) -> TrainingInputs:
		"""Read checked single/multipart data and require matching source provenance."""
		try:
			storage = CacheStorage(self.path)
			if not storage.exists():
				raise ValueError(
					"configured supplement is missing; run scripts/generate_synthetic_tie.py"
				)
			with storage.open("r") as session:
				metadata = pq.read_schema(session.path).metadata or {}
				identity = json.loads(metadata[self.METADATA_KEY])
				data = pd.read_parquet(session.path)
				fingerprint = _sha256(session.path)
			if identity.get("schema_version") != self.SCHEMA_VERSION:
				raise ValueError("unsupported synthetic-tie schema")
			if identity.get("source_sha256") != original.fingerprint:
				raise ValueError(
					"supplement source SHA-256 does not match the processed dataset"
				)
			if identity.get("target_count") != self.COUNT:
				raise ValueError("supplement target count must be 3,822")
			self.validate(data, original.data)
			return TrainingInputs(data, fingerprint)
		except Exception as error:
			raise ModelExecutionError(
				f"Invalid build supplement {self.path}: {error}"
			) from error

	@classmethod
	def validate(cls, data: pd.DataFrame, original: pd.DataFrame) -> None:
		"""Check exact count/score, original orientation, unique content and sources."""
		required = [
			*TrainingInputLoader.REQUIRED_COLUMNS,
			*cls.PROVENANCE_COLUMNS,
			"model_a",
			"model_b",
			"winner_model_a",
			"winner_model_b",
			"winner_tie",
		]
		if len(data) != cls.COUNT or data.loc[:, required].isna().any().any():
			raise ValueError("supplement must contain 3,822 complete rows")
		for name in (
			"id",
			"source_original_id",
			"historical_wins",
			"historical_occurrences",
		):
			if not pd.api.types.is_integer_dtype(data[name]):
				raise ValueError(f"{name} must contain integers")
		if data.id.duplicated().any() or data.id.isin(original.id).any():
			raise ValueError(
				"supplement IDs are duplicated or collide with original IDs"
			)
		if not pd.api.types.is_bool_dtype(data.is_swapped) or data.is_swapped.any():
			raise ValueError("supplement must contain unswapped rows only")
		if not (
			data.split.eq("build").all()
			and data.label.eq(2).all()
			and data.winner_model_a.eq(0).all()
			and data.winner_model_b.eq(0).all()
			and data.winner_tie.eq(1).all()
			and data.model_a.eq(data.model_b).all()
			and data.response_a.eq(data.response_b).all()
			and data.source_side.isin(("a", "b")).all()
		):
			raise ValueError("supplement must contain build-only self-response ties")
		score = Fraction()
		keys: set[tuple[str, str]] = set()
		prompts = {cls.canonical(value) for value in data.prompt}
		tie_prompts = {
			cls.canonical(value)
			for value in original.loc[original.label.eq(2), "prompt"]
		}
		if prompts & tie_prompts:
			raise ValueError("supplement includes a prompt with an original tie")
		sources = original.set_index("id")
		for row in data.itertuples(index=False):
			wins, occurrences = (
				int(row.historical_wins),
				int(row.historical_occurrences),
			)
			if occurrences <= 0 or not 0 <= wins <= occurrences:
				raise ValueError(f"invalid historical score for ID {row.id}")
			score += Fraction(wins, occurrences)
			key = (cls.canonical(row.prompt), cls.canonical(row.response_a))
			if key in keys:
				raise ValueError("duplicate normalized synthetic (prompt, response)")
			keys.add(key)
			source = sources.loc[row.source_original_id]
			if not (
				row.prompt == source.prompt
				and row.source_split == source.split
				and row.group_id == source.group_id
				and row.response_a == source[f"response_{row.source_side}"]
			):
				raise ValueError(f"source provenance mismatch for ID {row.id}")
		if score != Fraction(cls.COUNT, 2):
			raise ValueError(f"historical score must be exactly 1911, found {score}")
		for row in original.itertuples(index=False):
			prompt = cls.canonical(row.prompt)
			if prompt in prompts:
				a, b = cls.canonical(row.response_a), cls.canonical(row.response_b)
				if a == b and (prompt, a) in keys:
					raise ValueError(
						"supplement duplicates an original self-response battle"
					)
