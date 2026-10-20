"""Persist frozen branch embeddings by token content and backbone computation identity."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch import Tensor

from ...errors import ModelExecutionError


@dataclass(frozen=True, slots=True)
class EmbeddingCacheKey:
	"""Backbone and computation identity, independent of dataset and input caps."""

	schema_version: int
	model_name: str
	base_revision: str | None
	local_weights_sha256: str | None
	hidden_size: int
	precision: str
	attention_implementation: str
	allow_tf32: bool
	pooling: str = "last-nonpadding-token-v1"
	storage_dtype: str = "float16"
	extraction_version: int = 2


@dataclass(frozen=True, slots=True)
class CachedEmbeddings:
	"""Canonical IDs and their A/B pooled states on CPU."""

	ids: Tensor
	h_a: Tensor
	h_b: Tensor

	@property
	def hidden_size(self) -> int:
		return int(self.h_a.shape[1])


@dataclass(frozen=True, slots=True)
class EmbeddingCacheRows:
	"""Requested branch rows plus the positions of each missing unique input."""

	features: Tensor
	missing: dict[str, tuple[int, ...]]
	reused_entries: int


class EmbeddingCache:
	"""Share individual branch features across training/test and configuration runs."""

	CHUNK_SIZE = 256

	def __init__(self, root: Path) -> None:
		self.root = root.resolve()

	def path(self, key: EmbeddingCacheKey) -> Path:
		identity = json.dumps(asdict(key), sort_keys=True, separators=(",", ":"))
		digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
		return self.root / f"{digest}.sqlite3"

	@staticmethod
	def _corrupt(error: sqlite3.DatabaseError) -> bool:
		return getattr(error, "sqlite_errorcode", 0) & 0xFF in (
			sqlite3.SQLITE_CORRUPT,
			sqlite3.SQLITE_NOTADB,
		)

	@staticmethod
	def _archive(path: Path) -> None:
		archive = path.with_name(f"{path.stem}.corrupt-{uuid4().hex}.sqlite3")
		try:
			path.replace(archive)
		except OSError as error:
			raise ModelExecutionError(
				f"Could not archive embedding cache {path}: {error}"
			) from error
		print(f"Archived corrupt embedding cache: {archive}", flush=True)

	def load(
		self,
		key: EmbeddingCacheKey,
		input_keys: Sequence[str],
		*,
		refresh_cache: bool = False,
	) -> EmbeddingCacheRows:
		"""Rebind content-addressed vectors to current row positions, leaving misses empty."""
		positions: dict[str, list[int]] = {}
		for index, input_key in enumerate(input_keys):
			positions.setdefault(input_key, []).append(index)
		missing = {name: tuple(indices) for name, indices in positions.items()}
		features = torch.empty(
			(len(input_keys), key.hidden_size), dtype=torch.float16, device="cpu"
		)
		path = self.path(key)
		if refresh_cache or not path.is_file():
			return EmbeddingCacheRows(features, missing, 0)
		connection = None
		try:
			connection = sqlite3.connect(path, timeout=30)
			names = list(positions)
			for start in range(0, len(names), self.CHUNK_SIZE):
				chunk = names[start : start + self.CHUNK_SIZE]
				placeholders = ",".join("?" for _ in chunk)
				for name, blob, checksum in connection.execute(
					f"SELECT input_sha256, features, checksum FROM embeddings WHERE input_sha256 IN ({placeholders})",
					chunk,
				):
					if (
						not isinstance(blob, bytes)
						or len(blob) != 2 * key.hidden_size
						or hashlib.sha256(blob).hexdigest() != checksum
					):
						continue
					values = np.frombuffer(blob, dtype="<f2")
					if not np.isfinite(values).all():
						continue
					vector = torch.from_numpy(values.astype(np.float16, copy=True))
					features[positions[name]] = vector
					del missing[name]
		except sqlite3.DatabaseError as error:
			if connection is not None:
				connection.close()
				connection = None
			if self._corrupt(error):
				self._archive(path)
				missing = {name: tuple(indices) for name, indices in positions.items()}
			elif "no such table: embeddings" not in str(error):
				raise ModelExecutionError(
					f"Could not read embedding cache {path}: {error}"
				) from error
		except OSError as error:
			raise ModelExecutionError(
				f"Could not read embedding cache {path}: {error}"
			) from error
		finally:
			if connection is not None:
				connection.close()
		return EmbeddingCacheRows(features, missing, len(positions) - len(missing))

	def save(self, key: EmbeddingCacheKey, rows: EmbeddingCacheRows) -> None:
		"""Transactionally replace only newly computed or explicitly refreshed vectors."""
		if not rows.missing:
			return
		path = self.path(key)
		try:
			self.root.mkdir(parents=True, exist_ok=True)
			for attempt in range(2):
				try:
					self._save_rows(path, key, rows)
					return
				except sqlite3.DatabaseError as error:
					if attempt or not self._corrupt(error):
						raise
					self._archive(path)
		except (sqlite3.Error, OSError) as error:
			raise ModelExecutionError(
				f"Could not write embedding cache {path}: {error}"
			) from error

	def _save_rows(
		self, path: Path, key: EmbeddingCacheKey, rows: EmbeddingCacheRows
	) -> None:
		connection = sqlite3.connect(path, timeout=30)
		try:
			with connection:
				connection.execute(
					"CREATE TABLE IF NOT EXISTS embeddings ("
					"input_sha256 TEXT PRIMARY KEY, features BLOB NOT NULL, checksum TEXT NOT NULL) WITHOUT ROWID"
				)
			pending = []
			for input_key, positions in rows.missing.items():
				vector = rows.features[positions[0]].detach().cpu()
				if (
					vector.shape != (key.hidden_size,)
					or vector.dtype != torch.float16
					or not torch.isfinite(vector).all()
				):
					raise ModelExecutionError("Cannot cache invalid frozen embeddings.")
				blob = vector.numpy().astype("<f2", copy=False).tobytes()
				pending.append((input_key, blob, hashlib.sha256(blob).hexdigest()))
				if len(pending) == self.CHUNK_SIZE:
					with connection:
						connection.executemany(
							"INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
							pending,
						)
					pending.clear()
			if pending:
				with connection:
					connection.executemany(
						"INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)", pending
					)
		finally:
			connection.close()
