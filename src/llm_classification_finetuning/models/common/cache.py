"""Persist frozen branch embeddings by token content and backbone computation identity."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch import Tensor

from ...errors import ModelExecutionError
from .cache_storage import CacheCorruptionError, CacheStorage


@dataclass(frozen=True, slots=True)
class EmbeddingCacheKey:
	"""Backbone and computation identity, independent of dataset and input caps."""

	schema_version: int
	model_name: str
	base_revision: str | None
	local_weights_sha256: str | None
	hidden_size: int
	precision: str
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
	_STORAGE_FORMATS: ClassVar[dict[str, tuple[torch.dtype, np.dtype]]] = {
		"float16": (torch.float16, np.dtype("<f2")),
		"bfloat16": (torch.bfloat16, np.dtype("<u2")),
		"float32": (torch.float32, np.dtype("<f4")),
	}

	def __init__(self, root: Path) -> None:
		self.root = root.resolve()

	def path(self, key: EmbeddingCacheKey) -> Path:
		identity = json.dumps(asdict(key), sort_keys=True, separators=(",", ":"))
		digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
		return self.root / f"{digest}.sqlite3"

	@classmethod
	def _storage_format(cls, key: EmbeddingCacheKey) -> tuple[torch.dtype, np.dtype]:
		try:
			return cls._STORAGE_FORMATS[key.storage_dtype]
		except KeyError as error:
			raise ModelExecutionError(
				f"Unsupported embedding cache storage dtype: {key.storage_dtype}"
			) from error

	@staticmethod
	def _corrupt(error: sqlite3.DatabaseError | CacheCorruptionError) -> bool:
		return isinstance(error, CacheCorruptionError) or getattr(
			error, "sqlite_errorcode", 0
		) & 0xFF in (
			sqlite3.SQLITE_CORRUPT,
			sqlite3.SQLITE_NOTADB,
		)

	def load(
		self,
		key: EmbeddingCacheKey,
		input_keys: Sequence[str],
		*,
		refresh_cache: bool = False,
	) -> EmbeddingCacheRows:
		"""Rebind content-addressed vectors to current row positions, leaving misses empty."""
		dtype, numpy_dtype = self._storage_format(key)
		positions: dict[str, list[int]] = {}
		for index, input_key in enumerate(input_keys):
			positions.setdefault(input_key, []).append(index)
		missing = {name: tuple(indices) for name, indices in positions.items()}
		features = torch.empty(
			(len(input_keys), key.hidden_size), dtype=dtype, device="cpu"
		)
		path = self.path(key)
		storage = CacheStorage(path)
		if refresh_cache or not storage.exists():
			return EmbeddingCacheRows(features, missing, 0)
		try:
			with storage.open("r") as session:
				try:
					with closing(
						sqlite3.connect(session.path, timeout=30)
					) as connection:
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
									or len(blob)
									!= numpy_dtype.itemsize * key.hidden_size
									or hashlib.sha256(blob).hexdigest() != checksum
								):
									continue
								values = np.frombuffer(blob, dtype=numpy_dtype)
								vector = torch.from_numpy(
									values.astype(
										numpy_dtype.newbyteorder("="), copy=True
									)
								).view(dtype)
								if not torch.isfinite(vector).all():
									continue
								features[positions[name]] = vector
								del missing[name]
				except sqlite3.DatabaseError as error:
					if self._corrupt(error):
						session.archive()
						missing = {
							name: tuple(indices) for name, indices in positions.items()
						}
					elif "no such table: embeddings" not in str(error):
						raise
		except CacheCorruptionError, FileNotFoundError:
			missing = {name: tuple(indices) for name, indices in positions.items()}
		except (sqlite3.Error, OSError) as error:
			raise ModelExecutionError(
				f"Could not read embedding cache {path}: {error}"
			) from error
		return EmbeddingCacheRows(features, missing, len(positions) - len(missing))

	def save(self, key: EmbeddingCacheKey, rows: EmbeddingCacheRows) -> None:
		"""Transactionally replace only newly computed or explicitly refreshed vectors."""
		if not rows.missing:
			return
		path = self.path(key)
		storage = CacheStorage(path)
		try:
			for attempt in range(2):
				try:
					with storage.open("rw") as session:
						try:
							self._save_rows(session.path, key, rows)
						except sqlite3.DatabaseError as error:
							if not attempt and self._corrupt(error):
								session.archive()
							raise
						session.mark_dirty()
					return
				except (sqlite3.DatabaseError, CacheCorruptionError) as error:
					if attempt or not self._corrupt(error):
						raise
		except (sqlite3.Error, OSError) as error:
			raise ModelExecutionError(
				f"Could not write embedding cache {path}: {error}"
			) from error

	def _save_rows(
		self, path: Path, key: EmbeddingCacheKey, rows: EmbeddingCacheRows
	) -> None:
		dtype, numpy_dtype = self._storage_format(key)
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
					or vector.dtype != dtype
					or not torch.isfinite(vector).all()
				):
					raise ModelExecutionError("Cannot cache invalid frozen embeddings.")
				values = (
					vector.contiguous().view(torch.uint16).numpy()
					if dtype == torch.bfloat16
					else vector.numpy()
				)
				blob = values.astype(numpy_dtype, copy=False).tobytes()
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
