"""Cache text encodings independently of datasets, model weights, and batching."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ...errors import ModelExecutionError
from .data import PairwiseText


@dataclass(frozen=True, slots=True)
class TokenizedTexts:
	"""Unpadded int32 encodings in the requested text order, without dataset IDs."""

	tokens: np.ndarray
	offsets: np.ndarray
	content_key: str

	def __len__(self) -> int:
		return len(self.offsets) - 1

	def __getitem__(self, index: int) -> np.ndarray:
		if not 0 <= index < len(self):
			raise IndexError(index)
		return self.tokens[self.offsets[index] : self.offsets[index + 1]]


class TokenizationCache:
	"""Persist content-addressed text encodings in SQLite shards per encoding identity."""

	SCHEMA_VERSION = 1
	CHUNK_SIZE = 256

	def __init__(self, cache_dir: Path, tokenizer: PreTrainedTokenizerBase) -> None:
		self.root = cache_dir.resolve()
		self.tokenizer = tokenizer
		self.tokenizer_key = self._fingerprint(tokenizer)
		self._refreshed: set[tuple[str, str]] = set()

	@classmethod
	def from_pretrained(
		cls, cache_dir: Path, model_name: str, *, revision: str | None = None
	) -> TokenizationCache:
		"""Load only the tokenizer; model name and revision are not cache identities."""
		try:
			return cls(
				cache_dir, AutoTokenizer.from_pretrained(model_name, revision=revision)
			)
		except ModelExecutionError:
			raise
		except Exception as error:
			raise ModelExecutionError(
				f"Could not prepare tokenizer cache for {model_name}: {error}"
			) from error

	@staticmethod
	def _key(value: Any) -> str:
		return hashlib.sha256(
			json.dumps(
				value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
			).encode("utf-8")
		).hexdigest()

	@classmethod
	def _normalize(cls, value: Any) -> Any:
		"""Represent slow-tokenizer settings and AddedToken flags deterministically."""
		if value is None or isinstance(value, (str, int, float, bool)):
			return value
		if isinstance(value, Path):
			return cls._file_key(value)
		if isinstance(value, bytes):
			return {"sha256": hashlib.sha256(value).hexdigest()}
		if isinstance(value, dict):
			return {str(key): cls._normalize(item) for key, item in value.items()}
		if isinstance(value, (list, tuple)):
			return [cls._normalize(item) for item in value]
		if hasattr(value, "content"):
			return {
				name: getattr(value, name)
				for name in (
					"content",
					"single_word",
					"lstrip",
					"rstrip",
					"normalized",
					"special",
				)
			}
		raise ModelExecutionError(
			f"Cannot fingerprint tokenizer setting of type {type(value).__name__}."
		)

	@staticmethod
	def _file_key(path: Path) -> dict[str, str]:
		digest = hashlib.sha256()
		with path.open("rb") as stream:
			for chunk in iter(lambda: stream.read(1024 * 1024), b""):
				digest.update(chunk)
		return {"file_sha256": digest.hexdigest()}

	@classmethod
	def _fingerprint(cls, tokenizer: PreTrainedTokenizerBase) -> str:
		identity: dict[str, Any] = {
			"class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
			"transformers_version": transformers.__version__,
			"tokenizers_version": version("tokenizers"),
		}
		backend = getattr(tokenizer, "backend_tokenizer", None)
		if backend is not None:
			state = json.loads(backend.to_str())
			# Padding, truncation, and decoding do not define full text encodings.
			for name in ("padding", "truncation", "decoder"):
				state.pop(name, None)
			identity["backend"] = state
		else:
			ignored = {
				"name_or_path",
				"_commit_hash",
				"revision",
				"cache_dir",
				"token",
				"local_files_only",
				"force_download",
				"is_local",
				"model_max_length",
				"padding_side",
				"truncation_side",
				"chat_template",
				"clean_up_tokenization_spaces",
			}
			settings = {}
			for name, value in tokenizer.init_kwargs.items():
				if name in ignored:
					continue
				if (
					name.endswith("_file")
					and isinstance(value, str)
					and Path(value).is_file()
				):
					settings[name] = cls._file_key(Path(value))
				else:
					settings[name] = cls._normalize(value)
			identity.update(
				vocabulary=tokenizer.get_vocab(),
				settings=settings,
				added_tokens=cls._normalize(tokenizer.added_tokens_decoder),
			)
		return cls._key(identity)

	@property
	def pad_token_id(self) -> int:
		value = self.tokenizer.pad_token_id
		if value is None:
			value = self.tokenizer.eos_token_id
		if value is None:
			raise ModelExecutionError(
				"The configured tokenizer has neither a pad token nor an EOS token."
			)
		return int(value)

	def encode(
		self,
		texts: Sequence[str],
		*,
		add_special_tokens: bool = True,
		truncation: bool = False,
		max_length: int | None = None,
		refresh_cache: bool = False,
	) -> TokenizedTexts:
		"""Reuse complete encodings; cache exact tokenizer truncations only for overflow."""
		if not isinstance(add_special_tokens, bool) or not isinstance(truncation, bool):
			raise TypeError("Tokenization flags must be boolean.")
		if truncation and (
			isinstance(max_length, bool)
			or not isinstance(max_length, int)
			or max_length <= 0
		):
			raise ValueError("Truncation requires an explicit positive max_length.")
		if isinstance(texts, (str, bytes)) or any(
			not isinstance(text, str) for text in texts
		):
			raise TypeError("Tokenization cache inputs must be a sequence of strings.")
		# A caller may update the supplied tokenizer's vocabulary or processors.
		self.tokenizer_key = self._fingerprint(self.tokenizer)
		text_keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
		unique = dict(zip(text_keys, texts, strict=True))
		options = {
			"add_special_tokens": add_special_tokens,
			"split_special_tokens": bool(
				getattr(self.tokenizer, "split_special_tokens", False)
			),
			"truncation": False,
		}
		values, hits, misses = self._encodings(unique, options, refresh_cache)
		if truncation:
			options = {
				**options,
				"truncation": True,
				"max_length": max_length,
				"truncation_side": self.tokenizer.truncation_side,
			}
			overflow = {
				key: unique[key]
				for key, value in values.items()
				if len(value) // 4 > max_length
			}
			if overflow:
				bounded, reused, encoded = self._encodings(
					overflow, options, refresh_cache
				)
				values.update(bounded)
				hits += reused
				misses += encoded
		buffer = bytearray()
		offsets = np.zeros(len(texts) + 1, dtype=np.int64)
		for index, key in enumerate(text_keys):
			buffer.extend(values[key])
			offsets[index + 1] = len(buffer) // 4
		content_key = hashlib.sha256(self._namespace(options).encode("ascii"))
		for key in text_keys:
			content_key.update(key.encode("ascii"))
		print(
			f"Token cache: {hits:,} reused, {misses:,} encoded entries; {self.root}",
			flush=True,
		)
		return TokenizedTexts(
			np.frombuffer(buffer, dtype=np.int32), offsets, content_key.hexdigest()
		)

	def _namespace(self, options: dict[str, Any]) -> str:
		return self._key(
			{
				"schema_version": self.SCHEMA_VERSION,
				"tokenizer": self.tokenizer_key,
				**options,
			}
		)

	def _encodings(
		self, texts: dict[str, str], options: dict[str, Any], refresh: bool
	) -> tuple[dict[str, bytes], int, int]:
		if not texts:
			return {}, 0, 0
		namespace = self._namespace(options)
		path = self.root / f"{namespace}.sqlite3"
		try:
			self.root.mkdir(parents=True, exist_ok=True)
			for attempt in range(2):
				try:
					return self._read_or_encode(
						path, namespace, texts, options, refresh
					)
				except sqlite3.DatabaseError as error:
					code = getattr(error, "sqlite_errorcode", 0) & 0xFF
					if attempt or code not in (
						sqlite3.SQLITE_CORRUPT,
						sqlite3.SQLITE_NOTADB,
					):
						raise
					archive = path.with_name(
						f"{path.stem}.corrupt-{uuid4().hex}.sqlite3"
					)
					path.replace(archive)
					print(f"Archived corrupt token cache: {archive}", flush=True)
		except (sqlite3.Error, OSError) as error:
			raise ModelExecutionError(
				f"Could not read/write token cache {path}: {error}"
			) from error
		raise RuntimeError("Token cache recovery did not finish.")

	def _read_or_encode(
		self,
		path: Path,
		namespace: str,
		texts: dict[str, str],
		options: dict[str, Any],
		refresh: bool,
	) -> tuple[dict[str, bytes], int, int]:
		connection = sqlite3.connect(path, timeout=30)
		values: dict[str, bytes] = {}
		hits = misses = 0
		try:
			with connection:
				connection.execute(
					"CREATE TABLE IF NOT EXISTS encodings ("
					"text_sha256 TEXT PRIMARY KEY, ids BLOB NOT NULL, "
					"token_count INTEGER NOT NULL, checksum TEXT NOT NULL) WITHOUT ROWID"
				)
			keys = list(texts)
			for start in range(0, len(keys), self.CHUNK_SIZE):
				chunk = keys[start : start + self.CHUNK_SIZE]
				placeholders = ",".join("?" for _ in chunk)
				for key, blob, count, checksum in connection.execute(
					f"SELECT text_sha256, ids, token_count, checksum FROM encodings WHERE text_sha256 IN ({placeholders})",
					chunk,
				):
					if refresh and (namespace, key) not in self._refreshed:
						continue
					if (
						isinstance(blob, bytes)
						and isinstance(count, int)
						and count >= 0
						and len(blob) == count * 4
						and hashlib.sha256(blob).hexdigest() == checksum
					):
						values[key] = blob
				missing = [key for key in chunk if key not in values]
				hits += len(chunk) - len(missing)
				if not missing:
					continue
				kwargs = {
					name: value
					for name, value in options.items()
					if name != "truncation_side"
				}
				encoded = self.tokenizer(
					[texts[key] for key in missing],
					padding=False,
					return_attention_mask=False,
					return_token_type_ids=False,
					verbose=False,
					**kwargs,
				)["input_ids"]
				rows = []
				for key, ids in zip(missing, encoded, strict=True):
					array = np.asarray(ids, dtype=np.int64)
					if (
						array.ndim != 1
						or np.any(array < 0)
						or np.any(array > np.iinfo(np.int32).max)
					):
						raise ModelExecutionError(
							"Tokenizer IDs must fit nonnegative int32 values."
						)
					blob = array.astype(np.int32).tobytes()
					values[key] = blob
					rows.append(
						(key, blob, len(array), hashlib.sha256(blob).hexdigest())
					)
				with connection:
					connection.executemany(
						"INSERT OR REPLACE INTO encodings VALUES (?, ?, ?, ?)", rows
					)
				if refresh:
					self._refreshed.update((namespace, key) for key in missing)
				misses += len(missing)
		finally:
			connection.close()
		return values, hits, misses


class PairwiseTokenStore:
	"""Bind shared text encodings to canonical IDs, independent of label and fold."""

	def __init__(
		self,
		records: tuple[PairwiseText, ...],
		cache: TokenizationCache,
		*,
		revision: str | None,
		max_length: int | None = None,
		refresh_cache: bool = False,
	) -> None:
		self._records = records
		self.revision = revision
		self.positions = {record.row_id: index for index, record in enumerate(records)}
		if len(self.positions) != len(records):
			raise ModelExecutionError("Canonical conversations must have unique IDs.")
		encoded = cache.encode(
			[
				text
				for record in records
				for text in (record.response_a, record.response_b)
			],
			refresh_cache=refresh_cache,
		)
		self.tokens = encoded.tokens
		self.offsets = encoded.offsets
		self.content_key = encoded.content_key
		self.lengths = np.diff(self.offsets).reshape(-1, 2)
		self.pad_token_id = cache.pad_token_id
		self.validate_max_length(max_length)

	def matches(self, records: tuple[PairwiseText, ...], revision: str | None) -> bool:
		return self._records is records and self.revision == revision

	def validate_max_length(self, max_length: int | None) -> None:
		"""Keep LoRA's context checks separate from the complete reusable encodings."""
		invalid = self.lengths <= 0
		if max_length is not None:
			invalid |= self.lengths > max_length
		positions = np.argwhere(invalid)
		if not len(positions):
			return
		position, branch = (int(value) for value in positions[0])
		length = int(self.lengths[position, branch])
		row_id = self._records[position].row_id
		if length == 0:
			raise ModelExecutionError(
				f"Row {row_id} branch {'AB'[branch]} contains no tokens."
			)
		raise ModelExecutionError(
			f"Row {row_id} branch {'AB'[branch]} has {length:,} tokens, exceeding "
			f"qwen3_4b_lora.max_length={max_length:,}. Increase max_length to "
			f"at least {length:,}; LoRA inputs are never truncated."
		)
