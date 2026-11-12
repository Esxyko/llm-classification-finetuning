"""Atomically publish cache and processed-data files within the per-file size limit."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal
from uuid import uuid4


class CacheCorruptionError(OSError):
	"""A multipart cache cannot be reconstructed and must be rebuilt."""


@dataclass(slots=True)
class CacheFileSession:
	"""A private working file, published only after a successful writable session."""

	path: Path
	_storage: CacheStorage
	_mode: Literal["r", "rw", "w"]
	_dirty: bool = False
	_archived: bool = False

	def mark_dirty(self) -> None:
		"""Publish an updated file when the caller has closed its file handles."""
		if self._mode == "r" or self._archived:
			raise ValueError("Cannot publish a read-only or archived cache session.")
		self._dirty = True

	def archive(self) -> Path | None:
		"""Quarantine the original cache while its session lock is still held."""
		archive = self._storage._archive_locked()
		self._dirty = False
		self._archived = True
		return archive


class CacheStorage:
	"""Present single files and checked multipart files through one disk-backed API."""

	SPLIT_THRESHOLD = 100 * 1024 * 1024
	PART_SIZE = 99 * 1024 * 1024
	COPY_SIZE = 1024 * 1024
	FORMAT_VERSION = 1
	LOCK_TIMEOUT = 30.0
	_DIGEST = re.compile(r"[0-9a-f]{64}")
	_PART_NAME = re.compile(r"[0-9]{6,}-[0-9a-f]{64}\.part")

	def __init__(self, path: Path) -> None:
		self.path = path.resolve()
		self.manifest_path = self.path.with_name(f"{self.path.name}.manifest.json")
		self.parts_path = self.path.with_name(f"{self.path.name}.parts")
		self._work_root = self.path.parent / ".cache-work"
		self._lock_key = hashlib.sha256(
			os.fsencode(os.path.normcase(self.path))
		).hexdigest()

	def exists(self) -> bool:
		"""Report logical presence; actual integrity is checked under the read lock."""
		return self.manifest_path.is_file() or self.path.is_file()

	def missing_components(self) -> tuple[Path, ...]:
		"""List absent or wrong-sized files without reconstructing or archiving data."""
		with self._lock():
			return tuple(
				path
				for path, size in self._components(self._component_manifest())
				if not self._matches_component(path, size)
			)

	def copy_missing_from(
		self,
		source: CacheStorage,
		*,
		on_copy: Callable[[Path, Path], None] | None = None,
	) -> tuple[Path, ...]:
		"""Fill incomplete components, publishing a new manifest after its volumes.

		A local manifest remains authoritative. Source components already available
		locally are not required. Checksums remain the responsibility of ``open``.
		The optional callback receives source and destination before each copy.
		"""
		with self._lock():
			manifest = self._component_manifest()
			if manifest is None and self.path.is_file():
				return ()
			publish_manifest = False
			if manifest is None:
				manifest = source._component_manifest()
				publish_manifest = manifest is not None
			copies = [
				(
					target,
					source.parts_path / target.name if manifest else source.path,
					size,
				)
				for target, size in self._components(manifest)
				if not self._matches_component(target, size)
			]
			if not copies and not publish_manifest:
				return ()
			unavailable = [
				f"{target} <- {origin}"
				+ (f" (expected {size} bytes)" if size is not None else "")
				for target, origin, size in copies
				if not self._matches_component(origin, size)
			]
			if unavailable:
				raise FileNotFoundError(
					"Missing or wrong-sized artifact components (local <- source):\n"
					+ "\n".join(unavailable)
					+ f"\nSource artifact: {source.path} or {source.manifest_path}"
				)
			copied: list[Path] = []
			with tempfile.TemporaryDirectory(dir=self._work_root) as temporary:
				staged: list[tuple[Path, Path]] = []
				for index, (target, origin, size) in enumerate(copies):
					if on_copy is not None:
						on_copy(origin, target)
					working = Path(temporary) / str(index)
					try:
						expected_size = origin.stat().st_size if size is None else size
						shutil.copyfile(origin, working)
						if working.stat().st_size != expected_size:
							raise OSError(f"Expected {expected_size} copied bytes.")
						staged.append((working, target))
					except OSError as error:
						raise OSError(
							f"Could not copy {origin} to {target}: {error}"
						) from error
				if publish_manifest:
					if on_copy is not None:
						on_copy(source.manifest_path, self.manifest_path)
					working_manifest = Path(temporary) / "manifest.json"
					working_manifest.write_text(
						json.dumps(manifest, sort_keys=True, indent=2) + "\n",
						encoding="utf-8",
					)
					if working_manifest.stat().st_size >= self.SPLIT_THRESHOLD:
						raise OSError("Cache manifest exceeds the file size limit.")
					staged.append((working_manifest, self.manifest_path))
				for working, target in staged:
					target.parent.mkdir(parents=True, exist_ok=True)
					self._replace_file(working, target)
					copied.append(target)
			return tuple(copied)

	def _component_manifest(self) -> dict | None:
		if not self.manifest_path.is_file():
			return None
		try:
			return self._read_manifest()
		except (ValueError, KeyError, TypeError, RecursionError) as error:
			raise CacheCorruptionError(
				f"Invalid multipart manifest {self.manifest_path}: {error}"
			) from error

	def _components(self, manifest: dict | None) -> tuple[tuple[Path, int | None], ...]:
		if manifest is None:
			return ((self.path, None),)
		return tuple(
			(self.parts_path / part["name"], part["size"]) for part in manifest["parts"]
		)

	@staticmethod
	def _matches_component(path: Path, size: int | None) -> bool:
		return path.is_file() and (size is None or path.stat().st_size == size)

	@contextmanager
	def open(self, mode: Literal["r", "rw", "w"] = "r") -> Iterator[CacheFileSession]:
		"""Read, update, or replace a cache; failed sessions never publish staged data."""
		if mode not in ("r", "rw", "w"):
			raise ValueError(f"Invalid cache session mode: {mode}")
		with (
			self._lock(),
			tempfile.TemporaryDirectory(
				prefix=f"{self._lock_key}-", dir=self._work_root
			) as temporary,
		):
			working = Path(temporary) / self.path.name
			if mode != "w" and (mode == "r" or self.exists()):
				try:
					self._materialize(working)
				except CacheCorruptionError:
					self._archive_locked()
					raise
			session = CacheFileSession(working, self, mode, mode == "w")
			yield session
			if session._dirty and not session._archived:
				self._publish(working)

	def archive(self) -> Path | None:
		"""Preserve corrupt files using the same single-file or multipart layout."""
		with self._lock():
			return self._archive_locked()

	@contextmanager
	def _lock(self) -> Iterator[None]:
		self._work_root.mkdir(parents=True, exist_ok=True)
		with (self._work_root / f"{self._lock_key}.lock").open("a+b") as handle:
			if handle.seek(0, os.SEEK_END) == 0:
				handle.write(b"\0")
				handle.flush()
			deadline = time.monotonic() + self.LOCK_TIMEOUT
			while True:
				try:
					self._file_lock(handle, unlock=False)
					break
				except OSError as error:
					if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
						raise
					if time.monotonic() >= deadline:
						raise TimeoutError(
							f"Timed out locking cache {self.path}"
						) from error
					time.sleep(0.05)
			try:
				yield
			finally:
				self._file_lock(handle, unlock=True)

	@staticmethod
	def _file_lock(handle: BinaryIO, *, unlock: bool) -> None:
		if os.name == "nt":
			import msvcrt

			handle.seek(0)
			msvcrt.locking(
				handle.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK, 1
			)
		else:
			import fcntl

			fcntl.flock(
				handle.fileno(),
				fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB,
			)

	def _materialize(self, destination: Path) -> None:
		if not self.manifest_path.is_file():
			shutil.copyfile(self.path, destination)
			return
		try:
			manifest = self._read_manifest()
			digest = hashlib.sha256()
			with destination.open("wb") as output:
				for part in manifest["parts"]:
					part_digest = hashlib.sha256()
					with (self.parts_path / part["name"]).open("rb") as source:
						if os.fstat(source.fileno()).st_size != part["size"]:
							raise ValueError("Incorrect cache volume length.")
						while block := source.read(self.COPY_SIZE):
							output.write(block)
							part_digest.update(block)
							digest.update(block)
					if part_digest.hexdigest() != part["sha256"]:
						raise ValueError("Incorrect cache volume checksum.")
			if digest.hexdigest() != manifest["sha256"]:
				raise ValueError("Incorrect complete cache checksum.")
		except (
			FileNotFoundError,
			IsADirectoryError,
			NotADirectoryError,
			ValueError,
			KeyError,
			TypeError,
			RecursionError,
		) as error:
			raise CacheCorruptionError(
				f"Invalid multipart cache {self.path}: {error}"
			) from error

	def _read_manifest(self) -> dict:
		if self.manifest_path.stat().st_size >= self.SPLIT_THRESHOLD:
			raise ValueError("Cache manifest exceeds the file size limit.")
		manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
		if not isinstance(manifest, dict):
			raise TypeError("Invalid cache manifest.")
		if (
			type(manifest.get("version")) is not int
			or manifest["version"] != self.FORMAT_VERSION
		):
			raise ValueError("Unsupported cache storage version.")
		total = manifest.get("size")
		parts = manifest.get("parts")
		if (
			type(total) is not int
			or total < self.SPLIT_THRESHOLD
			or not isinstance(parts, list)
			or len(parts) != (total + self.PART_SIZE - 1) // self.PART_SIZE
			or not self._valid_digest(manifest.get("sha256"))
		):
			raise ValueError("Invalid multipart cache metadata.")
		for index, part in enumerate(parts):
			if not isinstance(part, dict):
				raise TypeError("Invalid cache volume metadata.")
			size = min(self.PART_SIZE, total - index * self.PART_SIZE)
			checksum = part.get("sha256")
			if (
				type(part.get("size")) is not int
				or part["size"] != size
				or not self._valid_digest(checksum)
				or part.get("name") != f"{index + 1:06d}-{checksum}.part"
			):
				raise ValueError(
					"Invalid cache volume order, name, length, or checksum."
				)
		return manifest

	@classmethod
	def _valid_digest(cls, value: object) -> bool:
		return isinstance(value, str) and cls._DIGEST.fullmatch(value) is not None

	@staticmethod
	def _replace_file(source: Path, destination: Path) -> None:
		"""Publish atomically with destination-directory permissions on Windows."""
		if os.name != "nt":
			os.replace(source, destination)
			return
		# Windows moves retain the private TemporaryDirectory's restrictive ACL.
		# Create the published file beside its destination so it inherits that ACL.
		descriptor, name = tempfile.mkstemp(
			prefix=".cache-publish-", suffix=".tmp", dir=destination.parent
		)
		staged = Path(name)
		try:
			with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
				shutil.copyfileobj(input_file, output, length=CacheStorage.COPY_SIZE)
			os.replace(staged, destination)
		finally:
			staged.unlink(missing_ok=True)

	def _publish(self, source: Path) -> None:
		size = source.stat().st_size
		if size < self.SPLIT_THRESHOLD:
			self._replace_file(source, self.path)
			# Until this unlink succeeds, the old multipart generation is authoritative.
			self.manifest_path.unlink(missing_ok=True)
			self._cleanup_parts(set())
			return
		self.parts_path.mkdir(parents=True, exist_ok=True)
		parts = []
		digest = hashlib.sha256()
		with source.open("rb") as input_file:
			while input_file.tell() < size:
				part_file = source.with_name(f".{source.name}.volume.tmp")
				part_digest = hashlib.sha256()
				part_size = 0
				with part_file.open("wb") as output:
					while part_size < self.PART_SIZE:
						block = input_file.read(
							min(self.COPY_SIZE, self.PART_SIZE - part_size)
						)
						if not block:
							break
						output.write(block)
						part_digest.update(block)
						digest.update(block)
						part_size += len(block)
				if part_size != min(self.PART_SIZE, size - len(parts) * self.PART_SIZE):
					raise OSError(
						"Cache working file changed while publishing volumes."
					)
				checksum = part_digest.hexdigest()
				name = f"{len(parts) + 1:06d}-{checksum}.part"
				self._replace_file(part_file, self.parts_path / name)
				parts.append({"name": name, "size": part_size, "sha256": checksum})
		manifest_file = source.with_name(f".{source.name}.manifest.tmp")
		manifest_file.write_text(
			json.dumps(
				{
					"version": self.FORMAT_VERSION,
					"size": size,
					"sha256": digest.hexdigest(),
					"parts": parts,
				},
				sort_keys=True,
				indent=2,
			)
			+ "\n",
			encoding="utf-8",
		)
		if manifest_file.stat().st_size >= self.SPLIT_THRESHOLD:
			raise OSError("Cache manifest exceeds the file size limit.")
		self._replace_file(manifest_file, self.manifest_path)
		# Publication is complete. Cleanup failures must not invalidate that generation.
		try:
			self.path.unlink(missing_ok=True)
		except OSError:
			pass
		self._cleanup_parts({part["name"] for part in parts})

	def _cleanup_parts(self, keep: set[str]) -> None:
		try:
			if not self.parts_path.is_dir():
				return
			for part in self.parts_path.iterdir():
				if self._PART_NAME.fullmatch(part.name) and part.name not in keep:
					part.unlink(missing_ok=True)
			if not keep:
				self.parts_path.rmdir()
		except OSError:
			# Unreferenced volumes are harmless and can be removed by the next write.
			pass

	def _archive_locked(self) -> Path | None:
		if not self.exists():
			return None
		archive = CacheStorage(
			self.path.with_name(
				f"{self.path.stem}.corrupt-{uuid4().hex}{self.path.suffix}"
			)
		)
		if self.manifest_path.is_file():
			moved_parts = self.parts_path.exists()
			if moved_parts:
				os.replace(self.parts_path, archive.parts_path)
			try:
				os.replace(self.manifest_path, archive.manifest_path)
			except OSError:
				if moved_parts:
					os.replace(archive.parts_path, self.parts_path)
				raise
		if self.path.is_file():
			with tempfile.TemporaryDirectory(dir=self._work_root) as temporary:
				working = Path(temporary) / self.path.name
				shutil.copyfile(self.path, working)
				# Preserve a leftover single file separately from an archived manifest.
				target = (
					archive
					if not archive.exists()
					else CacheStorage(
						archive.path.with_name(f"{archive.path.name}.single")
					)
				)
				target._publish(working)
				self.path.unlink()
		# Corruption can also enlarge a volume or the manifest beyond the limit.
		components = [archive.manifest_path]
		if archive.parts_path.is_dir():
			components.extend(archive.parts_path.iterdir())
		for component in components:
			if (
				component.is_symlink()
				or not component.is_file()
				or component.stat().st_size < self.SPLIT_THRESHOLD
			):
				continue
			with tempfile.TemporaryDirectory(dir=self._work_root) as temporary:
				working = Path(temporary) / component.name
				shutil.copyfile(component, working)
				CacheStorage(component)._publish(working)
		print(f"Archived corrupt cache: {archive.path}", flush=True)
		return archive.path
