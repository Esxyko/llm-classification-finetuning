"""Store each downloaded Hub model in one ZIP64 archive."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any


class ModelArchive:
	"""Publish verified archives and restore native Hub repository layouts."""

	COPY_SIZE = 8 * 1024 * 1024

	def __init__(self, path: Path) -> None:
		self.path = path.expanduser().resolve()

	@contextmanager
	def lock(self) -> Iterator[None]:
		"""Serialize archive migration, extraction, downloads, and publication."""
		work = self.path.parent / ".cache-work"
		work.mkdir(parents=True, exist_ok=True)
		with (work / f"{self.path.name}.lock").open("a+b") as handle:
			if handle.seek(0, os.SEEK_END) == 0:
				handle.write(b"\0")
				handle.flush()
			while True:
				try:
					if os.name == "nt":
						import msvcrt

						handle.seek(0)
						msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
					else:
						import fcntl

						fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
					break
				except OSError as error:
					if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
						raise
					time.sleep(0.1)
			try:
				yield
			finally:
				if os.name == "nt":
					handle.seek(0)
					msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
				else:
					fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

	@staticmethod
	def entries(repository: Path) -> list[Path]:
		"""Reject external links and omit unfinished Hub downloads."""
		root = repository.resolve(strict=True)
		if not root.is_dir() or repository.is_symlink() or repository.is_junction():
			raise ValueError(f"Expected a regular model cache directory: {repository}")
		entries = []
		for entry in sorted(root.rglob("*")):
			if entry.is_junction():
				raise ValueError(f"Directory junctions are not supported: {entry}")
			if entry.is_symlink():
				target = entry.resolve(strict=True)
				if not target.is_relative_to(root) or not target.is_file():
					raise ValueError(
						f"Expected a file link within its model cache: {entry}"
					)
			elif not entry.is_file() and not entry.is_dir():
				raise ValueError(f"Unsupported model cache entry: {entry}")
			if not entry.name.endswith(".incomplete"):
				entries.append(entry)
		return entries

	@classmethod
	def signature(cls, repository: Path) -> tuple:
		"""Detect downloaded files and updated references without hashing weights."""
		if not repository.exists():
			return ()
		return tuple(
			(
				entry.relative_to(repository).as_posix(),
				entry.stat().st_size,
				entry.stat().st_mtime_ns,
				str(entry.readlink()) if entry.is_symlink() else None,
			)
			for entry in cls.entries(repository)
			if not entry.is_dir()
		)

	def publish(self, repository: Path) -> None:
		"""Write a ZIP64 file, verify every entry, and atomically replace the archive.

		The caller holds ``lock()``. Links are stored once as relative references;
		weight blobs use ZIP_STORED to avoid expensive, ineffective compression.
		"""
		entries = self.entries(repository)
		before = self.signature(repository)
		self.path.parent.mkdir(parents=True, exist_ok=True)
		with tempfile.TemporaryDirectory(
			prefix=".pack-model-", dir=self.path.parent
		) as temporary:
			staged = Path(temporary) / self.path.name
			digests = {}
			with zipfile.ZipFile(
				staged, "w", compression=zipfile.ZIP_STORED
			) as archive:
				for entry in entries:
					name = entry.relative_to(repository).as_posix()
					if entry.is_symlink():
						link = os.path.relpath(entry.resolve(), entry.parent).replace(
							"\\", "/"
						)
						info = zipfile.ZipInfo(name)
						info.create_system = 3
						info.external_attr = (stat.S_IFLNK | 0o777) << 16
						payload = link.encode("utf-8")
						archive.writestr(info, payload)
						digests[name] = hashlib.sha256(payload).digest()
					elif entry.is_dir():
						archive.mkdir(f"{name}/")
					else:
						digest = hashlib.sha256()
						with (
							entry.open("rb") as source,
							archive.open(name, "w", force_zip64=True) as target,
						):
							while chunk := source.read(self.COPY_SIZE):
								target.write(chunk)
								digest.update(chunk)
						digests[name] = digest.digest()
			with zipfile.ZipFile(staged) as archive:
				for name, expected in digests.items():
					with archive.open(name) as stream:
						if hashlib.file_digest(stream, "sha256").digest() != expected:
							raise ValueError(
								f"Model archive verification failed: {name}"
							)
			if self.signature(repository) != before:
				raise ValueError(f"Model cache changed while packing: {repository}")
			os.replace(staged, self.path)

	@staticmethod
	def _member_path(root: Path, name: str) -> Path:
		parts = PurePosixPath(name)
		if (
			not parts.parts
			or parts.is_absolute()
			or ".." in parts.parts
			or "\\" in name
			or ":" in name
		):
			raise ValueError(f"Invalid path in model archive: {name}")
		target = root.joinpath(*parts.parts)
		if not target.resolve().is_relative_to(root):
			raise ValueError(f"Model archive entry escapes its repository: {name}")
		return target

	def restore(self, repository: Path) -> None:
		"""Merge cached immutable files and refresh refs under the archive lock."""
		repository.mkdir(parents=True, exist_ok=True)
		root = repository.resolve()
		with zipfile.ZipFile(self.path) as archive:
			links = []
			seen = set()
			for info in archive.infolist():
				target = self._member_path(root, info.filename)
				if target in seen:
					raise ValueError(f"Duplicate model archive entry: {info.filename}")
				seen.add(target)
				if stat.S_ISLNK(info.external_attr >> 16):
					links.append((info, target))
					continue
				if info.is_dir():
					target.mkdir(parents=True, exist_ok=True)
					continue
				if target.is_file() and not info.filename.startswith("refs/"):
					continue
				target.parent.mkdir(parents=True, exist_ok=True)
				with tempfile.NamedTemporaryFile(
					dir=target.parent, delete=False
				) as destination:
					staged = Path(destination.name)
					try:
						with archive.open(info) as source:
							shutil.copyfileobj(source, destination, self.COPY_SIZE)
					except BaseException:
						destination.close()
						staged.unlink()
						raise
				try:
					os.replace(staged, target)
				finally:
					staged.unlink(missing_ok=True)
			for info, target in links:
				link = archive.read(info).decode("utf-8")
				if "\\" in link or ":" in link or PurePosixPath(link).is_absolute():
					raise ValueError(f"Invalid model archive link: {info.filename}")
				resolved = (target.parent / link).resolve(strict=True)
				if not resolved.is_relative_to(root) or not resolved.is_file():
					raise ValueError(
						f"Model archive link escapes its repository: {info.filename}"
					)
				if target.is_file():
					continue
				target.parent.mkdir(parents=True, exist_ok=True)
				try:
					target.symlink_to(os.path.relpath(resolved, target.parent))
				except OSError as error:
					if getattr(error, "winerror", None) != 1314:
						raise
					shutil.copy2(resolved, target)


class ModelStore:
	"""Keep native Hub paths and revision identities in a private process cache."""

	def __init__(self, root: Path) -> None:
		self.root = root.expanduser().resolve()
		work = self.root / ".cache-work"
		work.mkdir(parents=True, exist_ok=True)
		self._temporary = tempfile.TemporaryDirectory(prefix="hf-", dir=work)
		self._hub = Path(self._temporary.name) / "hub"
		self._restored: dict[str, tuple[int, int]] = {}
		self._lock = threading.RLock()

	def load(self, loader: Callable[..., Any], model_name: str, **kwargs: Any) -> Any:
		"""Restore on demand and publish newly downloaded files after a load."""
		parts = model_name.split("/")
		if len(parts) not in (1, 2) or any(
			not part or part in (".", "..") or "\\" in part or ":" in part
			for part in parts
		):
			raise ValueError(f"Invalid Hub model name: {model_name}")
		name = "models--" + "--".join(parts)
		archive = ModelArchive(self.root / f"{name}.zip")
		repository = self._hub / name
		with self._lock, archive.lock():
			if archive.path.exists():
				metadata = archive.path.stat()
				stamp = (metadata.st_mtime_ns, metadata.st_size)
				if self._restored.get(name) != stamp:
					print(f"Unpacking model cache: {archive.path}", flush=True)
					archive.restore(repository)
					self._restored[name] = stamp
			before = archive.signature(repository)
			try:
				return loader(model_name, cache_dir=str(self._hub), **kwargs)
			finally:
				# Preserve completed downloads even if subsequent model setup fails.
				if archive.signature(repository) != before:
					print(f"Packing model cache: {archive.path}", flush=True)
					archive.publish(repository)
					metadata = archive.path.stat()
					self._restored[name] = (metadata.st_mtime_ns, metadata.st_size)


_stores: dict[Path, ModelStore] = {}
_stores_lock = threading.Lock()


def load_pretrained(loader: Callable[..., Any], model_name: str, **kwargs: Any) -> Any:
	"""Apply MODEL_PATH to Hub downloads; retain standard behavior otherwise."""
	model_path = os.environ.get("MODEL_PATH", "").strip()
	if not model_path or Path(model_name).expanduser().is_dir():
		return loader(model_name, **kwargs)
	root = Path(model_path).expanduser().resolve()
	with _stores_lock:
		if root not in _stores:
			_stores[root] = ModelStore(root)
	return _stores[root].load(loader, model_name, **kwargs)
