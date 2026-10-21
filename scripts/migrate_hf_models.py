"""Migrate cached Hugging Face models into the configured DATA_PATH."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv


class HuggingFaceModelMigrator:
	"""Copy complete model caches, preserving snapshots and verifying file contents."""

	def __init__(self, source: Path, destination: Path) -> None:
		self._source = source.expanduser().resolve()
		self._destination = destination.expanduser().resolve()

	def run(self, *, move: bool, dry_run: bool) -> None:
		"""Publish verified repositories and optionally remove their source copies."""
		print(f"Source Hub cache: {self._source}", flush=True)
		print(f"Destination Hub cache: {self._destination}", flush=True)
		if self._source == self._destination:
			print("Source and destination are identical; nothing to migrate.")
			return
		if self._source.is_relative_to(
			self._destination
		) or self._destination.is_relative_to(self._source):
			raise ValueError(
				"Source and destination cache directories must not overlap."
			)
		if not self._source.is_dir():
			raise ValueError(f"Source Hub cache does not exist: {self._source}")

		repositories = sorted(self._source.glob("models--*"))
		for repository in repositories:
			self._validate_repository(repository)
			target = self._destination / repository.name
			if target.exists() or target.is_symlink():
				raise ValueError(
					f"Destination model already exists: {target}. "
					"No existing model directories will be overwritten."
				)
		if not repositories:
			print("No cached model repositories found.")
			return

		for repository in repositories:
			target = self._destination / repository.name
			print(
				f"{'Would move' if move else 'Would copy'} {repository.name}"
				if dry_run
				else f"Copying and verifying {repository.name}...",
				flush=True,
			)
			if dry_run:
				continue
			self._destination.mkdir(parents=True, exist_ok=True)
			with TemporaryDirectory(
				prefix=".migrate-hf-", dir=self._destination
			) as work:
				staged = Path(work) / repository.name
				self._copy_repository(repository, staged)
				self._verify_repository(repository, staged)
				if target.exists() or target.is_symlink():
					raise ValueError(
						f"Destination model appeared during migration: {target}"
					)
				staged.rename(target)
			if move:
				# Recheck the resolved deletion boundary immediately before removal.
				self._validate_repository(repository)
				shutil.rmtree(repository)
			print(f"{'Moved' if move else 'Copied'} {repository.name}", flush=True)
		print(f"{'Previewed' if dry_run else 'Migrated'} {len(repositories)} model(s).")

	def _validate_repository(self, repository: Path) -> None:
		if (
			not repository.is_dir()
			or repository.is_symlink()
			or repository.is_junction()
			or repository.resolve().parent != self._source
		):
			raise ValueError(
				f"Expected a model directory directly in the cache: {repository}"
			)

	@staticmethod
	def _entries(repository: Path) -> list[Path]:
		entries = sorted(repository.rglob("*"))
		for entry in entries:
			if entry.is_junction():
				raise ValueError(f"Directory junctions are not supported: {entry}")
			if entry.is_symlink():
				resolved = entry.resolve(strict=True)
				if not resolved.is_relative_to(repository) or not resolved.is_file():
					raise ValueError(
						f"Expected a file link within its model cache: {entry}"
					)
			elif not entry.is_dir() and not entry.is_file():
				raise ValueError(f"Unsupported cache entry: {entry}")
		return entries

	def _copy_repository(self, source: Path, destination: Path) -> None:
		destination.mkdir()
		for entry in self._entries(source):
			target = destination / entry.relative_to(source)
			if entry.is_dir():
				target.mkdir(parents=True, exist_ok=True)
				continue
			target.parent.mkdir(parents=True, exist_ok=True)
			if entry.is_symlink():
				# Rebase absolute links too, so deleting the source cannot break snapshots.
				linked_file = destination / entry.resolve().relative_to(source)
				try:
					target.symlink_to(os.path.relpath(linked_file, target.parent))
					continue
				except OSError as error:
					if getattr(error, "winerror", None) != 1314:
						raise
					# Windows without symlink privileges uses regular snapshot files.
			shutil.copy2(entry, target)

	def _verify_repository(self, source: Path, destination: Path) -> None:
		entries = self._entries(source)
		if {entry.relative_to(source) for entry in entries} != {
			entry.relative_to(destination) for entry in self._entries(destination)
		}:
			raise ValueError(f"Cache entries changed during migration: {source}")
		for entry in entries:
			if not entry.is_file():
				continue
			target = destination / entry.relative_to(source)
			if (
				not target.is_file()
				or entry.stat().st_size != target.stat().st_size
				or self._sha256(entry) != self._sha256(target)
			):
				raise ValueError(f"Copied cache file failed verification: {entry}")

	@staticmethod
	def _sha256(path: Path) -> bytes:
		with path.open("rb") as stream:
			return hashlib.file_digest(stream, "sha256").digest()


def _original_hub_cache() -> Path:
	"""Resolve the old Hub cache without importing project/Hugging Face settings."""
	default_home = Path.home() / ".cache"
	cache_home = os.environ.get("XDG_CACHE_HOME", str(default_home))
	hf_home = os.environ.get("HF_HOME", str(Path(cache_home) / "huggingface"))
	legacy_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub"))
	return Path(
		os.path.expandvars(os.environ.get("HF_HUB_CACHE", legacy_cache))
	).expanduser()


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--source",
		type=Path,
		help="Old Hub cache containing models--* directories (default: HF cache settings).",
	)
	parser.add_argument(
		"--move",
		action="store_true",
		help="Delete each source model directory after its copy passes SHA-256 verification.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Preview migration without changing files.",
	)
	args = parser.parse_args()
	load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
	data_path = os.environ.get("DATA_PATH", "").strip()
	if not data_path:
		parser.error("Set a non-empty DATA_PATH in .env or the environment first.")
	try:
		destination = Path(data_path).expanduser().resolve() / "huggingface" / "hub"
		migrator = HuggingFaceModelMigrator(
			args.source if args.source is not None else _original_hub_cache(),
			destination,
		)
		migrator.run(move=args.move, dry_run=args.dry_run)
	except (OSError, ValueError, shutil.Error) as error:
		print(f"Model migration failed: {error}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
