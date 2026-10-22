"""Pack cached Hugging Face models into the configured MODEL_PATH."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from llm_classification_finetuning.model_storage import ModelArchive


class HuggingFaceModelMigrator:
	"""Publish one verified archive per repository before removing source data."""

	def __init__(self, source: Path, destination: Path) -> None:
		self._source = source.expanduser().resolve()
		self._destination = destination.expanduser().resolve()

	def run(self, *, move: bool, dry_run: bool) -> None:
		"""Pack repositories, preserving revisions, blobs, and internal file links."""
		print(f"Source Hub cache: {self._source}", flush=True)
		print(f"Destination model archives: {self._destination}", flush=True)
		if not self._source.is_dir():
			raise ValueError(f"Source Hub cache does not exist: {self._source}")
		repositories = sorted(self._source.glob("models--*"))
		repositories = [path for path in repositories if not path.name.endswith(".zip")]
		for repository in repositories:
			self._validate_repository(repository)
			ModelArchive.entries(repository)
			archive = ModelArchive(self._destination / f"{repository.name}.zip")
			if archive.path.is_relative_to(repository.resolve()):
				raise ValueError(
					"Destination archives must be outside source model directories."
				)
			if archive.path.exists():
				raise ValueError(f"Destination archive already exists: {archive.path}")
		if not repositories:
			print("No cached model repositories found.")
			return

		for repository in repositories:
			archive = ModelArchive(self._destination / f"{repository.name}.zip")
			print(
				f"Would pack {'and remove ' if move else ''}{repository} -> {archive.path}"
				if dry_run
				else f"Packing and verifying {repository.name}...",
				flush=True,
			)
			if dry_run:
				continue
			with archive.lock():
				if archive.path.exists():
					raise ValueError(
						f"Destination archive appeared during migration: {archive.path}"
					)
				archive.publish(repository)
				if move:
					# Check the resolved deletion boundary immediately before removal.
					self._validate_repository(repository)
					if any(repository.rglob("*.incomplete")):
						raise ValueError(
							f"Archive published, but unfinished downloads remain in {repository}; "
							"the source was retained."
						)
					shutil.rmtree(repository)
			print(f"{'Moved' if move else 'Packed'} {archive.path}", flush=True)
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


def _original_hub_cache() -> Path:
	"""Resolve native Hub settings without importing Hugging Face libraries."""
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
		help="Old Hub directory containing models--* folders (default: MODEL_PATH/huggingface/hub if present, otherwise native HF settings).",
	)
	parser.add_argument(
		"--move",
		action="store_true",
		help="Remove each source model directory after its archive passes SHA-256 verification.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Preview migration without creating or deleting files.",
	)
	args = parser.parse_args()
	load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
	model_path = os.environ.get("MODEL_PATH", "").strip()
	if not model_path:
		parser.error("Set a non-empty MODEL_PATH in .env or the environment first.")
	try:
		destination = Path(model_path).expanduser().resolve()
		legacy = destination / "huggingface" / "hub"
		source = (
			args.source
			if args.source is not None
			else (legacy if legacy.is_dir() else _original_hub_cache())
		)
		HuggingFaceModelMigrator(source, destination).run(
			move=args.move, dry_run=args.dry_run
		)
	except (OSError, ValueError, shutil.Error, zipfile.BadZipFile) as error:
		print(f"Model migration failed: {error}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
