"""Extract Qwen3-4B-lora-minimal.zip from FILE_PATH into DATA_PATH."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

ARCHIVE_NAME = "Qwen3-4B-lora-minimal.zip"
REPOSITORY_NAME = "models--Qwen--Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SNAPSHOT_FILES = (
	"config.json",
	"tokenizer.json",
	"tokenizer_config.json",
	"model.safetensors.index.json",
	"LICENSE",
	"model-00001-of-00003.safetensors",
	"model-00002-of-00003.safetensors",
	"model-00003-of-00003.safetensors",
)


class Qwen3ArchiveExtractor:
	"""Stage and validate the uploaded minimal archive before publishing its cache."""

	def __init__(self, file_path: Path, data_path: Path) -> None:
		self._root = data_path.expanduser().resolve()
		self._archive = file_path.expanduser().resolve() / ARCHIVE_NAME
		self._destination = self._root / REPOSITORY_NAME

	def run(self) -> None:
		"""Extract ordinary files, preserving the archive and any existing cache."""
		self._check_destination()
		if not self._archive.is_file():
			raise ValueError(
				f"Place {ARCHIVE_NAME} in FILE_PATH: {self._archive.parent}"
			)
		print(f"Archive: {self._archive}", flush=True)
		print(f"Destination: {self._destination}", flush=True)
		self._verify_checksum()
		with zipfile.ZipFile(self._archive) as archive:
			members = self._validated_members(archive)
			total_bytes = sum(member.file_size for member in members)
			self._root.mkdir(parents=True, exist_ok=True)
			if shutil.disk_usage(self._root).free < total_bytes:
				raise ValueError(
					f"Not enough free space in DATA_PATH; need {total_bytes:,} bytes."
				)
			work_root = self._root / ".cache-work"
			if work_root.resolve().parent != self._root:
				raise ValueError(
					f"Extraction work directory must stay in DATA_PATH: {work_root}"
				)
			work_root.mkdir(exist_ok=True)
			with TemporaryDirectory(prefix="extract-qwen3-4b-", dir=work_root) as work:
				staging = Path(work)
				for number, member in enumerate(members, start=1):
					print(
						f"[{number}/{len(members)}] Extracting {member.filename}",
						flush=True,
					)
					target = staging / member.filename
					target.parent.mkdir(parents=True, exist_ok=True)
					with (
						archive.open(member) as source,
						target.open("xb") as destination,
					):
						# Reading to EOF checks each ZIP member's CRC before publication.
						shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
					if target.stat().st_size != member.file_size:
						raise ValueError(
							f"Incomplete extracted file: {member.filename}"
						)
				self._check_destination()
				(staging / REPOSITORY_NAME).rename(self._destination)
		print(f"Extracted {len(members)} files ({total_bytes:,} bytes).", flush=True)
		print(f"To load Qwen/Qwen3-4B, set MODEL_PATH to: {self._root}")

	def _check_destination(self) -> None:
		if self._destination.exists() or self._destination.is_symlink():
			raise ValueError(
				f"Destination model already exists: {self._destination}. "
				"Existing model directories will not be overwritten."
			)

	def _verify_checksum(self) -> None:
		checksum = self._archive.with_suffix(".zip.sha256")
		if not checksum.is_file():
			return
		fields = checksum.read_text(encoding="utf-8").split()
		if not fields or re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]) is None:
			raise ValueError(f"Invalid SHA-256 checksum file: {checksum}")
		print("Verifying archive SHA-256...", flush=True)
		with self._archive.open("rb") as source:
			actual = hashlib.file_digest(source, "sha256").hexdigest()
		if actual != fields[0].lower():
			raise ValueError("Archive SHA-256 mismatch; upload the complete ZIP again.")
		print("Archive SHA-256 verified.", flush=True)

	@staticmethod
	def _validated_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
		# Accept only the nine known paths, excluding traversal, links and duplicates.
		expected = {f"{REPOSITORY_NAME}/refs/main"} | {
			f"{REPOSITORY_NAME}/snapshots/{REVISION}/{name}" for name in SNAPSHOT_FILES
		}
		members = archive.infolist()
		if (
			len(members) != len(expected)
			or {member.filename for member in members} != expected
		):
			raise ValueError(
				f"Unexpected archive layout; use the supplied {ARCHIVE_NAME}."
			)
		for member in members:
			file_type = stat.S_IFMT(member.external_attr >> 16)
			if (
				file_type not in {0, stat.S_IFREG}
				or member.is_dir()
				or member.flag_bits & 1
			):
				raise ValueError(
					f"Expected an unencrypted regular file: {member.filename}"
				)
		if (
			archive.read(f"{REPOSITORY_NAME}/refs/main").decode("ascii").strip()
			!= REVISION
		):
			raise ValueError(
				"Archive revision does not match the supplied Qwen3-4B model."
			)
		return members


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.parse_args()
	load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
	file_path = os.environ.get("FILE_PATH", "").strip()
	data_path = os.environ.get("DATA_PATH", "").strip()
	if not file_path or not data_path:
		parser.error(
			"Set non-empty FILE_PATH and DATA_PATH directories in .env or the environment first."
		)
	try:
		Qwen3ArchiveExtractor(Path(file_path), Path(data_path)).run()
	except (OSError, ValueError, zipfile.BadZipFile, NotImplementedError) as error:
		print(f"Model extraction failed: {error}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
