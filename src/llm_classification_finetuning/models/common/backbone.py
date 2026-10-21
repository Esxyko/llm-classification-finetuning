"""Content identities for local backbone checkpoints shared by model caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...errors import ModelExecutionError


class LocalBackboneFingerprint:
	"""Hash checkpoint/configuration content independently of its directory."""

	@staticmethod
	def compute(path: Path) -> str:
		root = path.resolve()
		files = [root / "config.json"]
		for name in (
			"model.safetensors",
			"model.safetensors.index.json",
			"pytorch_model.bin",
			"pytorch_model.bin.index.json",
		):
			checkpoint = root / name
			if not checkpoint.is_file():
				continue
			files.append(checkpoint)
			if name.endswith(".index.json"):
				index = json.loads(checkpoint.read_text(encoding="utf-8"))
				files.extend(
					root / name for name in sorted(set(index["weight_map"].values()))
				)
			break
		else:
			raise ModelExecutionError(
				f"No supported local backbone checkpoint found in {root}."
			)
		digest = hashlib.sha256()
		for file_path in files:
			if (
				not file_path.is_relative_to(root)
				or ".." in file_path.relative_to(root).parts
			):
				raise ModelExecutionError(
					"Local backbone checkpoint shards must remain inside the model directory."
				)
			# Keep snapshot-relative names; Hub snapshots may symlink to blobs.
			name = file_path.relative_to(root).as_posix().encode("utf-8")
			digest.update(len(name).to_bytes(8, "big"))
			digest.update(name)
			digest.update(file_path.stat().st_size.to_bytes(8, "big"))
			with file_path.open("rb") as stream:
				for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
					digest.update(chunk)
		return digest.hexdigest()
