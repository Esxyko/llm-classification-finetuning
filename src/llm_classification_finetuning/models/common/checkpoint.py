"""Persist trained model classifier-head parameters."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

from ...errors import ModelExecutionError


@dataclass(frozen=True, slots=True)
class HeadCheckpointCompatibility:
	"""Model identity and training settings recorded with a trained head."""

	schema_version: int
	model_name: str
	max_length: int
	serializer_version: str
	precision: str
	attention_implementation: str
	allow_tf32: bool
	classifier_hidden_size: int
	class_count: int = 3
	ab_swap: str = "aug"


class ModelCheckpointStore:
	"""Atomically replace one model head checkpoint."""

	def __init__(self, checkpoint_path: Path) -> None:
		self._checkpoint_path = checkpoint_path.resolve()

	@property
	def path(self) -> Path:
		"""Return the configured checkpoint location."""
		return self._checkpoint_path

	def save(
		self,
		head: nn.Module,
		compatibility: HeadCheckpointCompatibility,
		*,
		backbone_hidden_size: int,
		training_source_sha256: str,
	) -> None:
		"""Atomically replace the checkpoint with CPU classifier parameters."""
		temporary_path: Path | None = None
		try:
			self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
			with tempfile.NamedTemporaryFile(
				dir=self._checkpoint_path.parent,
				prefix=".head-",
				suffix=".tmp.pt",
				delete=False,
			) as temporary_file:
				temporary_path = Path(temporary_file.name)
			torch.save(
				{
					"metadata": {
						**asdict(compatibility),
						"backbone_hidden_size": backbone_hidden_size,
						"training_source_sha256": training_source_sha256,
						"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
					},
					"state_dict": {
						name: parameter.detach().to(device="cpu")
						for name, parameter in head.state_dict().items()
					},
				},
				temporary_path,
			)
			os.replace(temporary_path, self._checkpoint_path)
			temporary_path = None
		except Exception as error:
			raise ModelExecutionError(
				f"Could not write model checkpoint {self._checkpoint_path}: {error}"
			) from error
		finally:
			if temporary_path is not None:
				try:
					temporary_path.unlink(missing_ok=True)
				except OSError:
					pass
