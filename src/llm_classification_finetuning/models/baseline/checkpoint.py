"""Persist and validate trained baseline classifier-head parameters."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor

from ..errors import BaselineError
from .model import PairwiseClassificationHead


@dataclass(frozen=True, slots=True)
class HeadCheckpointCompatibility:
    """Configuration fields that must match when loading a trained head."""

    schema_version: int
    model_name: str
    max_length: int
    serializer_version: str
    precision: str
    attention_implementation: str
    allow_tf32: bool
    classifier_hidden_size: int
    class_count: int = 3


@dataclass(frozen=True, slots=True)
class LoadedHeadCheckpoint:
    """Validated classifier parameters and their backbone input dimension."""

    state_dict: dict[str, Tensor]
    backbone_hidden_size: int


class BaselineCheckpointStore:
    """Atomically replace and strictly load one baseline head checkpoint."""

    def __init__(self, checkpoint_path: Path) -> None:
        self._checkpoint_path = checkpoint_path.resolve()

    @property
    def path(self) -> Path:
        """Return the configured checkpoint location."""
        return self._checkpoint_path

    def save(
        self,
        head: PairwiseClassificationHead,
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
            raise BaselineError(
                f"Could not write baseline checkpoint {self._checkpoint_path}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def load(
        self,
        expected: HeadCheckpointCompatibility,
    ) -> LoadedHeadCheckpoint:
        """Load a checkpoint and reject incompatible or invalid contents."""
        if not self._checkpoint_path.is_file():
            raise BaselineError(
                f"Baseline checkpoint not found: {self._checkpoint_path}. "
                "Run 'uv run baseline --build' first."
            )
        try:
            checkpoint = torch.load(
                self._checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(checkpoint, dict):
                raise TypeError("checkpoint root must be a mapping")
            metadata = checkpoint.get("metadata")
            state_dict = checkpoint.get("state_dict")
            if not isinstance(metadata, dict) or not isinstance(state_dict, dict):
                raise TypeError("metadata and state_dict mappings are required")

            mismatches = [
                key
                for key, value in asdict(expected).items()
                if metadata.get(key) != value
            ]
            if mismatches:
                fields = ", ".join(mismatches)
                raise BaselineError(
                    "Baseline checkpoint is incompatible with the current "
                    f"configuration ({fields}). Run 'uv run baseline --build'."
                )

            backbone_hidden_size = metadata.get("backbone_hidden_size")
            if (
                isinstance(backbone_hidden_size, bool)
                or not isinstance(backbone_hidden_size, int)
                or backbone_hidden_size < 1
            ):
                raise ValueError("backbone_hidden_size must be a positive integer")
            if not state_dict or not all(
                isinstance(name, str) and isinstance(value, Tensor)
                for name, value in state_dict.items()
            ):
                raise ValueError("state_dict must contain tensor parameters")
            if not all(torch.isfinite(value).all() for value in state_dict.values()):
                raise ValueError("state_dict contains non-finite parameters")
            return LoadedHeadCheckpoint(
                state_dict=state_dict,
                backbone_hidden_size=backbone_hidden_size,
            )
        except BaselineError:
            raise
        except Exception as error:
            raise BaselineError(
                f"Could not load baseline checkpoint {self._checkpoint_path}: {error}"
            ) from error
