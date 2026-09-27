"""Persist and validate frozen pooled Qwen embeddings."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from ..errors import BaselineError


@dataclass(frozen=True, slots=True)
class EmbeddingCacheKey:
    """Inputs whose changes require fresh backbone features."""

    schema_version: int
    processed_sha256: str
    model_name: str
    max_length: int
    serializer_version: str
    precision: str
    attention_implementation: str
    allow_tf32: bool


@dataclass(frozen=True, slots=True)
class CachedEmbeddings:
    """Canonical IDs and their A/B pooled states on CPU."""

    ids: Tensor
    h_a: Tensor
    h_b: Tensor

    @property
    def hidden_size(self) -> int:
        """Return the Qwen hidden dimension."""
        return int(self.h_a.shape[1])


class EmbeddingCache:
    """Read and atomically replace a local embedding cache."""

    def __init__(
        self,
        processed_path: Path,
        artifact_stem: str = "baseline_embeddings",
    ) -> None:
        self._tensor_path = processed_path.parent / f"{artifact_stem}.pt"
        self._metadata_path = processed_path.parent / f"{artifact_stem}.json"

    @property
    def tensor_path(self) -> Path:
        """Return the cache tensor path for status reporting."""
        return self._tensor_path

    def load(
        self,
        expected_key: EmbeddingCacheKey,
        expected_ids: Tensor,
    ) -> CachedEmbeddings | None:
        """Return a valid cache, or None when it is absent, stale, or corrupt."""
        if not self._tensor_path.is_file() or not self._metadata_path.is_file():
            return None
        try:
            metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            expected_metadata = asdict(expected_key)
            if any(
                metadata.get(key) != value for key, value in expected_metadata.items()
            ):
                return None

            tensors = torch.load(
                self._tensor_path,
                map_location="cpu",
                weights_only=True,
            )
            cached = CachedEmbeddings(
                ids=tensors["ids"],
                h_a=tensors["h_a"],
                h_b=tensors["h_b"],
            )
            self._validate(cached, expected_ids)
            return cached
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            json.JSONDecodeError,
        ):
            return None

    def save(
        self,
        cache_key: EmbeddingCacheKey,
        embeddings: CachedEmbeddings,
    ) -> None:
        """Write cache tensors and metadata through same-directory temporary files."""
        tensor_temporary: Path | None = None
        metadata_temporary: Path | None = None
        try:
            self._tensor_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=self._tensor_path.parent,
                prefix=".baseline-embeddings-",
                suffix=".tmp.pt",
                delete=False,
            ) as temporary_file:
                tensor_temporary = Path(temporary_file.name)
            torch.save(
                {
                    "ids": embeddings.ids.to(device="cpu", dtype=torch.int64),
                    "h_a": embeddings.h_a.to(device="cpu", dtype=torch.float16),
                    "h_b": embeddings.h_b.to(device="cpu", dtype=torch.float16),
                },
                tensor_temporary,
            )

            metadata = {
                **asdict(cache_key),
                "rows": int(embeddings.ids.shape[0]),
                "hidden_size": embeddings.hidden_size,
                "storage_dtype": "float16",
            }
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._metadata_path.parent,
                prefix=".baseline-embeddings-",
                suffix=".tmp.json",
                delete=False,
            ) as temporary_file:
                metadata_temporary = Path(temporary_file.name)
                json.dump(metadata, temporary_file, indent=2, sort_keys=True)
                temporary_file.write("\n")

            os.replace(tensor_temporary, self._tensor_path)
            tensor_temporary = None
            os.replace(metadata_temporary, self._metadata_path)
            metadata_temporary = None
        except Exception as error:
            raise BaselineError(
                f"Could not write embedding cache {self._tensor_path}: {error}"
            ) from error
        finally:
            for temporary_path in (tensor_temporary, metadata_temporary):
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        pass

    @staticmethod
    def _validate(embeddings: CachedEmbeddings, expected_ids: Tensor) -> None:
        if embeddings.ids.ndim != 1 or not torch.equal(
            embeddings.ids.to(dtype=torch.int64), expected_ids.to(dtype=torch.int64)
        ):
            raise ValueError("Cached IDs do not match canonical source IDs.")
        if (
            embeddings.h_a.ndim != 2
            or embeddings.h_a.shape != embeddings.h_b.shape
            or embeddings.h_a.shape[0] != embeddings.ids.shape[0]
            or embeddings.h_a.shape[1] < 1
            or not embeddings.h_a.is_floating_point()
            or not embeddings.h_b.is_floating_point()
            or not torch.isfinite(embeddings.h_a).all()
            or not torch.isfinite(embeddings.h_b).all()
        ):
            raise ValueError("Cached embedding tensors have invalid shapes or values.")
