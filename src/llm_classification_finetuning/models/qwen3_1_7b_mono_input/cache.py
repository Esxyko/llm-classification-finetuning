"""Persist and validate row-aligned mono-input Qwen embeddings."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from ...errors import ModelExecutionError
from ..common.cache import EmbeddingCacheKey


@dataclass(frozen=True, slots=True)
class MonoInputCachedEmbeddings:
    """Augmented row identities and their pooled states on CPU."""

    ids: Tensor
    is_swapped: Tensor
    features: Tensor

    @property
    def hidden_size(self) -> int:
        return int(self.features.shape[1])


class MonoInputEmbeddingCache:
    """Read and atomically replace a mono-input embedding cache."""

    def __init__(self, processed_path: Path, artifact_stem: str) -> None:
        self._tensor_path = processed_path.parent / f"{artifact_stem}.pt"
        self._metadata_path = processed_path.parent / f"{artifact_stem}.json"

    @property
    def tensor_path(self) -> Path:
        return self._tensor_path

    def load(
        self,
        expected_key: EmbeddingCacheKey,
        expected_ids: Tensor,
        expected_swapped: Tensor,
    ) -> MonoInputCachedEmbeddings | None:
        if not self._tensor_path.is_file() or not self._metadata_path.is_file():
            return None
        try:
            metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            if any(
                metadata.get(key) != value
                for key, value in asdict(expected_key).items()
            ):
                return None
            tensors = torch.load(
                self._tensor_path,
                map_location="cpu",
                weights_only=True,
            )
            cached = MonoInputCachedEmbeddings(
                ids=tensors["ids"],
                is_swapped=tensors["is_swapped"],
                features=tensors["features"],
            )
            self._validate(cached, expected_ids, expected_swapped)
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
        embeddings: MonoInputCachedEmbeddings,
    ) -> None:
        tensor_temporary: Path | None = None
        metadata_temporary: Path | None = None
        try:
            self._tensor_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=self._tensor_path.parent,
                prefix=f".{self._tensor_path.stem}-",
                suffix=".tmp.pt",
                delete=False,
            ) as temporary_file:
                tensor_temporary = Path(temporary_file.name)
            torch.save(
                {
                    "ids": embeddings.ids.to(device="cpu", dtype=torch.int64),
                    "is_swapped": embeddings.is_swapped.to(
                        device="cpu",
                        dtype=torch.bool,
                    ),
                    "features": embeddings.features.to(
                        device="cpu",
                        dtype=torch.float16,
                    ),
                },
                tensor_temporary,
            )
            metadata = {
                **asdict(cache_key),
                "rows": int(embeddings.ids.shape[0]),
                "hidden_size": embeddings.hidden_size,
                "storage_dtype": "float16",
                "layout": "row-aligned-mono-input-v1",
            }
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._metadata_path.parent,
                prefix=f".{self._metadata_path.stem}-",
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
            raise ModelExecutionError(
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
    def _validate(
        embeddings: MonoInputCachedEmbeddings,
        expected_ids: Tensor,
        expected_swapped: Tensor,
    ) -> None:
        if embeddings.ids.ndim != 1 or not torch.equal(
            embeddings.ids.to(dtype=torch.int64),
            expected_ids.to(dtype=torch.int64),
        ):
            raise ValueError("Cached IDs do not match serialized row order.")
        if embeddings.is_swapped.ndim != 1 or not torch.equal(
            embeddings.is_swapped.to(dtype=torch.bool),
            expected_swapped.to(dtype=torch.bool),
        ):
            raise ValueError("Cached orientations do not match serialized row order.")
        if (
            embeddings.features.ndim != 2
            or embeddings.features.shape[0] != embeddings.ids.shape[0]
            or embeddings.features.shape[1] < 1
            or not embeddings.features.is_floating_point()
            or not torch.isfinite(embeddings.features).all()
        ):
            raise ValueError("Cached mono-input embeddings are invalid.")


__all__ = ("MonoInputCachedEmbeddings", "MonoInputEmbeddingCache")
