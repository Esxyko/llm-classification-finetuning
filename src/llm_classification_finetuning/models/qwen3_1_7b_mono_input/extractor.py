"""Extract frozen Qwen states for structured mono inputs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from transformers import AutoTokenizer

from ...config import GPUConfig, ModelConfig
from ...errors import ModelExecutionError
from ..common.hardware import GPUEnvironment
from .cache import MonoInputCachedEmbeddings
from .data import MonoInputText, StructuredInputSerializer
from .model import FrozenQwenEncoder


class MonoInputEmbeddingExtractor:
    """Tokenize complete comparisons and pool the frozen backbone."""

    def __init__(
        self,
        gpu_config: GPUConfig,
        model_config: ModelConfig,
        devices: tuple[torch.device, ...],
    ) -> None:
        self._config = model_config
        self._serializer = StructuredInputSerializer()
        dtype = GPUEnvironment.PRECISION_DTYPES[gpu_config.precision]
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
            self._tokenizer.padding_side = "right"
            if self._tokenizer.pad_token_id is None:
                if self._tokenizer.eos_token_id is None:
                    raise ModelExecutionError(
                        "The configured tokenizer has neither a pad token nor an EOS token."
                    )
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._encoder = FrozenQwenEncoder(
                model_name=model_config.model_name,
                devices=devices,
                dtype=dtype,
                attention_implementation=gpu_config.attention_implementation,
            )
        except torch.cuda.OutOfMemoryError as error:
            raise ModelExecutionError(
                "CUDA ran out of memory while loading the Qwen backbone. Close other "
                "GPU workloads or select a lower-precision gpu.precision setting."
            ) from error
        except ModelExecutionError:
            raise
        except Exception as error:
            raise ModelExecutionError(
                f"Could not load {model_config.model_name}: {error}"
            ) from error

    def extract(
        self,
        records: Sequence[MonoInputText],
    ) -> MonoInputCachedEmbeddings:
        if not records:
            raise ModelExecutionError(
                "No mono-input records are available for extraction."
            )

        ids: list[int] = []
        orientations: list[bool] = []
        batches: list[torch.Tensor] = []
        batch_size = self._config.extraction_batch_size
        total_batches = math.ceil(len(records) / batch_size)
        try:
            for batch_number, start in enumerate(
                range(0, len(records), batch_size),
                start=1,
            ):
                batch = records[start : start + batch_size]
                texts = [record.text for record in batch]
                tokenized = self._tokenizer(
                    texts,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                if tokenized["input_ids"].shape[1] > self._config.max_length:
                    lengths = tokenized["attention_mask"].sum(dim=1).tolist()
                    texts = [
                        self._serializer.fit_to_length(
                            record,
                            self._tokenizer,
                            self._config.max_length,
                        )
                        if length > self._config.max_length
                        else record.text
                        for record, length in zip(batch, lengths, strict=True)
                    ]
                    tokenized = self._tokenizer(
                        texts,
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    )
                    if tokenized["input_ids"].shape[1] > self._config.max_length:
                        raise ModelExecutionError(
                            "Balanced mono-input serialization exceeded max_length."
                        )
                pooled = self._encoder(
                    tokenized["input_ids"],
                    tokenized["attention_mask"],
                ).to(device="cpu", dtype=torch.float16)
                ids.extend(record.row_id for record in batch)
                orientations.extend(record.is_swapped for record in batch)
                batches.append(pooled.contiguous())
                if batch_number == 1 or batch_number % 500 == 0:
                    print(
                        "Extracted mono-input embedding batch "
                        f"{batch_number:,}/{total_batches:,}."
                    )
        except torch.cuda.OutOfMemoryError as error:
            raise ModelExecutionError(
                "CUDA ran out of memory during mono-input embedding extraction. "
                "Lower qwen3_1_7b_mono_input.extraction_batch_size or max_length."
            ) from error
        except ModelExecutionError:
            raise
        except Exception as error:
            raise ModelExecutionError(
                f"Could not extract mono-input Qwen embeddings: {error}"
            ) from error

        print(f"Extracted embeddings for {len(records):,} mono-input records.")
        return MonoInputCachedEmbeddings(
            ids=torch.tensor(ids, dtype=torch.int64),
            is_swapped=torch.tensor(orientations, dtype=torch.bool),
            features=torch.cat(batches),
        )


__all__ = ("MonoInputEmbeddingExtractor",)
