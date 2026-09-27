"""Extract frozen Qwen states for canonical A/B response pairs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from transformers import AutoTokenizer

from ...config import GPUConfig, ModelConfig
from ...errors import ModelExecutionError
from .cache import CachedEmbeddings
from .data import PairwiseText
from .hardware import GPUEnvironment
from .model import PairwiseQwenClassifier


class QwenEmbeddingExtractor:
    """Tokenize response pairs and pool the frozen shared backbone."""

    def __init__(
        self,
        gpu_config: GPUConfig,
        model_config: ModelConfig,
        devices: tuple[torch.device, ...],
    ) -> None:
        self._devices = devices
        self._config = model_config
        dtype = GPUEnvironment.PRECISION_DTYPES[gpu_config.precision]

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
            self._tokenizer.padding_side = "right"
            self._tokenizer.truncation_side = "right"
            if self._tokenizer.pad_token_id is None:
                if self._tokenizer.eos_token_id is None:
                    raise ModelExecutionError(
                        "The configured tokenizer has neither a pad token nor an EOS token."
                    )
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = PairwiseQwenClassifier(
                model_name=model_config.model_name,
                classifier_hidden_size=model_config.hidden_size,
                dropout=model_config.dropout,
                devices=self._devices,
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

    def extract(self, records: Sequence[PairwiseText]) -> CachedEmbeddings:
        """Return float16 CPU embeddings in canonical record order."""
        if not records:
            raise ModelExecutionError(
                "No canonical records are available for extraction."
            )

        ids: list[int] = []
        h_a_batches: list[torch.Tensor] = []
        h_b_batches: list[torch.Tensor] = []
        batch_size = self._config.extraction_batch_size
        total_batches = math.ceil(len(records) / batch_size)

        try:
            for batch_number, start in enumerate(
                range(0, len(records), batch_size),
                start=1,
            ):
                batch = records[start : start + batch_size]
                flat_texts = [
                    text
                    for record in batch
                    for text in (record.response_a, record.response_b)
                ]
                tokenized = self._tokenizer(
                    flat_texts,
                    padding=True,
                    truncation=True,
                    max_length=self._config.max_length,
                    return_tensors="pt",
                )
                length = int(tokenized["input_ids"].shape[1])
                pair_count = len(batch)
                input_ids = tokenized["input_ids"].reshape(pair_count, 2, length)
                attention_mask = tokenized["attention_mask"].reshape(
                    pair_count, 2, length
                )
                pooled = self._model.encode(
                    input_ids,
                    attention_mask,
                ).to(device="cpu", dtype=torch.float16)

                ids.extend(record.row_id for record in batch)
                h_a_batches.append(pooled[:, 0].contiguous())
                h_b_batches.append(pooled[:, 1].contiguous())
                if batch_number == 1 or batch_number % 500 == 0:
                    print(
                        f"Extracted embedding batch {batch_number:,}/{total_batches:,}."
                    )
        except torch.cuda.OutOfMemoryError as error:
            raise ModelExecutionError(
                "CUDA ran out of memory during embedding extraction. Lower "
                "the selected model's extraction_batch_size or max_length."
            ) from error
        except ModelExecutionError:
            raise
        except Exception as error:
            raise ModelExecutionError(
                f"Could not extract Qwen embeddings: {error}"
            ) from error

        print(f"Extracted embeddings for {len(records):,} canonical records.")
        return CachedEmbeddings(
            ids=torch.tensor(ids, dtype=torch.int64),
            h_a=torch.cat(h_a_batches),
            h_b=torch.cat(h_b_batches),
        )
