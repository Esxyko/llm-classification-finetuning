# Caching and Python APIs

[Documentation index](../README.md)

Caches sit beside the processed Parquet selected by
[`data.processed_path`](configuration.md#configuration-files-and-paths).
Frozen embeddings, shared text encodings, and LoRA batch plans have separate
compatibility rules. LoRA warmup, joint-validation, and submission features stay
in temporary FP32 CPU memory and are not persisted. Warmup reuses its feature
pool; each joint-validation or submission call extracts fresh features with the
current model and releases them on return.

## Frozen embeddings

The frozen model shares content-addressed embeddings between training and test
under `embedding_cache/` beside the processed Parquet. Each SQLite shard
identifies the model and exact Hub revision (or local checkpoint/configuration
content), backbone hidden size, precision, attention implementation, TF32,
pooling, storage format, and extraction version. Individual entries identify
the effective token IDs of one branch, after any requested truncation. Different
compute identities coexist without overwriting one another.
Pooled vectors are stored as FP16 CPU tensors, including when backbone compute
precision is BF16 or FP32; the classifier consumes them as FP32.

Changing labels, folds, IDs, row order, or dataset file encoding does not
invalidate existing vectors. Overlapping datasets and A/B branches reuse each
other's entries; only missing inputs are encoded. Changing `max_length` or the
serializer also reuses vectors when the resulting token IDs remain identical.
Cache rows are mapped back to the current canonical IDs and order. When all
inputs hit, the backbone weights are not loaded. Local checkpoint identities
require reading the selected weight files to hash their content.

Missing branches are right-padded and encoded independently; changing batch
composition can change floating-point rounding. The precision, attention, and
TF32 boundaries remain separate. Corrupt vectors are recomputed, SQLite writes
are transactional, and corrupt shards are archived before rebuilding. Older
dataset-wide `.pt`/`.json` embedding caches, including `baseline_*`, are left
untouched and are no longer read; their metadata cannot establish the new
backbone/input identities. Embedding reuse does not imply saved-head
compatibility; see the separate [checkpoint rules](models.md#checkpoint-compatibility).

## Shared token encodings

`token_cache/` beside the processed Parquet is shared by frozen-model extraction,
LoRA training/warmup/validation, and LoRA test inference. Each SQLite shard
identifies the tokenizer's encoding content, implementation/library versions,
encoding options, and cache schema; entries identify individual UTF-8 texts by
SHA-256. IDs, labels, folds, row order, dataset fingerprints, model names, model
weight revisions, and training settings are excluded. Only changed or missing
texts are encoded; duplicates and overlapping datasets reuse existing entries.
The tokenizer is loaded to fingerprint its actual content, but cache hits do not
run tokenization. Padding is applied by each caller after cache lookup.

Full untruncated encodings are the common reusable representation. Frozen-model
extraction additionally caches the tokenizer's exact truncated output for texts
exceeding its selected length limit, preserving special-token behavior. LoRA
always validates the full cached lengths against `max_length` and rejects empty
or over-limit branches without truncation. Changing length limits therefore
keeps full tokens reusable. Corrupt entries are recomputed; corrupt SQLite
shards are archived and rebuilt. SQLite transactions publish complete entries.

## LoRA batch plans

LoRA's `qwen3_4b_lora_batches/plans/*.pt` files remain separate and identify the
sorted selected `(id, is_swapped, A_length, B_length)` rows, budget, update group
size, and planning algorithm version. Labels, fold numbers, reference order,
text/token content beyond lengths, and unselected validation inputs do not enter
the plan key. Cached groups store sample identities and are mapped to current
dataset indices on every read; current labels are always used for training.
A/B mode and fold selection determine membership before lookup. Changing
budget or effective batch size rebuilds plans while reusing tokens. With
`training_token_budget: 0`, tokens are cached but no global plan is created.
Earlier LoRA `tokens/*.pt` files are left untouched and are no longer read.

`--refresh-cache` replaces requested token entries and model-specific caches;
shared training encodings remain reusable between folds in that run. Warmup
features and live LoRA encodings are never persisted in these token caches.

## Python APIs

The independent text API accepts arbitrary serialized strings and a shared
cache directory; an already loaded tokenizer can also be passed to
`TokenizationCache(cache_dir, tokenizer)`:

```python
from pathlib import Path

from llm_classification_finetuning.models.common import TokenizationCache

cache = TokenizationCache.from_pretrained(
    Path("data/processed/token_cache"), "Qwen/Qwen3-4B"
)
encoded = cache.encode(["first input", "second input"])
first_ids = encoded[0]  # Unpadded int32 token IDs, in requested input order.
```

The Python API can prepare a CPU loader without loading model weights or
initializing CUDA:

```python
from pathlib import Path

from llm_classification_finetuning.config import AppConfig
from llm_classification_finetuning.models.qwen3_4b_lora import (
    LoRATrainingBatchRepository,
)

config = AppConfig.load(Path("config.yaml"))
batches = LoRATrainingBatchRepository.from_config(config)
training = batches.get_training(fold=0)  # None selects a full-data build.
training.set_epoch(0)  # Zero-based joint epoch; fixed batching keeps its RNG stream.
loader = training.loader  # Yields PairwiseBatch; training.tokens is shared.
```

See [LoRA training](lora.md#optional-token-budget-training-batches) for planning
behavior and effective-batch semantics. The source modules that implement these
APIs and their development invariants are mapped in [structure.md](../structure.md).
