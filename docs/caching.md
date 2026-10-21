# Caching and Python APIs

[Documentation index](../README.md)

Caches sit beside the processed Parquet selected by
[`data.processed_path`](configuration.md#configuration-files-and-paths).
Frozen embeddings, shared text encodings, adapter batch plans, and head-warmup
states have separate compatibility rules. LoRA/QLoRA warmup, joint-validation,
and submission features stay in temporary CPU memory (BF16 in BF16 mode,
otherwise FP32) and are not persisted. Uncached warmup reuses its feature
pool; each joint-validation or submission call extracts fresh features with the
current model and releases them on return.

## Frozen embeddings

The frozen model shares content-addressed embeddings between training and test
under `embedding_cache/` beside the processed Parquet. Each SQLite shard
identifies the model and exact Hub revision (or local checkpoint/configuration
content), backbone hidden size, precision, pooling, storage format, and
extraction version. Attention implementation and TF32 do not affect cache reuse.
Individual entries identify the effective token IDs of one branch, after any
requested truncation. Different compute identities coexist without overwriting
one another.
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
composition, attention implementation, or TF32 can change floating-point
rounding. Precision remains part of the cache identity. Reads validate blob
length and checksum without scanning vectors for NaN/Inf; newly extracted
vectors are checked for finite values before caching. Entries with invalid
lengths or checksums are recomputed, SQLite writes are transactional, and
corrupt shards are archived before rebuilding. SQLite shards keyed by the
former attention/TF32 identity are left untouched and are no longer reused.
Older dataset-wide `.pt`/`.json` embedding caches, including `baseline_*`, are left
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
Reads validate byte length, token count, and checksum without scanning token
IDs for negative values; newly encoded IDs are checked before caching.

## LoRA batch plans

LoRA's `qwen3_4b_lora_batches/plans/*.pt` files and QLoRA's independent
`qwen3_4b_qlora_batches/plans/*.pt` files identify the
sorted selected `(id, is_swapped, A_length, B_length)` rows, budget, update group
size, and planning algorithm version. Labels, fold numbers, reference order,
text/token content beyond lengths, and unselected validation inputs do not enter
the plan key. Cached groups store sample identities and are mapped to current
dataset indices on every read; current labels are always used for training.
A/B mode and fold selection determine membership before lookup. Changing
budget or effective batch size rebuilds plans while reusing tokens. With
`training_token_budget: 0`, tokens are cached but no global plan is created.
Earlier LoRA `tokens/*.pt` files are left untouched and are no longer read.
Both adapter profiles share the text-token cache when tokenizer content and
encoding options match; quantization does not change token identity.

`--refresh-cache` replaces requested token entries and model-specific caches,
including head warmup. Shared training encodings remain reusable between folds
in that run. Warmup
features and live LoRA encodings are never persisted in these token caches.

## LoRA/QLoRA head warmup

Cross-validation and builds automatically save completed head warmup under
`qwen3_4b_lora_warmup/<key>.pt` or `qwen3_4b_qlora_warmup/<key>.pt` beside the
processed Parquet. Each validation fold and the full-data build have distinct
identities. Checkpoint tags do not affect this cache. Submission inference
does not use it.

For cross-validation, the data key hashes the sorted original training ID set
and validation ID set separately. Reassigning training IDs among the other
folds does not invalidate the current split. Builds hash only the full original
ID set, independently of fold assignments, and use a separate build marker.
Fold numbers are excluded from this data identity; the effective training seed
is checked separately. File paths, Parquet bytes, reference order, text, and
labels are excluded. Consequently, changing text or labels while retaining
the selected ID sets deliberately reuses the old warmup; use `--refresh-cache`
to retrain it. Input validation and token-length checks still run for the
current data.

Other keys identify the model and exact Hub revision or local weight/configuration
content, backbone/head dimensions, tokenizer content, serialization version,
quantization mode, head dropout, A/B objective, warmup epochs/training batch
size/learning rate, effective seed, weight decay, gradient clipping, and cache
schema. Local model paths are excluded when weight/configuration contents
match. Compute precision, attention implementation, TF32, warmup extraction
batch size, and all LoRA rank/alpha/dropout/target-module settings are excluded,
as are joint-training learning rates, epoch count, batch size, accumulation,
and token budget. Context limits do not enter the key because adapter inputs are never
truncated and their lengths are checked independently. LoRA and QLoRA caches
remain separate.

After the last warmup validation (or the last warmup training epoch for builds),
the cache atomically stores FP32 classifier weights, named head AdamW moments
and step counters, gradient-scaler state, source precision, and Python, NumPy,
Torch CPU, and selected-device CUDA random states. It contains no adapters or
features. Reuse restores these states into a freshly initialized model, skips
all warmup extraction/training/validation, and starts the current joint schedule
with fresh adapters and the configured joint learning rates. Only FP16-to-FP16
reuse restores the saved gradient scaler; other transitions initialize it for
the current precision. Reuse across precision, extraction, or LoRA setting
changes continues from the original warmup result rather than reproducing a
fresh warmup under the new settings. Rank/target-module changes can affect
classification-head initialization through random-number consumption on a
cache miss; cached heads and named optimizer states remain reusable.

Schema v2 uses selected-ID identities. Legacy v1 warmup files use full fold
assignment hashes and remain untouched; they are not read under this identity.

Reads use CPU tensor loading and validate metadata, exact head/moment shapes,
FP32 dtypes, finite weights/moments, and restorable scaler/random states before
changing the model. Missing, corrupt, and incompatible entries are rebuilt;
write failures are reported as model execution errors. Logs show rebuild/reuse,
fold/build, path, and precision. Skipped warmup contributes no progress work,
epoch history, or historical timing/validation metrics; joint epoch numbers
retain their configured warmup offset. Build epoch counts report only epochs
executed in the current run.

`warmup_epochs: 0` disables warmup cache reads and writes. `--refresh-cache`
forces warmup again and atomically replaces the requested entries. Manually
constructed `PreparedTrainingSplit` objects without the optional
`warmup_context` retain uncached training; repository-prepared splits supply
that context automatically.

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

To prepare QLoRA batches, pass its profile explicitly; omitting `profile`
continues to select LoRA:

```python
from llm_classification_finetuning.models.qwen3_4b_qlora import PROFILE

batches = LoRATrainingBatchRepository.from_config(config, profile=PROFILE)
```

See [LoRA training](lora.md#optional-token-budget-training-batches) for planning
behavior and effective-batch semantics. The source modules that implement these
APIs and their development invariants are mapped in [structure.md](../structure.md).
