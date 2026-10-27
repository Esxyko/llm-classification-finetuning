# Caching and Python APIs

[Documentation index](../README.md)

Training caches live under `CACHE_PATH`, defaulting to `./cache`. Downloaded
model files use optional `MODEL_PATH`. Both follow the
[working-directory path rules](configuration.md#configuration-files-and-paths).
Production checkpoints under `models/` are separate artifacts.

## Cache overview

| Artifact under `CACHE_PATH` | Reuse depends on | Important exclusions |
| --- | --- | --- |
| `token_cache/` | Text, tokenizer content/implementation, encoding options, library/schema versions | IDs, labels, split, source path, model weights, training settings |
| `embedding_cache/` (historical frozen model) | Effective branch token IDs, exact backbone identity, hidden size, precision, pooling/storage/extraction version | IDs, labels, split, attention implementation, TF32, extraction budget |
| `adapter_batches/plans/` | Selected sample identities/orientations and A/B lengths, checkpointing threshold, update-group size, algorithm version | Labels, source order, token content beyond lengths, second-pass budget |
| `SR2_Qwen3_4B_lora_warmup/` | Original training ID set and effective SR2 warmup identity | Text/labels, validation IDs, precision, joint-training settings |

Retired profiles' caches, including `qwen3_4b_lora_EXPT2_warmup/`, remain
historical artifacts. SR2 and standalone warmup do not read that namespace.

Cache compatibility and production-checkpoint compatibility are independent.
For example, reusable tokens do not establish that a head or warmup state matches.
`--refresh-cache` refreshes requested entries and mode-specific caches, not Hub
base weights or every artifact in the cache root.

LoRA features are temporary CPU tensors: warmup reuses them within that stage;
each joint-validation/submission call extracts fresh ones with the current
model and releases them on return. Live LoRA embeddings are never persisted.

## Hugging Face model downloads

When `MODEL_PATH` is nonempty, project config/tokenizer/backbone loaders pass
it as `cache_dir`. `Skywork/Skywork-Reward-V2-Qwen3-4B`, for example, uses
`<MODEL_PATH>/models--Skywork--Skywork-Reward-V2-Qwen3-4B/` with native refs, snapshots, and blobs.
Hugging Face manages revisions, locks, and resumable downloads. Files persist
across processes without archive packing or unpacking.

Missing/blank `MODEL_PATH` preserves native Hugging Face cache settings.
Explicit local model directories load directly. `HF_HOME` and authentication
settings are preserved. Model downloads are outside the multipart policy below;
see [endpoint defaults](getting-started.md#hugging-face-downloads).

## Shared storage and multipart files

The four training-cache types and processed training Parquet use `CacheStorage`.
The logical `.sqlite3`, `.pt`, or `.parquet` path stays unchanged in configuration
and APIs regardless of its physical storage:

| Serialized size | Published layout |
| --- | --- |
| Below 100 MiB (104,857,600 bytes) | One file at the logical path |
| At least 100 MiB | `<name>.manifest.json` and `<name>.parts/`, each volume at most 99 MiB (103,809,024 bytes) |

The rule is fixed and does not compress the serialized payload. It excludes
production checkpoints and native HF downloads. Processed Parquet itself is
Zstandard-compressed before this storage rule applies.

Manifest v1 records total bytes/SHA-256 and ordered volume names, lengths, and
SHA-256 checksums. Volumes are named `000001-<sha256>.part`, etc. Keep the manifest
and **all referenced volumes together** when committing, moving, or copying.
Keep YAML pointing to the logical `.parquet`, not the manifest. The saved split
and selected cache directories use regular Git storage; `.gitattributes`
preserves binary volume bytes.

Reads reconstruct/checksum large files on disk under `.cache-work/` beside the
logical artifact, not as one in-memory byte string. Session subdirectories are
removed on exit; the ignored working directory and locks can remain. Allow space
for a reconstructed file plus staged volumes during updates, including on Drive
when `CACHE_PATH` points there.

Windows/Linux sessions use a per-artifact process lock with a 30-second timeout.
Updates work on private copies. Small files publish by atomic replacement;
multipart writes publish volumes before replacing the authoritative manifest.
Unchanged hits do not republish. A failed publication keeps the previous valid
artifact authoritative. A successful later write cleans unreferenced volumes
left by interrupted publication or cleanup.

Missing volumes, bad metadata, or checksum failures archive the artifact before
the cache-specific miss/rebuild policy runs. Corrupt processed data requires
preprocessing again; it is not silently regenerated from raw data by training.
Legacy single-file artifacts remain readable. See [migration](migration.md) for
old cache locations and formats.

## Shared token encodings

Each SQLite shard fingerprints actual tokenizer encoding content, implementation
and library versions, encoding options, and schema. Individual entries use a
SHA-256 of UTF-8 text. Duplicate texts and overlapping datasets share entries;
only missing or changed texts are encoded. A tokenizer is still loaded to
fingerprint it, even on complete hits.

SR2 renders complete conversations with its tokenizer's official chat template
and encodes them with `add_special_tokens=False`. The resulting text hashes and
encoding options distinguish these entries from historical plain-text inputs.
The serializer marker `sr2-qwen3-chat-template-v1` also isolates head-warmup
state and production checkpoints from the previous format. Old shards and
warmup files remain historical artifacts; no global cache deletion is needed.

The common representation is the full untruncated token sequence in compact
int32 CPU views. Retained historical frozen extraction separately caches exact
truncated outputs for overflowing texts, preserving special-token behavior. SR2 checks
full lengths against `max_length` and rejects empty or oversized inputs. Padding
happens after lookup.

Reads validate count, byte length, and checksum; newly encoded IDs are checked
before storage. Invalid entries are recomputed, writes are transactional, and
corrupt SQLite shards are archived and rebuilt.

## Frozen embeddings

This section describes retained infrastructure for the retired frozen Qwen
profile. SR2 uses temporary features and does not read this persistent cache.

Each SQLite shard identifies the exact Hub revision or local weight/configuration
content, backbone hidden size, compute precision, pooling, storage format, and
extraction version. Entries identify a branch's effective token IDs after any
truncation. Different computation identities coexist. Local identities require
reading weight files to hash content.

Labels, splits, IDs, row order, and dataset encoding do not invalidate vectors.
Changing the serializer or length limit still reuses a vector if the effective
token IDs are identical. Returned features follow current canonical IDs/order.
Complete hits skip backbone-weight loading, though hardware, config, and tokenizer
checks still run.

Pooled vectors are stored as FP16 CPU tensors and consumed by the frozen head as
FP32. Missing branches use the retained best-fit extraction sampler
and are scattered back to their requested positions. Batch composition, attention,
and TF32 can affect rounding, even though the latter two are outside the key.

Reads check blob length/checksum; fresh vectors are checked for finite values.
Invalid entries are recomputed and corrupt shards archived. Old dataset-wide
embedding files and shards using retired identities are not read. See
[historical frozen checkpoint compatibility](migration.md#historical-frozen-model).

## LoRA batch plans

Plans under `adapter_batches/plans/*.pt` use schema v4. Keys contain sorted
selected `(id, is_swapped, A_length, B_length)` rows, checkpointing threshold,
update-group size, and planning version. Cached identities map back to current
dataset indices and labels on each read.

Changing selected rows, lengths, threshold, or effective update size selects a
new plan. Changing labels, reference order, split names, unselected validation
inputs, or only `gradient_checkpointing_budget` does not. With threshold `0`,
there is no persistent global plan; full tokens are still cached.

Only base groups are persisted. Epoch shuffling, short-tail merging, and the
second pass over checkpointed singletons happen in memory. The second-pass
budget can therefore change while reusing the base plan. Logged plan
micro-batch counts describe the base plan; update counts follow finalized merged
groups. See [joint batching](lora.md#optional-token-budget-training-batches).

## LoRA head warmup

SR2 and standalone warmup write `SR2_Qwen3_4B_lora_warmup/<key>.pt` using
Skywork features in the shipped presets. Both use the
`swap_equivariant_tie_from_wins_v2` head architecture and warmup schema v5.
The v2 head adds zero-initialized, trainable win/tie readout biases. Its architecture
marker selects new cache keys within the existing namespace; v1 entries remain
on disk and are not reused or migrated. Warmup is regenerated for v2.
Submission does not use warmup caches.

### Identity

| Included in compatibility | Excluded from compatibility |
| --- | --- |
| Sorted original training ID set | Validation IDs, holdout/build label, source path/Parquet bytes, row order, training text and labels |
| Exact Hub model/revision or local weight/configuration content, hidden dimensions, tokenizer content, serializer, head architecture, quantization, schema | Local path spelling when contents match |
| Head dropout, A/B objective, warmup epochs/batch size/rate, seed, weight decay, clipping, nonzero warmup label smoothing | Compute precision, attention, TF32, extraction budget, context ceiling |
| Non-quantized training identity for active profiles | LoRA rank/alpha/dropout/targets; joint learning rates, epochs, batch size, accumulation, and token controls |

**Editing training text or labels while keeping the same IDs can reuse stale
warmup state. Use `--refresh-cache` when that state must be retrained.** Tokens
and input-length checks still follow current content. Changing only validation
IDs preserves reuse because validation neither updates the head nor selects an
early-stopping checkpoint.

Normal holdout and standalone warmup use train-partition IDs; full-data builds
use all original IDs. These normally select different cache keys. Checkpoint tags
do not enter the identity. Context limits are excluded because inputs are never
truncated and are independently validated.

### Saved state and restoration

After the final warmup validation, or final warmup training epoch for builds,
the cache atomically stores:

- FP32 classifier weights and named head AdamW moments/step counters.
- Gradient-scaler state and source precision.
- Python, NumPy, Torch CPU, and selected-device CUDA random states.

It contains no adapters or features. CPU loading validates metadata, exact
head/moment shapes, FP32 dtypes, finite values, and restorable scaler/RNG states
before changing the model. Missing/corrupt/incompatible states rebuild; write
failures are model execution errors. Retired-profile heads and earlier schemas
are not reused.

A hit restores the head/optimizer/RNG state, skips all warmup feature extraction,
training, and validation, and starts joint training with fresh adapters and fixed
joint rates. Only FP16-to-FP16 reuse restores the gradient scaler; other precision
transitions initialize a scaler for the current mode. Across changed precision
or LoRA settings, reuse continues the original cached warmup rather than
reproducing a fresh warmup under the new settings. Adapter initialization can
consume different RNG draws on a cache miss.

Logs identify reuse/rebuild, source path, and precision. Reused work contributes
no old epoch metrics or timing; joint epoch numbers retain their warmup offset.
`warmup_epochs: 0` disables reads/writes. `--refresh-cache` forces replacement.
Manually constructed `PreparedTrainingSplit` objects without `warmup_context`
remain uncached; repository-prepared splits supply it.

### Standalone warmup build

`model warmup --build` prepares train-only tokens and checks the same
non-quantized SR2 identity. A hit returns before backbone-weight loading and
reports zero executed epochs. A miss extracts a frozen backbone without adapters
and trains only the head. There is no validation, joint plan, production
checkpoint, or result report.

Reuse in inline SR2 requires matching both training IDs and effective model/head
settings, not merely using the same cache directory. See
[standalone configuration](configuration.md#standalone-warmup). Fresh standalone
and inline runs can differ because standalone skips adapter-initialization RNG
draws. A train-partition standalone cache can prewarm holdout validation; the
full-data build uses a different set of training IDs and needs its own entry.

## Python APIs

Import public exports rather than private helpers. The examples below run from
the project root; token/loader preparation may download tokenizer/config files
but does not load backbone weights or initialize CUDA.

### Text encodings

```python
from pathlib import Path

from llm_classification_finetuning.config import AppConfig
from llm_classification_finetuning.models.common import TokenizationCache

config = AppConfig.load(Path("config.yaml"))
cache = TokenizationCache.from_pretrained(
    config.cache_dir / "token_cache", "Skywork/Skywork-Reward-V2-Qwen3-4B"
)
encoded = cache.encode(["first input", "second input"])
first_ids = encoded[0]  # Unpadded int32 IDs, in requested input order.
```

An already loaded tokenizer can be passed to `TokenizationCache(cache_dir, tokenizer)`.
For model inputs, construct `ConversationSerializer(cache.tokenizer)` and pass it
as the required `serializer=` argument of `ModelDataRepository` or
`ModelTestDataRepository`. Encode the returned A/B texts with
`cache.encode(texts, add_special_tokens=False)`. `PairwiseTokenStore` sets this
option automatically. Formatting and encoding must use the same tokenizer.

Submission loads the checkpoint compatibility and saved base revision before
creating its tokenizer and test serializer. `LoRATrainer.prepare_tokens` can
prepare and validate those inputs with the same token cache on CPU before
backbone weights are loaded.

Model registration and SR2 pipeline/trainer exports are loaded on demand, so
importing CPU data/batch preparation does not import PEFT or initialize CUDA
through bitsandbytes. Requesting an execution class or the model registry loads
the normal training dependencies.

### CPU training-loader preparation

```python
from pathlib import Path

from llm_classification_finetuning.config import AppConfig
from llm_classification_finetuning.models.SR2_Qwen3_4B_lora import (
    LoRATrainingBatchRepository,
)

config = AppConfig.load(Path("config.yaml"))
batches = LoRATrainingBatchRepository.from_config(config)
training = batches.get_training()
training.set_epoch(0)
loader = training.loader
```

`get_training(full_data=True)` selects all original IDs for a build. The default
profile for this Python API is SR2. The loader yields `PairwiseBatch`; shared
tokens are available through `training.tokens`.

### Storage sessions

```python
from pathlib import Path

import torch

from llm_classification_finetuning.models.common import CacheStorage

storage = CacheStorage(Path("cache/example.pt"))
with storage.open("w") as session:
    torch.save({"example": 1}, session.path)
with storage.open("r") as session:
    payload = torch.load(session.path, map_location="cpu", weights_only=True)
```

This example uses an explicit path. Use `AppConfig.cache_dir` for the configured
root in application code. `open("w")` publishes on successful exit;
`open("rw")` publishes only after `session.mark_dirty()`; `open("r")` never
publishes. Close SQLite connections and other handles before leaving the session.

`exists()` checks logical presence; `archive()` quarantines an artifact, and
`session.archive()` does so under an existing lock. Missing reads raise
`FileNotFoundError`; malformed multipart files raise `CacheCorruptionError`
after archival. Each caller owns its recovery policy.
