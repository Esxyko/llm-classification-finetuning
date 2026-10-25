# Caching and Python APIs

[Documentation index](../README.md)

Training caches sit beside the processed Parquet selected by
[`data.processed_path`](configuration.md#configuration-files-and-paths).
Relative processed paths resolve from the working directory independently of
`MODEL_PATH`, so generated training caches stay there too.
Frozen embeddings, shared text encodings, adapter batch plans, and head-warmup
states have separate compatibility rules. LoRA warmup, joint-validation,
and submission features stay in temporary CPU memory (BF16 in BF16 mode,
otherwise FP32) and are not persisted. Uncached warmup reuses its feature
pool; each joint-validation or submission call extracts fresh features with the
current model and releases them on return.

## Hugging Face model downloads

Downloaded base models, tokenizers, and model configurations use a persistent
native Hub cache when `MODEL_PATH` is non-empty in `.env` or the environment.
For example, `Qwen/Qwen3-4B` is stored directly at
`<MODEL_PATH>/models--Qwen--Qwen3-4B/`. Relative paths and `~` follow the
[configuration path rules](configuration.md#configuration-files-and-paths).
With no `MODEL_PATH`, Hugging Face keeps its usual cache configuration.

Project model/config/tokenizer loaders pass `MODEL_PATH` as `cache_dir`.
Model IDs, refs, snapshots, blobs, negative-cache markers, and commit identities
retain their native layout. Hugging Face manages download locks and resumable
downloads. Files remain on disk and are reused across processes without packing
or unpacking. Explicit local model directories load directly. `HF_HOME` and
authentication settings are preserved. These native model downloads are outside
the training-cache multipart policy below.

To migrate existing downloads, set `MODEL_PATH`, stop processes using the old
cache, and run from the project root:

```shell
uv run python scripts/migrate_hf_models.py --move --dry-run
uv run python scripts/migrate_hf_models.py --move
```

The source defaults to `HF_HUB_CACHE`, then legacy `HUGGINGFACE_HUB_CACHE`,
then `<HF_HOME>/hub`, or `~/.cache/huggingface/hub` with the normal
`XDG_CACHE_HOME` fallback. The script reads these settings without importing
project or HF modules. You can explicitly select the old **Hub cache directory**,
containing `models--*` directories:

```shell
uv run python scripts/migrate_hf_models.py --source "C:/Users/me/.cache/huggingface/hub" --move
```

Without `--move`, the script copies and retains the source. `--dry-run` prints
the plan without creating or deleting files. Each model repository is copied
into a temporary destination directory, checked file by file with SHA-256, and
published directly under `MODEL_PATH` only after verification. `--move` removes
that repository's source only after publication. Existing destination model
directories are rejected without overwriting them; source and destination paths
must not overlap. An error stops the script with exit code `1`; previously
completed repositories remain in the destination.

The script preserves `blobs`, `refs`, snapshots, and internal file symlinks;
absolute links are rebased so they do not depend on the old cache. On Windows
without symlink privileges, linked snapshots become regular files and may use
more disk space. Only native `models--*` repository directories are migrated.
Authentication tokens, datasets, spaces, download locks, and Xet caches are
excluded. Existing ZIP archives are not read by project loaders or this script.

## Shared storage and multipart files

All four cache types use the common `CacheStorage` file interface. Cache keys,
SQLite tables, and PyTorch payloads retain their existing formats. Logical paths
such as `<key>.sqlite3` and `<key>.pt` identify caches even when the corresponding
single file is replaced by volumes.

The [processed training Parquet](data-preparation.md#processed-data) uses this
same interface and size policy for pipeline output and input loading. Its
configured `.parquet` path remains the logical path; the default saved split's
manifest and volumes are tracked with regular Git storage.

Files smaller than 100 MiB (104,857,600 bytes) remain single files. Files at or
above that threshold are split into volumes of at most 99 MiB (103,809,024 bytes),
keeping every published file below GitHub's 100 MiB per-file limit. This rule is
fixed, without compression or additional configuration, and does not apply to
production model checkpoints. See [GitHub's file size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

A multipart cache uses `<original-name>.manifest.json` and an accompanying
`<original-name>.parts/` directory. Manifest version 1 records the complete byte
length and SHA-256, plus an ordered list of volume names, lengths, and SHA-256
checksums. Volumes are named `000001-<sha256>.part`, `000002-<sha256>.part`, and
so on. Commit the manifest and all referenced volumes together. Git preserves
volume bytes without line-ending conversion through `.gitattributes`.

Reads reconstruct and validate multipart files on disk rather than loading an
entire serialized file into memory. Each session uses a private `.cache-work/`
directory beside the logical cache, removed when the session ends; lock files
remain there for reuse. These working directories are ignored by Git and may
temporarily contain files larger than the publication limit. Allow disk space
for a reconstructed file and newly staged volumes when updating large caches.

Sessions hold a per-cache process lock on Windows and Linux, with a 30-second
lock timeout. Updates operate on private working copies. After serialization
and all file handles have closed, small files are atomically replaced; multipart
updates publish all volumes before atomically replacing the manifest. Unchanged
hits do not rewrite published files. Old volumes are cleaned after publication;
an interrupted write or unsuccessful cleanup can leave unreferenced volumes,
which the next successful write removes. A failed publication keeps the previous
valid cache authoritative.

Missing volumes, invalid metadata, and checksum mismatches archive the multipart
artifact before reporting a cache miss or rebuilding, according to the cache
type. SQLite corruption also uses the shared archive operation. Single-file
archives at or above the threshold are split rather than retained as oversized
files. Oversized corrupt manifest or volume components are themselves split
inside the archive, preserving their bytes for inspection through `CacheStorage`.
Legacy single-file caches remain readable and use the size rule on their
next write. `--refresh-cache` keeps its existing meaning.

The shared Python interface is exported from `models.common`:

```python
from pathlib import Path

import torch

from llm_classification_finetuning.models.common import CacheStorage

storage = CacheStorage(Path("data/processed/adapter_batches/example.pt"))
with storage.open("w") as session:
    torch.save({"example": 1}, session.path)
with storage.open("r") as session:
    payload = torch.load(session.path, map_location="cpu", weights_only=True)
```

`open("w")` publishes on successful exit; `open("rw")` copies an existing cache
or prepares a new path and publishes only after `session.mark_dirty()`.
`open("r")` never publishes. Close SQLite connections or other file handles
before leaving any session. `exists()` checks logical presence; `archive()`
quarantines an artifact, and `session.archive()` does so within an existing
session lock. Missing artifacts raise `FileNotFoundError` on reads; malformed
multipart artifacts raise `CacheCorruptionError` after archival. These storage
errors remain subject to each cache's existing recovery policy.

## Frozen embeddings

The frozen model shares content-addressed embeddings between training and test
under `embedding_cache/` beside the processed Parquet. Each SQLite shard
identifies the model and exact Hub revision (or local checkpoint/configuration
content), backbone hidden size, precision, pooling, storage format, and
extraction version. Attention implementation, TF32, and `extraction_budget` do
not affect cache reuse.
Individual entries identify the effective token IDs of one branch, after any
requested truncation. Different compute identities coexist without overwriting
one another.
Pooled vectors are stored as FP16 CPU tensors, including when backbone compute
precision is BF16 or FP32; the classifier consumes them as FP32.

Changing labels, splits, IDs, row order, or dataset file encoding does not
invalidate existing vectors. Overlapping datasets and A/B branches reuse each
other's entries; only missing inputs are encoded. Changing `max_length` or the
serializer also reuses vectors when the resulting token IDs remain identical.
Cache rows are mapped back to the current canonical IDs and order. When all
inputs hit, the backbone weights are not loaded. Local checkpoint identities
require reading the selected weight files to hash their content.

Missing branches are sorted by descending length and best-fit packed within
`extraction_budget` padded tokens per micro-batch, then independently right-padded.
Extracted vectors are scattered back to the original branch positions. Changing
batch composition, attention implementation, or TF32 can change floating-point
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
SHA-256. IDs, labels, splits, row order, dataset fingerprints, model names, model
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

LoRA uses shared `adapter_batches/plans/*.pt` beside the processed Parquet.
Plans identify the sorted selected `(id, is_swapped, A_length, B_length)` rows,
checkpointing threshold, update group size, and planning algorithm version.
Labels, split names, reference order,
text/token content beyond lengths, and unselected validation inputs do not enter
the plan key. Cached groups store sample identities and are mapped to current
dataset indices on every read; current labels are always used for training.
A/B mode and holdout/build selection determine membership before lookup. Changing
threshold or effective batch size rebuilds plans while reusing tokens. With
`gradient_checkpointing_threshold: 0`, tokens are cached but no global plan is
created, and all fixed joint-training batches use gradient checkpointing.
Plan schema version `4` stores `threshold` instead of the old `budget`
metadata; older plan versions are rebuilt. The obsolete checked-in plans have
been removed.
Legacy per-model batch directories and per-model `tokens/*.pt` files are no
longer read. Identical plan keys remain reusable across configurations and
training modes. Different thresholds or selected rows still select
different plans.

Cached packing groups keep their existing format and membership, including any
short final group. Each epoch shuffles the full groups, then appends the short
group's micro-batches to the last shuffled full group for a single merged update.
An epoch containing only a short group runs it alone. The optional
`gradient_checkpointing_budget` pass then combines checkpointed singletons
inside each final update group, before shuffling its micro-batches. That pass
uses the configured budget and current epoch's merged membership, and is never
written into the base plan. Changing only this budget therefore reuses the same
base cache. The merged joint-training update runs last after packing and
shuffling; cached group membership and warmup ordering remain unchanged.
A budget of `0` leaves the original micro-batches and their order within each
update unchanged. With a zero threshold and positive budget, fixed-size initial
batches are shuffled by `random_state + joint_epoch` and repacked in memory
without a global plan cache. The reported plan micro-batch count describes the
base plan; the update count reflects the merged optimizer groups.
LoRA shares the text-token cache across stages when tokenizer content and
encoding options match.

`--refresh-cache` replaces requested token entries and model-specific caches,
including head warmup. Shared training encodings remain reusable across stages
in that run. Warmup
features and live LoRA encodings are never persisted in these token caches.

## LoRA head warmup

Holdout validation and builds automatically save completed head warmup under
`qwen3_4b_lora_warmup/<key>.pt` beside the processed Parquet. The standalone
`model warmup --build` command also writes
the LoRA directory using only the saved train partition. Identity follows the
actual training data, rather than the holdout/build mode. Checkpoint tags do not
affect this cache. Submission inference
does not use it.

The experimental `qwen3-4b-lora-EXPT` profile stores its built-in warmup in
`qwen3_4b_lora_EXPT_warmup/<key>.pt`. Its identity includes
`head_architecture: swap_equivariant_reparam_v1` and the fixed original-only
training policy. Meta-device validation constructs the matching head and
checks its three reparameterized weight tensors and named optimizer states.
Standard MLP caches retain their existing keys and cannot be reused by EXPT.
The standalone warmup command continues to build the standard MLP cache.

The data key hashes only the sorted original training ID set. For normal model
builds this is the full original ID set; for holdout runs and standalone warmup
it is the saved training partition.
Validation IDs and split names do not enter the key: validation only measures
loss and does not update the head or choose an early-stopping checkpoint.
Changing only the validation set therefore retains a compatible warmup cache.
Holdout and build calls with the same training IDs and remaining settings can
reuse the same state. The configured model seed is checked separately.

File paths, Parquet bytes, reference order, text, and labels remain excluded.
Changing training text or labels while retaining IDs thus reuses the old warmup;
use `--refresh-cache` to retrain it. Input and token-length checks still run.

Other keys identify the model and exact Hub revision or local weight/configuration
content, backbone/head dimensions, `use_abs_diff`, tokenizer content,
serialization version, quantization mode, head dropout, A/B objective, warmup epochs/training batch
size/learning rate, effective seed, weight decay, gradient clipping, and cache
schema. Local model paths are excluded when weight/configuration contents
match. Compute precision, attention implementation, TF32, warmup extraction
token budget, and all LoRA rank/alpha/dropout/target-module settings are excluded,
as are joint-training learning rates, epoch count, batch size, accumulation,
and token budget. Context limits do not enter the key because adapter inputs are never
truncated and their lengths are checked independently. LoRA lookup requires
the `training_quantization: none` identity while strictly checking all other
compatibility fields. Historical quantized warmup caches are not reused by LoRA.

After the last warmup validation (or the last warmup training epoch for builds),
the cache atomically stores FP32 classifier weights, named head AdamW moments
and step counters, gradient-scaler state, source precision, and Python, NumPy,
Torch CPU, and selected-device CUDA random states. It contains no adapters or
features. Reuse restores these states into a freshly initialized model, skips
all warmup extraction/training/validation, and starts joint training with fresh
adapters and the configured fixed joint learning rates. Only FP16-to-FP16 reuse
restores the saved gradient scaler; other transitions initialize it for
the current precision. Reuse across precision, extraction, or LoRA setting
changes continues from the original warmup result rather than reproducing a
fresh warmup under the new settings. Rank/target-module changes can affect
classification-head initialization through random-number consumption on a
cache miss; cached heads and named optimizer states remain reusable.

Schema v5 supports the default `[A, B, |A - B|]` classifier with a `3H` input and
the optional `[A, B]` classifier with a `2H` input. Disabled heads include
`use_abs_diff: false` in their cache identity; enabled heads retain their existing
v5 cache paths. Standalone warmup and LoRA must select the same switch for reuse.
The schema retains training-ID identities without a validation-set binding and
merges a short final batch into the preceding full optimizer update. Legacy
v1/v2/v3/v4 warmup files are retained and no longer read.
Shared token encodings, frozen embeddings, and LoRA batch plans retain their
existing cache identities.

Reads use CPU tensor loading and validate metadata, exact head/moment shapes,
FP32 dtypes, finite weights/moments, and restorable scaler/random states before
changing the model. Missing, corrupt, and incompatible entries are rebuilt;
write failures are reported as model execution errors. Logs show rebuild/reuse,
holdout/build, path, and precision. Skipped warmup contributes no progress work,
epoch history, or historical timing/validation metrics; joint epoch numbers
retain their configured warmup offset. Build epoch counts report only epochs
executed in the current run.

`warmup_epochs: 0` disables warmup cache reads and writes. `--refresh-cache`
forces warmup again and atomically replaces the requested entries. Manually
constructed `PreparedTrainingSplit` objects without the optional
`warmup_context` retain uncached training; repository-prepared splits supply
that context automatically.

### Standalone warmup build

`uv run model warmup --build` uses the `warmup` section and the
shared data, validation, and GPU settings. It prepares tokens only for train
IDs, applies the configured `A/B_swap` objective, and checks the same schema-v3
non-quantized LoRA key. A valid cache returns before backbone-weight loading,
feature extraction, or training; the summary reports zero executed epochs.

On a miss it loads a frozen, non-quantized backbone without injecting adapters,
extracts each train ID once into temporary CPU features, and runs only the
configured head warmup. It saves the same head, named AdamW, scaler, and RNG
payload after the final training epoch. It performs no validation, creates no
joint batch plan, and publishes no production checkpoint or report. Because
there are no adapter-initialization RNG draws, a new standalone warmup may
differ from a fresh inline LoRA warmup; compatible cached results remain usable
by LoRA under the existing key rules.

`--refresh-cache` forces standalone warmup to replace its requested LoRA cache
and refresh train tokens. Reuse logs include the actual source path and
quantization. A train-partition cache does not match a normal
full-data build; `warmup.warmup_epochs: 0` disables all LoRA warmup cache I/O.
Warmup epochs, training batch size, and learning rate are shared through this
section; its other settings configure standalone warmup. The standalone command
requires positive warmup epochs.

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
training = batches.get_training()  # Saved training partition only.
# batches.get_training(full_data=True) selects all IDs for a build.
training.set_epoch(0)  # Zero-based joint epoch; fixed batching keeps its RNG stream.
loader = training.loader  # Yields PairwiseBatch; training.tokens is shared.
```

See [LoRA training](lora.md#optional-token-budget-training-batches) for planning
behavior and effective-batch semantics. The source modules that implement these
APIs and their development invariants are mapped in [structure.md](../structure.md).
