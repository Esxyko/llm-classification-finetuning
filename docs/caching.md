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
| `embedding_cache/` (frozen SR2, SR2 warmup, and historical frozen model) | Effective branch token IDs, exact backbone identity, hidden size, precision, pooling/storage/extraction version | IDs, labels, split, head/training settings, attention implementation, TF32, extraction budget |
| `adapter_batches/plans/` | Selected sample identities/orientations and A/B lengths, checkpointing threshold, update-group size, algorithm version | Labels, source order, token content beyond lengths, second-pass budget |
| `SR2_Qwen3_4B_lora_warmup/` | Original training ID set and effective SR2 warmup identity | Text/labels, validation IDs, precision, joint-training settings |

Retired profiles' caches, including `qwen3_4b_lora_EXPT2_warmup/`, remain
historical artifacts. SR2 and standalone warmup do not read that namespace.

Cache compatibility and production-checkpoint compatibility are independent.
For example, reusable tokens do not establish that a head or warmup state matches.
`--refresh-cache` refreshes requested entries and mode-specific caches, not Hub
base weights or every artifact in the cache root.

Adapter-free SR2 warmup features are persisted across runs and configurations;
their in-memory views are shared within the warmup stage. Each joint-validation
call extracts fresh temporary features with the current adapters
and releases them on return. Live LoRA embeddings are never persisted.

## Hugging Face model downloads

When `MODEL_PATH` is nonempty, project config/tokenizer/backbone loaders pass
it as `cache_dir`. `Skywork/Skywork-Reward-V2-Qwen3-4B`, for example, uses
`<MODEL_PATH>/models--Skywork--Skywork-Reward-V2-Qwen3-4B/` with native refs, snapshots, and blobs.
Hugging Face manages revisions, locks, and resumable downloads. Files persist
across processes without archive packing or unpacking.

The runtime `.env` and Colab validation/build notebooks enable `HF_HUB_OFFLINE=1`.
Project loaders then pass `local_files_only=True`, reading the existing config,
tokenizer, and required weights without contacting the Hub. Populate a complete
native cache first, including refs, snapshot files, blobs, and every weight shard;
the separate `download.ipynb` explicitly uses `HF_HUB_OFFLINE=0`. Missing or
unreadable required files report the model, revision, resolved cache/local path,
and original error. Downloads must run separately with offline mode disabled.

Backbone weight loading prints start/end messages with elapsed time and hides
Hugging Face progress bars and Transformers `LOAD REPORT` logs. Other warnings
and loading exceptions retain their normal behavior.

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
profile. SR2's adapter-free [warmup features](#warmup-features) share the storage
component and directory, using a separate computation identity.

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
FP32. Missing branches use the shared extraction sampler, comparing descending
contiguous DP cuts with best-fit by micro-batch count, then total padded branch
tokens; exact ties retain best-fit. Features are scattered back to their requested
positions. Extraction plans stay in memory and this packing choice does not
change cache identities. Batch composition, attention, and TF32 can affect
rounding, even though the latter two are outside the key. See
[extraction packing](configuration.md#extraction-token-budgets).

Reads check blob length/checksum and finite values; fresh vectors are also checked
for finite values. Invalid entries are recomputed and corrupt shards archived. Old dataset-wide
embedding files and shards using retired identities are not read. See
[historical frozen checkpoint compatibility](migration.md#historical-frozen-model).

## Warmup features

Frozen SR2, standalone warmup, and inline non-quantized SR2 warmup share per-branch features
under `embedding_cache/<key>.sqlite3`. `WarmupFeatureCache` uses the shared
`EmbeddingCache` and `CacheStorage`, including locks, checksums, atomic publication,
and multipart storage. A first run extracts each missing unique branch once;
later runs load hits and extract only missing branches. Identical branch tokens
share one vector, including across train/validation partitions and full-data builds.

Each shard identifies the exact Hub revision or local weight/configuration
fingerprint, backbone hidden size, compute precision, storage dtype, and the
`last-real-token-right-padding-no-mask-v1` pooling policy. Warmup feature schema
and extraction versions both start at v1, independently of the head warmup schema.
These fields isolate warmup features from historical masked frozen extraction.
Entries hash complete, untruncated token IDs in canonical A/B orientation.
Changing IDs, labels, split, row order, head width/dropout, optimization settings,
seed, or extraction budget does not invalidate compatible branch vectors.
Attention implementation and TF32 are excluded, as in retained frozen caching;
batch composition and these settings can still affect numerical rounding.

Stored vectors preserve the model's feature dtype: BF16 compute stores BF16,
while FP16 or FP32 compute stores FP32 pooled features. BF16 is serialized as its
raw 16-bit representation without an FP16 conversion; the shared component also
retains the historical FP16 storage format and default cache keys. Features are
rebound to current IDs, labels, and A/B orientation when loaders are constructed.

Logs print reused/missing unique branch counts and the logical cache path.
Frozen SR2 and standalone head-only warmup create their classifier from model
configuration, then check features before loading any backbone weights. Complete
feature hits train/evaluate only the head and create no extraction progress stage.
Partial hits load the frozen backbone lazily and extract only missing unique
branches, checking its revision and hidden size against the prepared identity.
Config/tokenizer, local-weight fingerprint, hardware, and input validation still
run. Head parameters use their normal seeded random initialization; deferred
weight loading saves/restores CPU/CUDA RNG state so cache misses do not advance
the head's random stream. Existing cache identities and schemas remain unchanged.
Inline LoRA warmup retains its eagerly initialized adapter-bearing model for the
following joint stage. Cache I/O and publication remain outside extraction timing.
The shuffled head-training loader and
fixed-order train/validation loaders share the resulting CPU tensor, whose views
are released before joint training.

`--refresh-cache` recomputes and replaces requested branch entries, preserving
unrequested entries. Invalid vectors become misses; corrupt storage follows the
shared archive/rebuild policy, while storage access failures remain explicit
execution errors. Splits without `warmup_context` and retained quantized profiles
continue using temporary extraction without this feature cache. Head state,
losses, and current-adapter joint-validation features are separate.

## LoRA batch plans

Plans under `adapter_batches/plans/*.pt` use schema v4 and planning algorithm
v3. Keys contain sorted selected `(id, is_swapped, A_length, B_length)` rows,
checkpointing threshold, update-group size, and planning version. Cached
identities map back to current dataset indices and labels on each read.

Algorithm v3 constrains packing by total padded A/B tokens
`P = rows * (longest_A + longest_B)`. It compares three global sorts and ranks
plans by micro-batch count, then total `P`. Within each group, a suffix DP finds
the optimal contiguous partition of threshold-eligible rows in that global
sort's order; oversized rows stay alone. It is compared with the retained
descending-A+B best-fit candidate using the same objective, retaining best-fit
on exact ties. Best-fit placements minimize remaining budget, then micro-batch
creation order; exact global-plan ties retain the first sort. The stored
`padded_tokens` is the sum of `P` across all micro-batches; validation uses `P`
for both the threshold and this total. Historical best-fit v1 and single-branch
v2 plans have separate cache keys and are not reused or migrated; their files
remain on disk. Matching v3 plans are reused after validation.

Changing selected rows, lengths, threshold, or effective update size selects a
new plan. Changing labels, reference order, split names, unselected validation
inputs, or only `gradient_checkpointing_budget` does not. With threshold `0`,
there is no persistent global plan; full tokens are still cached.

Only base groups are persisted. Epoch shuffling, short-tail merging, and the
second pass over checkpointed singletons happen in memory. The second pass
compares DP cuts in descending A+B order with best-fit under the second-pass
budget, within each finalized update. That budget can therefore change while
reusing the base plan. Logged plan micro-batch counts describe the base plan;
update counts follow finalized merged groups. See
[joint batching](lora.md#optional-token-budget-training-batches).

## Frozen SR2 head training

`SR2_Qwen3_4B` uses the same token, adapter-free feature, and completed head
state caches as standalone/inline LoRA warmup. Its independent `epochs`,
`training_batch_size`, `learning_rate`, and `lr_decay_base` map to
the shared internal `warmup_epochs`, `warmup_training_batch_size`,
`warmup_learning_rate`, and `warmup_lr_decay_base` settings. The public `warmup`
section uses the same unprefixed names as frozen SR2; this naming change leaves
cache identities unchanged. Matching curves share head state without
changing the cache namespace or schema. Other compatibility rules below,
including training ID sets and the treatment of changed text/labels, still apply.

Exact head hits evaluate current train/validation data for validation runs, or
all training data for builds, without optimizing or rewriting the cached state.
Fixed-rate partial hits (`lr_decay_base: 1`) continue only the remaining epochs
with the existing shuffle/RNG restoration. Exponential decay requires an exact
target epoch count. Validation publishes a new report even when no epochs run; build
saves its independent `head[_TAG].pt` artifact even when the head is fully cached.
`--refresh-cache` refreshes requested token, feature, and head entries.

## LoRA head warmup

Frozen SR2, LoRA SR2, and standalone warmup write `SR2_Qwen3_4B_lora_warmup/<key>.pt` using
Skywork features in the shipped presets. All use the
`swap_equivariant_harmonic_discrepancy_v3` head architecture and warmup schema v5.
This head packs L/M into one projection for the wins and separately projects
X/Y with the FP32 parameter difference L-M and a shared zero-initialized B3.
Undropped X/Y feed harmonic fusion and the discrepancy branch. The correction q
is computed once for outputs `[r_A + q, r_B + q, s - q + b]`, with one scalar
output bias. The architecture marker selects new cache keys within the existing
namespace; harmonic/discrepancy v1/v2 and earlier tie-from-wins v1/v2 entries
remain on disk and are not reused or migrated. Exact head/optimizer tensor names
and shapes additionally reject earlier structures without the fused L/M layout
and B3, even if their metadata is relabeled.
Warmup is regenerated for the new architecture.

SR2 also includes `head_initialization_version` in its warmup metadata, currently
`scaled_uniform_zero_discrepancy_v1`. The full metadata is hashed with SHA-256
to form the cache key and must match the selected source exactly when loading. Earlier
Xavier-initialized SR2 entries lack this field and therefore select different
keys; they remain on disk without reuse or migration. Initialization versioning
does not change the head architecture, warmup schema v5, production-checkpoint
compatibility, or other heads' cache identities. V3 retains this initialization
version, including zero B3; its architecture marker independently isolates the
new head. Compatible v3 entries remain shared by standalone and inline SR2 warmup
when all identity fields match. Token, adapter-free feature, and batch-plan
caches retain their own reuse rules because their representations are unchanged.

With exponential decay (`warmup.lr_decay_base > 1`), lookup checks only the exact
target `N`-epoch key. Metadata additionally includes
`warmup_lr_schedule: exponential_epoch_v1` and `warmup_lr_decay_base`, retaining
the original base rather than its logarithm. The existing `warmup_epochs` field
identifies the completed target. A different base or target selects a different
key and trains from epoch index `0` on a miss. Lower-epoch exponential entries
are not continuation sources. Earlier polynomial/cosine entries and old
fixed-rate entries are not reused for exponential decay; they remain on disk
without migration.

With `warmup.lr_decay_base: 1`, both extra fields are omitted, preserving old
fixed-rate identities. Lookup first checks the exact `N`-epoch key,
then checks `N-1` down to `1`, changing only `warmup_epochs` in the metadata.
The first fully validated state is selected. Missing, corrupt, or incompatible
candidates are skipped; higher-epoch states are never used for a smaller target.
Lookup computes existing keys directly and supports both single-file and multipart
caches. Schema v5 and the saved-state layout remain unchanged; existing
fixed-rate entries need no migration and remain on disk.

### Identity

| Included in compatibility | Excluded from compatibility |
| --- | --- |
| Sorted original training ID set | Validation IDs, holdout/build label, source path/Parquet bytes, row order, training text and labels |
| Exact Hub model/revision or local weight/configuration content, hidden dimensions, tokenizer content, serializer, head architecture, SR2 initialization version, quantization, schema | Local path spelling when contents match |
| Head dropout, A/B objective, warmup epochs/batch size/rate, exponential schedule version and original decay base when decay is enabled, seed, weight decay, clipping, nonzero warmup label smoothing | Compute precision, attention, TF32, extraction budget, context ceiling |
| Non-quantized training identity for active profiles | LoRA rank/alpha/dropout/targets; joint learning rates, epochs, batch size, accumulation, and token controls |

Warmup epochs remain part of each stored key. A lower-epoch entry is compatible
as a continuation source only for fixed-rate training and when every other
identity field matches.

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

After the final warmup epoch's training-set evaluation and any validation
evaluation both succeed, the cache atomically stores:

- FP32 classifier weights and named head AdamW moments/step counters.
- Gradient-scaler state and source precision.
- Python, NumPy, Torch CPU, and selected-device CUDA random states.

It contains no adapters, features, or loss metrics. CPU loading validates metadata,
exact head/moment shapes, FP32 dtypes, finite values, and restorable scaler/RNG states
before changing the model. If no candidate is valid, warmup rebuilds; write
failures are model execution errors. Retired-profile heads and earlier schemas
are not reused.

A complete hit in inline LoRA restores the head/optimizer/RNG state, skips all warmup
feature extraction, training, and evaluation, and starts joint training with fresh
adapters and the configured joint schedule.

A fixed-rate partial hit at epoch `K < N` loads compatible frozen features and
extracts any missing branches, restores the head/optimizer/RNG state after
feature and loader preparation, and trains only epochs `K+1` through `N` at the
configured fixed warmup rate. The independent
training-loader generator is reconstructed by replaying `K` epochs through a
lightweight index-only DataLoader with the same length, batch size, shuffle, and
seed. Replay reads no features and executes no model work. This preserves the
existing shuffle sequence without adding state to schema v5. Completed source
epochs contribute no historical losses or timing. The remaining Head warmup
progress covers only `N-K` epochs, while epoch numbers retain their target offset.

After the target epoch's train and any validation evaluation succeed, the resumed
state is atomically saved under the target `N`-epoch key. The source cache is
preserved; a failed continuation does not publish the target. Only the configured
target is saved, not each intermediate epoch.

Only FP16-to-FP16 reuse restores the gradient scaler; other precision
transitions initialize a scaler for the current mode. Across changed precision
or LoRA settings, reuse continues the original cached warmup rather than
reproducing a fresh warmup under the new settings. Adapter initialization can
consume different RNG draws on a cache miss.

Logs identify complete reuse, continuation, or rebuild; completed, target, and
remaining epochs; source/target paths on hits; and precision. Reused work contributes
no old epoch metrics or timing; joint epoch numbers retain their warmup offset.
`warmup.epochs: 0` disables warmup state and feature cache reads/writes.
`--refresh-cache` bypasses every reuse candidate, refreshes requested warmup
features, trains from scratch, and replaces the target head entry.
Manually constructed `PreparedTrainingSplit` objects without `warmup_context`
remain uncached; repository-prepared splits supply it.

### Standalone warmup build

`model warmup --build` prepares tokens for train and validation and checks the
same non-quantized SR2 identity, still based only on training IDs. A head miss loads
compatible frozen features and extracts missing branches without adapters for
both partitions, then trains only the head on original train rows. Each epoch
ends with full-training-set inference
using warmup label smoothing, followed by unsmoothed validation inference;
both use final epoch weights and disable dropout.

A complete head hit loads the backbone, restores the cached head, and loads
compatible train/validation features, extracting only missing branches, to
recompute both losses. It executes zero training epochs and does not rewrite the
cached head state. A partial hit follows the continuation
flow above, evaluates both partitions after each remaining epoch, and saves the
target cache. Its summary displays `Resumed`, returns the target cache path, and
counts only epochs executed in this invocation. Complete hits display `Reused`;
fresh builds display `Created`. The summary reports both losses on all paths;
no historical loss values are reused. This evaluation does
not change the existing cache identity or schema v5. There is no joint plan,
production checkpoint, or result report.

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
as the required `serializer=` argument of `ModelDataRepository`. Encode the
returned A/B texts with `cache.encode(texts, add_special_tokens=False)`. `PairwiseTokenStore` sets this
option automatically. Formatting and encoding must use the same tokenizer.

`LoRATrainer.prepare_tokens` prepares and validates complete inputs with the
same token cache on CPU; training and validation reuse this token storage.

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
