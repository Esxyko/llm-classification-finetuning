# Configuration

[Documentation index](../README.md)

Use [config.yaml](../config.yaml) or a complete file from [configs/](../configs/)
as a starting point. This reference describes parser rules; explicit preset
values live in YAML and may differ between profiles and files.

## Configuration files and paths

Every command accepts `--config PATH`, defaulting to `config.yaml`. Configurations
do **not** inherit or merge values from the root preset. The entire selected
file is validated before work begins, including model sections unused by the
command. Unknown keys are errors.

Relative paths resolve from the **process working directory**, not the YAML
file's directory. `~` expands to the user's home directory. `.env` is loaded
from that working directory without overriding existing environment variables.

Before importing PyTorch, package initialization defaults `PYTORCH_ALLOC_CONF`
to `expandable_segments:True` unless either `PYTORCH_ALLOC_CONF` or its legacy
alias `PYTORCH_CUDA_ALLOC_CONF` is already set in the environment or `.env`.
This process-wide allocator setting requires a fresh process to take effect.

| Setting or artifact | Location / fallback | Controls |
| --- | --- | --- |
| `--config PATH` | `./config.yaml` | Selected complete YAML file |
| `data.raw_dir` | Required YAML path; root preset `data/raw` | Raw competition CSVs |
| `data.processed_path` | Required YAML `.parquet` path; root preset `data/processed/train_validation.parquet` | Saved train/validation data, including multipart layout |
| `MODEL_PATH` | Optional environment or `.env`; missing/blank uses native HF settings | Downloaded model/config/tokenizer cache |
| `HF_HUB_OFFLINE` | Environment or `.env`; runtime template and Colab validation/build use `1` | Load required HF files locally without Hub requests; use `0` in a separate download process to fill missing files |
| `CACHE_PATH` | Environment or `.env`; missing/blank uses `./cache` | Tokens, frozen/warmup features, LoRA batch plans, and head-warmup states |
| `RESULT_PATH` | Environment or `.env`; missing/blank uses `./results` | Validation runs, submissions, comprehensive reports, and batch-run logs |
| Production checkpoints | `./models/SR2_Qwen3_4B/`, `./models/SR2_Qwen3_4B_lora/` | Frozen head or LoRA adapter/head files |

`RESULT_PATH` follows the same path rules as `CACHE_PATH`: surrounding
whitespace is stripped, `~` expands, and relative values resolve from the process
working directory. Validation runs are direct children of this root; submissions,
comprehensive reports, and batch logs use `submit/`, `comprehensive/`, and `logs/`.
The batch script loads `.env` and resolves relative paths from its project root,
matching its child commands even when launched from another directory.

Changing these settings selects storage for subsequent reads and writes; it does
not move existing files. YAML data paths and production checkpoint locations are
independent. `DATA_PATH` is unused.

For example, `uv run model SR2-Qwen3-4B-lora --config configs/2.yaml` still
resolves relative data and output paths from the project root when run there.
See [cache layout](caching.md) and [Colab persistence](colab.md).

## Configuration sections

| Section | Required when | Responsibility |
| --- | --- | --- |
| `data` | Always | Competition and data paths |
| `validation` | Always | Saved holdout fraction, seed, and balance tokenizer |
| `gpu` | Always | Training device, submission devices, precision, and attention |
| `SR2_Qwen3_4B` | Selecting `SR2-Qwen3-4B` | Frozen Skywork and independent SR2 head training |
| `SR2_Qwen3_4B_lora` | Selecting `SR2-Qwen3-4B-lora` | Skywork LoRA and swap-equivariant comparison head |
| `warmup` | LoRA section is present, or selecting `warmup --build` | Shared warmup controls and standalone warmup model |

Model sections may be absent for data/reporting or standalone warmup;
`model ALL` requires at least one configured model supporting the selected mode.
`warmup` is not part of `ALL`. Frozen SR2 is independent of both other sections.
SR2 LoRA inherits only the five controls listed under
[standalone warmup](#standalone-warmup) from `warmup`. Retired Qwen model
sections are unknown keys; see [migration](migration.md#retired-profiles).

## Required fields and defaults

All fields below are required unless a default is shown. Numerical values must
be finite, integers must be YAML integers, and Boolean fields must be actual
Booleans. Unquoted strings such as `bf16`, `sdpa`, and `cuda:0` are valid.

### Data and validation

| Field | Rule / parser default |
| --- | --- |
| `data.competition` | Nonempty string |
| `data.raw_dir` | Nonempty path |
| `data.processed_path` | Nonempty path ending in `.parquet` |
| `validation.random_state` | Integer |
| `validation.validation_fraction` | Number in `(0, 1)`; default `0.2` |
| `validation.tokenizer_name` | Hub ID or local tokenizer directory with a chat template; default `Skywork/Skywork-Reward-V2-Qwen3-4B` |
| `validation.tokenizer_revision` | Nonempty revision string or `null`; default `null` |
| `validation.max_token_deviation` | Number in `(0, 1)`; default `0.005` |

The tokenizer is used for preparation independently of optional model sections.
The shipped presets select the SR2 tokenizer and pin revision
`fd958fef475f323f4e6b195930e3dd918485c668` for reproducible chat-template token
counts. If you change the training tokenizer and rebuild the split, update this
tokenizer too. Preparation renders the complete conversation before counting,
without adding a second set of special tokens.

### GPU fields

| Field | Rule / parser default |
| --- | --- |
| `gpu.device` | One `cuda:N` device |
| `gpu.submit_devices` | List of exactly two distinct `cuda:N` devices; default `[cuda:0, cuda:1]` |
| `gpu.precision` | `fp16`, `bf16`, or `fp32` |
| `gpu.attention_implementation` | `sdpa` or `eager` |
| `gpu.allow_tf32` | Boolean |

### Model fields

The following names apply independently inside each relevant section.

| Fields | Sections | Rule / parser default |
| --- | --- | --- |
| `model_name` | All model sections and `warmup` | Nonempty Hub ID or local model directory |
| `max_length`, `hidden_size`, `extraction_budget` | All model sections and `warmup` | Positive integers; `hidden_size` is the classifier width |
| `dropout` | All model sections and `warmup` | Number in `[0, 1)` |
| `weight_decay` | All model sections and `warmup` | Number at least `0` |
| `random_state` | All model sections and `warmup` | Integer |
| `dataloader_workers` | All model sections and `warmup` | Integer at least `0` |
| `label_smoothing` | All model sections and `warmup` | Number in `[0, 1]`; default `0.0` |
| `training_batch_size`, `epochs`, `learning_rate` | Frozen SR2 and LoRA | Positive integers for batches/epochs; positive number for rate |
| `head_inference_batch_size`, `gradient_accumulation_steps` | LoRA | Positive integers |
| `min_lr_ratio` | Frozen SR2 and LoRA | Number in `[0, 1]`; default `0.125`; minimum learning rate as a fraction of `learning_rate` |
| `lr_decay_power` | Frozen SR2 | Finite positive number or `e` (Euler's number); default `2.0`; polynomial learning-rate decay exponent |
| `warmup_min_lr_ratio` | `warmup` | Number in `[0, 1]`; default `0.125`; minimum head-warmup learning rate as a fraction of `warmup_learning_rate` |
| `warmup_lr_decay_power` | `warmup` | Finite positive number or `e` (Euler's number); default `2.0`; polynomial warmup decay exponent |
| `max_grad_norm` | Frozen SR2 and LoRA | Positive number |
| `lora_rank`, `lora_alpha` | LoRA | Positive integers |
| `lora_dropout` | LoRA | Number in `[0, 1)` |
| `lora_target_modules` | LoRA | Nonempty list of unique nonempty strings |
| `gradient_checkpointing_threshold`, `gradient_checkpointing_budget` | LoRA | Integers at least `0`; each defaults to `0` |

`warmup` additionally requires `warmup_epochs` (integer at least `0`),
`warmup_training_batch_size` (positive integer), `warmup_learning_rate` and
`max_grad_norm` (positive numbers). It does not accept `training_batch_size`,
`epochs`, `learning_rate`, `min_lr_ratio`, `lr_decay_power`, or `A/B_swap`.

Both SR2 sections reject `head_learning_rate`, `A/B_swap`, and the
`warmup_*` optimization keys.
Internal `ab_swap`, `warmup_extraction_budget`, and `warmup_label_smoothing` are
not YAML fields. See [migration](migration.md) for retired keys.

`SR2_Qwen3_4B` trains only the SR2 head with polynomial-decayed `learning_rate`,
using `training_batch_size` for both optimization groups and head inference. It accepts
the common model fields above, `epochs`, `training_batch_size`, `learning_rate`,
`min_lr_ratio`, `lr_decay_power`, and `max_grad_norm`; it rejects LoRA,
joint-training, and `warmup_*` controls.
Its independent section needs neither `warmup` nor `SR2_Qwen3_4B_lora`.
Shipped values were copied once from each preset's warmup settings; changing
either section later does not modify the other. Compatible training states and
features still share the [warmup caches](caching.md#frozen-sr2-head-training).

Joint training uses `learning_rate` for both adapters and the classifier head.
One cosine curve spans all joint epochs and reaches
`learning_rate * min_lr_ratio` on the final planned optimizer update; a run with
only one update uses the initial rate. `min_lr_ratio: 0` decays to zero, while
`1` keeps the rate fixed. Omitting the field uses `0.125`.

Frozen SR2 uses polynomial decay across all head-training epochs, controlled by
`SR2_Qwen3_4B.lr_decay_power` (default `2.0`). Head warmup has an independent
polynomial curve controlled by `warmup.warmup_lr_decay_power` (default `2.0`),
from `warmup.warmup_learning_rate` to
`warmup.warmup_learning_rate * warmup.warmup_min_lr_ratio`, with the ratio also
defaulting to `0.125`. Both exponents accept finite positive numbers or the
case-insensitive string `e`, resolved to `math.e` (`2.718281828459045`). YAML
accepts both `lr_decay_power: e` and `lr_decay_power: "e"`; warmup supports the
same forms. Effective configuration snapshots, checkpoint metadata, and cache
identities use the resolved numeric exponent. The first update uses the initial
rate, the final update uses the minimum, and a single
update uses the initial rate. Set `SR2_Qwen3_4B.min_lr_ratio: 1` or
`warmup.warmup_min_lr_ratio: 1` to retain fixed-rate head training. See
[learning-rate decay](lora.md#learning-rate-decay) for formulas and update counting.

## Label smoothing

For three classes, training uses the target
`(1 - label_smoothing) * one_hot + label_smoothing / 3`.
This applies to warmup and joint-training cross entropy.
Warmup's epoch-end training-set inference uses the same smoothing as its
training objective. Validation and reported validation metrics use unsmoothed
log loss.

| Setting | Applies to |
| --- | --- |
| `SR2_Qwen3_4B.label_smoothing` | Frozen SR2 head training and epoch-end training-set evaluation |
| `SR2_Qwen3_4B_lora.label_smoothing` | SR2 joint training only |
| `warmup.label_smoothing` | All inline and standalone head warmup |

Omitting warmup smoothing gives `0.0` independently of SR2 joint training.
Nonzero warmup smoothing participates in the warmup cache key.

## Standalone warmup

SR2 takes exactly these controls from `warmup`:

| `warmup` field | Effect inside a LoRA run |
| --- | --- |
| `warmup_epochs` | Target head-only epochs; polynomial decay reuses only the exact target, while fixed-rate training can continue from a compatible lower epoch; `0` disables warmup and its cache I/O |
| `warmup_training_batch_size` | Head forward batch size during warmup |
| `warmup_learning_rate` | Initial warmup head learning rate |
| `warmup_min_lr_ratio` | Final warmup rate as a fraction of its initial rate; independent of joint `min_lr_ratio` |
| `warmup_lr_decay_power` | Polynomial warmup decay exponent; default `2.0`; independent of frozen SR2's exponent |
| `extraction_budget` | Warmup backbone extraction token budget |
| `label_smoothing` | Warmup training objective |

Inline LoRA warmup uses the selected LoRA section's own `model_name`,
`max_length`, head width/dropout, weight decay, clipping, seed, and workers.
The corresponding fields in `warmup` configure **only** `model warmup --build`.
They do not override SR2's model/head settings.

Standalone warmup requires positive `warmup_epochs` and uses the saved train
partition for training, the SR2 comparison head, and original rows only.
It also reads validation for evaluation. Epoch-end inference reports the full
training loss with warmup smoothing and validation's unsmoothed log loss;
both inference loaders use `warmup_training_batch_size`. Complete cache hits recompute
both losses from the restored head and current data without training.
Lower-epoch hits train and evaluate only the remaining epochs before saving a
separate target cache; all other identity fields must match.
It writes `SR2_Qwen3_4B_lora_warmup/`, sharing the inline SR2 namespace. The shipped
presets set `warmup.model_name` to `Skywork/Skywork-Reward-V2-Qwen3-4B`.
Adapter-free features are cached separately under `embedding_cache/` and shared
with inline warmup by token content and backbone computation identity. Head
settings and training IDs do not affect this feature cache; see
[warmup features](caching.md#warmup-features).

Matching shared controls is insufficient for cache reuse: model/head settings
and training IDs must also match. The current root preset and seven experiment
presets align standalone model/head settings with their SR2 validation settings.
If editing either section, keep those settings aligned when reuse is intended.
A train-partition cache differs from the normal full-data build's training ID set.
See [complete compatibility rules](caching.md#lora-head-warmup).

## Holdout token balance

Preparation measures canonical rows and tokens before A/B augmentation. With
validation fraction `f`, the criterion is:

`abs(validation_tokens / (all_tokens * f) - 1) < max_token_deviation`

Row counts use the same relative formula with a fixed strict limit of `0.005`.
At `f = 0.2`, the default means less than 0.5% relative error around the 20%
target. Whole prompt-group exchanges preserve the initial class and row counts;
preparation fails before publication if balance cannot be reached.
See [data preparation](data-preparation.md#one-shared-trainvalidation-split).

## Extraction token budgets

`extraction_budget` limits padded tokens in no-gradient backbone micro-batches:

| Setting | Used for |
| --- | --- |
| `warmup.extraction_budget` | Inline LoRA and standalone warmup extraction |
| A LoRA section's `extraction_budget` | Its joint-validation and submission extraction |
| A LoRA section's `head_inference_batch_size` | Head inference after extraction, including inline warmup train/validation evaluation; counts pairs, not tokens |
| `warmup.warmup_training_batch_size` | Standalone warmup head training and train/validation inference batch size |
| `SR2_Qwen3_4B.extraction_budget` | Frozen SR2 missing-branch feature extraction |
| `SR2_Qwen3_4B.training_batch_size` | Frozen SR2 head training and train/validation inference batch size |

Extraction sorts individual A/B branches by descending length, breaking ties by
original branch index. Batch cost is `branch_count * longest_branch_length`;
equality with the budget is allowed. A branch longer than the budget runs alone,
so the budget is **not a hard memory or input-length cap**. The remaining branches
are packed by comparing two candidates, minimizing micro-batch count first and
then total padded tokens:

- A suffix dynamic program finds the optimal contiguous partition in the sorted
  order. Equal DP scores retain the first enumerated cut, shortest prefix first.
- The retained best-fit candidate greedily fills batches in descending branch
  order. Exact ties between candidates retain best-fit.

Oversized singletons contribute the same fixed count and cost to both candidates.
When all eligible branches fit in one batch, that result is returned directly.
The DP uses linear extra space and quadratic time in the worst case; descending
lengths make each interval's maximum its first length and bound feasible cuts by
the budget. This shared sampler also handles retained frozen-model extraction.
Features are scattered back to their original pair positions.

LoRA rejects empty or over-limit branches and never truncates. Extraction plans
stay in memory. Changing the extraction budget does not invalidate tokens,
warmup features, warmup states, or checkpoints. Historical frozen caches are
described in
[caching](caching.md#frozen-embeddings).

## Joint-training packing

These two LoRA controls serve different purposes:

| Control | `0` | Positive value |
| --- | --- | --- |
| `gradient_checkpointing_threshold` | Initial fixed-size micro-batches; checkpoint all joint batches | Pack initial batches by total padded A/B tokens; checkpoint only batches above the threshold |
| `gradient_checkpointing_budget` | Disable the second packing pass | Combine checkpointed singletons inside each optimizer update, within this total padded A/B token budget |

Both controls use `rows * (longest_A + longest_B)`, with each branch padded
independently. Joint training retains both branch graphs until one backward pass.
Joint updates target `training_batch_size * gradient_accumulation_steps` rows.
A short tail merges into the preceding full update and uses the combined row
count for loss normalization. With positive token packing, micro-batch sizes can
exceed `training_batch_size`. When `gradient_checkpointing_threshold > 0`,
`gradient_checkpointing_budget` must exceed `2 * gradient_checkpointing_threshold`
to merge any checkpointed singletons; when `gradient_checkpointing_threshold: 0`,
use `training_batch_size: 1` so initial micro-batches are singletons eligible for
budget repacking. Warmup and inference disable checkpointing.
See [the packing algorithm](lora.md#optional-token-budget-training-batches) and
[memory tuning](lora.md#memory-tuning) before changing these values.

## GPU configuration

Validation, full-data builds, and standalone warmup use `gpu.device`.
Submission splits one backbone across `gpu.submit_devices`: embeddings/rotary
modules and the first half of decoder layers on the first GPU, remaining layers
and final normalization on the second, and the classifier head on the first.
Adapters follow their base layers; decoder layers remain intact.

Both submission GPUs must support the selected precision. There is no automatic
precision change, CPU/disk offload, or single-GPU submission fallback. Hardware
checks still run on complete warmup-cache hits. Standalone warmup still initializes
backbone weights on a hit, but compatible feature hits skip backbone forward
calls while the head recomputes current train/validation losses.
`ALL` executes SR2 on the configured device layout.

Device placement can change without invalidating caches or checkpoints.
SR2 checkpoints allow precision, attention, and TF32 changes; compatibility does
not guarantee identical predictions. See
[precision and attention behavior](lora.md#hardware-and-input-limits)
and [submission setup](getting-started.md#submission-environment-pytorch-211--cuda-128).

## Batch experiments

Use [batch validation](commands.md#batch-validation) for the script's selection,
ordering, failure handling, and log locations. Each run saves the effective
configuration beside its metrics; see [configuration snapshots](reports.md#configuration-snapshots).
