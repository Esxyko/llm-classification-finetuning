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

| Setting or artifact | Location / fallback | Controls |
| --- | --- | --- |
| `--config PATH` | `./config.yaml` | Selected complete YAML file |
| `data.raw_dir` | Required YAML path; root preset `data/raw` | Raw competition CSVs |
| `data.processed_path` | Required YAML `.parquet` path; root preset `data/processed/train_validation.parquet` | Saved train/validation data, including multipart layout |
| `MODEL_PATH` | Optional environment or `.env`; missing/blank uses native HF settings | Downloaded model/config/tokenizer cache |
| `CACHE_PATH` | Environment or `.env`; missing/blank uses `./cache` | Tokens, frozen embeddings, LoRA batch plans, and head-warmup states |
| Production checkpoints | `./models/SECTION/` | Frozen head or LoRA adapter/head files |
| Validation, submissions, reports | `./results/` | Run artifacts and comprehensive report |

Changing `MODEL_PATH` or `CACHE_PATH` does not relocate data, checkpoints, or
reports. `DATA_PATH` is unused. `DRIVE_RESULTS_DIR` belongs to the local Colab
notebook's copy step and is not an application configuration key.

For example, `uv run model SR2-Qwen3-4B-lora --config configs/2.yaml` still
resolves relative data and output paths from the project root when run there.
See [cache layout](caching.md) and [Colab persistence](colab.md).

## Configuration sections

| Section | Required when | Responsibility |
| --- | --- | --- |
| `data` | Always | Competition and data paths |
| `validation` | Always | Saved holdout fraction, seed, and balance tokenizer |
| `gpu` | Always | Training device, submission devices, precision, and attention |
| `qwen3_4b` | Selecting `qwen3-4b` | Frozen-backbone classifier |
| `qwen3_4b_lora_EXPT2` | Selecting `qwen3-4b-lora-EXPT2` | Qwen LoRA and EXPT2 head |
| `SR2_Qwen3_4B_lora` | Selecting `SR2-Qwen3-4B-lora` | Skywork LoRA and EXPT2 head |
| `warmup` | Either LoRA section is present, or selecting `warmup --build` | Shared warmup controls and standalone warmup model |

A model section may be absent; `model ALL` runs only present profiles in registry
order and requires at least one. `warmup` is not part of `ALL`.
Both LoRA sections are independent: neither inherits from the other or from
`warmup` beyond the five controls listed under [standalone warmup](#standalone-warmup).

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
| `validation.tokenizer_name` | Hub ID or local tokenizer directory; default `Qwen/Qwen3-4B` |
| `validation.tokenizer_revision` | Nonempty revision string or `null`; default `null` |
| `validation.max_token_deviation` | Number in `(0, 1)`; default `0.005` |

The tokenizer is used for preparation independently of optional model sections.
The shipped presets pin its revision for reproducible token counts. If you
change the training tokenizer and rebuild the split, update this tokenizer too.

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
| `training_batch_size`, `epochs`, `learning_rate` | Frozen and LoRA sections | Positive integers for batches/epochs; positive number for rate |
| `A/B_swap` | Frozen `qwen3_4b` only | `aug` or `inf`; default `aug` |
| `head_inference_batch_size`, `gradient_accumulation_steps` | Both LoRA sections | Positive integers |
| `head_learning_rate`, `max_grad_norm` | Both LoRA sections | Positive numbers |
| `lora_rank`, `lora_alpha` | Both LoRA sections | Positive integers |
| `lora_dropout` | Both LoRA sections | Number in `[0, 1)` |
| `lora_target_modules` | Both LoRA sections | Nonempty list of unique nonempty strings |
| `gradient_checkpointing_threshold`, `gradient_checkpointing_budget` | Both LoRA sections | Integers at least `0`; each defaults to `0` |

`warmup` additionally requires `warmup_epochs` (integer at least `0`),
`warmup_training_batch_size` (positive integer), `warmup_learning_rate` and
`max_grad_norm` (positive numbers). It does not accept `training_batch_size`,
`epochs`, `learning_rate`, or `A/B_swap`.

Both LoRA sections reject `A/B_swap` and the three `warmup_*` optimization keys.
Internal `ab_swap`, `warmup_extraction_budget`, and `warmup_label_smoothing` are
not YAML fields. See [migration](migration.md) for retired keys.

## Label smoothing

For three classes, training uses the target
`(1 - label_smoothing) * one_hot + label_smoothing / 3`.
This applies to ordinary cross entropy and the frozen `inf` objective over
averaged A/B probabilities. Validation and reporting use unsmoothed log loss.

| Setting | Applies to |
| --- | --- |
| `qwen3_4b.label_smoothing` | Frozen-head training |
| A LoRA section's `label_smoothing` | That profile's joint training only |
| `warmup.label_smoothing` | All inline and standalone head warmup |

Omitting warmup smoothing gives `0.0` independently of either LoRA section.
Nonzero warmup smoothing participates in the warmup cache key.

## Standalone warmup

Both LoRA profiles take exactly these controls from `warmup`:

| `warmup` field | Effect inside a LoRA run |
| --- | --- |
| `warmup_epochs` | Head-only epochs; `0` disables warmup and its cache I/O |
| `warmup_training_batch_size` | Head forward batch size during warmup |
| `warmup_learning_rate` | Fixed warmup head learning rate |
| `extraction_budget` | Warmup backbone extraction token budget |
| `label_smoothing` | Warmup training objective |

Inline LoRA warmup uses the selected LoRA section's own `model_name`,
`max_length`, head width/dropout, weight decay, clipping, seed, and workers.
The corresponding fields in `warmup` configure **only** `model warmup --build`.
They do not override either LoRA profile.

Standalone warmup requires positive `warmup_epochs` and uses the saved train
partition, the EXPT2 head, and original rows only. It writes the EXPT2 cache
namespace. SR2 creates its separate warmup during its own validation/build.

Matching shared controls is insufficient for cache reuse: model/head settings
and training IDs must also match. In the current root preset,
`warmup.hidden_size` is `128` and `qwen3_4b_lora_EXPT2.hidden_size` is `256`, so
standalone warmup does **not** prewarm that EXPT2 configuration. Align the
standalone settings in your selected YAML if reuse is intended. A train-partition
cache also differs from the normal full-data build's training ID set.
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
| `qwen3_4b.extraction_budget` | Missing frozen branch embeddings |
| `warmup.extraction_budget` | Inline LoRA and standalone warmup extraction |
| A LoRA section's `extraction_budget` | Its joint-validation and submission extraction |
| A LoRA section's `head_inference_batch_size` | Head inference after extraction; counts pairs, not tokens |

Extraction sorts individual A/B branches by descending length, breaking ties by
original branch index, then best-fit packs them at cost
`branch_count * longest_branch_length`. Equality is allowed. A branch longer
than the budget runs alone, so the budget is **not a hard memory or input-length
cap**. Features are scattered back to their original pair positions.

Frozen inputs may be truncated to `max_length`; LoRA rejects empty or over-limit
branches. Extraction plans stay in memory. Changing the extraction budget does
not invalidate tokens, frozen embeddings, warmup states, or checkpoints.

## Joint-training packing

These two LoRA controls serve different purposes:

| Control | `0` | Positive value |
| --- | --- | --- |
| `gradient_checkpointing_threshold` | Initial fixed-size micro-batches; checkpoint all joint batches | Pack initial batches by padded A/B tokens; checkpoint only batches above the threshold |
| `gradient_checkpointing_budget` | Disable the second packing pass | Combine checkpointed singletons inside each optimizer update, within this padded-token budget |

Joint updates target `training_batch_size * gradient_accumulation_steps` rows.
A short tail merges into the preceding full update and uses the combined row
count for loss normalization. With positive token packing, micro-batch sizes can
exceed `training_batch_size`. Warmup and inference disable checkpointing.
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
checks still run on complete frozen-cache hits, though backbone weights are not
loaded. `ALL` runs sequentially, not as replicas on separate GPUs.

Device placement can change without invalidating caches or checkpoints.
Precision changes select a different frozen embedding cache; attention and TF32
changes preserve its compatibility. Both checkpoint types allow those compute
changes. See [precision and attention behavior](lora.md#hardware-and-input-limits)
and [submission setup](getting-started.md#submission-environment-pytorch-211--cuda-128).

## Batch experiments

Use [batch validation](commands.md#batch-validation) for the script's selection,
ordering, failure handling, and log locations. Each run saves the effective
configuration beside its metrics; see [configuration snapshots](reports.md#configuration-snapshots).
