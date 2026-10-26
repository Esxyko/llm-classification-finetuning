# Configuration

[Documentation index](../README.md)

## Configuration files and paths

All commands read `config.yaml` from the working directory by default. Pass a
complete YAML file with `--config PATH` to `init`, `data`, `preprocess`, `model`,
or `result`. Relative config-file paths resolve from the working directory.
All commands load `.env` from that directory, preserving existing environment
variables.

Set `MODEL_PATH` in `.env` to choose the persistent native Hugging Face Hub cache
directory. Each repository uses a directory there, such
as `models--Qwen--Qwen3-4B/`. A relative `MODEL_PATH` resolves from the working
directory, and `~` expands to the user's home directory.

```dotenv
MODEL_PATH=D:/models
CACHE_PATH=cache
```

In this example, Qwen3-4B uses `D:/models/models--Qwen--Qwen3-4B/`.
Project model/config/tokenizer loaders pass `MODEL_PATH` as `cache_dir` and reuse
downloaded files across processes. Hugging Face manages refs, snapshots, blobs,
download locks, and resumable downloads in its native layout. Files remain on
disk after the process exits; no packing or unpacking occurs.
If `MODEL_PATH` is missing or blank, Hugging Face keeps its usual cache settings.
Explicit local model directories are loaded directly. See
[model cache layout](caching.md#hugging-face-model-downloads) for existing downloads.

Set `CACHE_PATH` in `.env` to choose the shared training-cache root for token
encodings, frozen embeddings, adapter batch plans, and head-warmup states.
A missing or blank value defaults to `./cache`. Relative paths resolve from the
working directory, `~` expands to the user's home directory, and absolute paths
stay absolute. Existing environment variables take precedence over `.env`.
The cache root is independent of YAML data paths and `MODEL_PATH`.

`MODEL_PATH` controls model downloads only. `DATA_PATH` is no longer used.
Relative YAML `data.raw_dir` and `data.processed_path` values resolve from the
working directory; absolute YAML paths stay absolute. With the default YAML,
raw files use `./data/raw`, processed data uses
`./data/processed/train_validation.parquet`, training caches use `./cache/`
unless `CACHE_PATH` selects another root, checkpoints use `./models/`, and
reports use `./results/`.

For example, from the project root:

```shell
uv run model qwen3-4b --config configs/2.yaml
```

## Configuration sections

| Section | Controls |
| --- | --- |
| `data` | Competition name and raw/processed paths |
| `validation` | Holdout fraction, random seed, tokenizer identity, and maximum validation token deviation |
| `gpu` | Training `device`, two `submit_devices` (default `[cuda:0, cuda:1]`), precision, attention implementation, and TF32, shared by all models |
| `warmup` | Shared warmup epochs, extraction token budget, training batch size, and learning rate for LoRA and `model warmup --build`; also configures the standalone warmup model, input limit, classifier, and optimizer |
| `qwen3_4b` | Qwen3-4B model, sequence length, batches, MLP, optimizer, and `A/B_swap` training mode (`aug` or `inf`) |
| `qwen3_4b_lora_EXPT2` | EXPT2 model, sequence length, training batches, extraction token budget, head inference batch size, gradient checkpointing, accumulation, adapters, and classifier |
| `SR2_Qwen3_4B_lora` | Skywork reward-backbone LoRA with the same fields and EXPT2 classifier |

Model sections are optional. Explicitly selecting a profile without its section
reports a configuration error; `ALL` requires at least one configured profile.
The `qwen3_4b_lora_EXPT2` and `SR2_Qwen3_4B_lora` sections enable their respective
profiles in `ALL`, after the frozen profile and in that order.
Retired `qwen3_4b_lora`, `qwen3_4b_lora_EXPT`, and `qwen3_4b_qlora` keys are
rejected as unknown root keys. Rename an older shared `qwen3_4b_lora` section
to `qwen3_4b_lora_EXPT2` to run EXPT2; see [retired profiles](models.md#retired-profiles).

Every command validates the entire selected YAML before doing work, including
any model sections it will not use. `data`, `validation`, and `gpu` are
required even for data preparation and reporting. Each model section, when
present, must include all its required fields; config files do not inherit or
merge values from the root `config.yaml`.

Only these YAML fields have parser defaults when omitted:

| Field | Default |
| --- | --- |
| `validation.tokenizer_name` | `Qwen/Qwen3-4B` |
| `validation.tokenizer_revision` | `null` (resolve the current revision) |
| `validation.validation_fraction` | `0.2` |
| `validation.max_token_deviation` | `0.005` (strictly less than 0.5% relative to the validation token target) |
| `gpu.submit_devices` | `[cuda:0, cuda:1]` |
| `qwen3_4b.A/B_swap` | `aug` |
| `qwen3_4b.label_smoothing` / `qwen3_4b_lora_EXPT2.label_smoothing` / `warmup.label_smoothing` | `0.0` |
| `qwen3_4b_lora_EXPT2.gradient_checkpointing_threshold` | `0` |
| `qwen3_4b_lora_EXPT2.gradient_checkpointing_budget` | `0` |
| `SR2_Qwen3_4B_lora.label_smoothing` | `0.0` |
| `SR2_Qwen3_4B_lora.gradient_checkpointing_threshold` | `0` |
| `SR2_Qwen3_4B_lora.gradient_checkpointing_budget` | `0` |

All other fields in a present section are required, including `gpu.device`,
`warmup.warmup_epochs`, extraction budgets, and the warmup training batch size
and learning rate. Unknown keys are rejected. Numeric values described as
presets in these guides refer to the
root [`config.yaml`](../config.yaml); files in `configs/` can use different values.

## Label smoothing

Each model section and `warmup` accept optional `label_smoothing`, a finite
number in `[0, 1]` defaulting to `0.0`. The root configuration and active presets
record it explicitly; values can differ between the frozen, EXPT2, SR2, and
warmup sections.
For three classes, smoothing replaces a one-hot target with
`(1 - label_smoothing) * one_hot + label_smoothing / 3`.
It applies to ordinary cross entropy and to the loss of averaged A/B
probabilities in `A/B_swap: inf` mode. Validation and reporting continue to use
unsmoothed log loss against the original labels.

`qwen3_4b.label_smoothing` controls frozen-head training.
Each LoRA section's `label_smoothing` controls only its joint training.
`warmup.label_smoothing` always controls head warmup, including standalone
`model warmup --build` and both LoRA profiles' inline warmup. If omitted, it
defaults to `0.0` independently of the model section's value.
Nonzero smoothing values enter the warmup-cache identity, while zero retains
existing unsmoothed cache keys. Effective values are included in result
configuration snapshots.

## Pairwise absolute difference

The frozen profile classifies `[A, B, |A - B|]` with a `3H` input. EXPT2, SR2, and
standalone warmup use the sum, signed difference, and absolute-difference
features described below. The absolute-difference feature is mandatory and
has no configuration switch. Metrics record each head's fixed combination.

## Experimental LoRA model

Select `model qwen3-4b-lora-EXPT2` using `qwen3_4b_lora_EXPT2`. EXPT2 uses a
fixed original-only policy. It derives the tie
representation from activated win branches before dropout; see the
[EXPT2 head definition](lora.md#expt2-equivariant-head).

Select `model SR2-Qwen3-4B-lora` using `SR2_Qwen3_4B_lora` for the same head and
training behavior over `Skywork/Skywork-Reward-V2-Qwen3-4B`. The original scalar
reward head is discarded. In the root config, SR2 matches the EXPT2 parameters
apart from `model_name`. Experiment presets can also differ in smoothing,
classifier size, learning rates, and epoch count; the sections are independent,
without implicit inheritance. SR2 uses its own tokenizer and backbone revision
while retaining the existing conversation serialization and saved split.

Warmup, holdout training, and full-data builds select only original rows from
the existing processed data. Validation and submission execute the head once
per pair. The shared `warmup` section supplies its three required optimization
fields and extraction budget, plus optional label smoothing. Standalone warmup
uses the same EXPT2 head and original-only policy. Both LoRA profiles' snapshots omit `A/B_swap`
from both sections.
`A/B_swap` is supported only in `qwen3_4b`; remove it from the two LoRA
sections and `warmup` when migrating older configs, as it is rejected there.

The existing EXPT2 checkpoint and warmup-cache namespaces remain unchanged.
SR2 has separate checkpoint, warmup-cache, and result namespaces. The batch
validation script selects models and config filenames through `MODEL_CONFIGS`;
see [batch experiments](#batch-experiments) for the current selection.

## Standalone warmup

The `warmup` section is required when either LoRA section is present. Both read
`warmup_epochs`, `warmup_training_batch_size`, and `warmup_learning_rate` from
this section; these three keys are rejected inside either LoRA section. Move their
existing values into `warmup` when migrating an older configuration.
LoRA warmup also uses `warmup.label_smoothing` and `warmup.extraction_budget`, while
each LoRA section's `extraction_budget` controls its joint-validation and submission
extraction. The resolved `warmup_extraction_budget` and `warmup_label_smoothing`
fields are internal and are rejected in YAML; they are omitted from configuration
snapshots.
Set `warmup.warmup_epochs: 0` to disable LoRA head warmup and its cache.

The section can also be used without any model sections. A config containing
`data`, `validation`, `gpu`, and `warmup` is sufficient for:

```shell
uv run model warmup --build --config config.yaml
```

Its fields are `model_name`, `max_length`, `hidden_size`, `dropout`,
`warmup_epochs`, `extraction_budget`, `warmup_training_batch_size`,
`warmup_learning_rate`, `weight_decay`, `max_grad_norm`, `random_state`,
`dataloader_workers`, and `label_smoothing`. All except `label_smoothing`
are required; its default is `0.0`.
`warmup_epochs` must be nonnegative;
`model warmup --build` requires it to be strictly positive. The other fields in
this section configure standalone warmup; LoRA continues to use its own model,
classifier, weight decay, clipping, seed, and workers settings. Both commands
always select original rows.
SR2's inline warmup uses its own Skywork backbone and head settings. The
standalone command continues to write the EXPT2 cache namespace; use SR2's
validation or build command to generate its separate warmup state.

The root preset uses Qwen3-4B, `max_length: 13514`, a 256-unit EXPT2 head with
dropout 0.1, one warmup epoch, extraction budget 524288, training batch size
1024, learning rate 0.001, weight decay 0.01, clipping 1.0, seed 42, and zero
workers. These are explicit preset values, not parser defaults. Shared GPU
settings control its single training device and precision. The root preset has
`warmup.label_smoothing: 0.12`, which applies to standalone and inline warmup
regardless of either LoRA section's joint-training smoothing value.

To reuse the result in LoRA, its warmup compatibility fields and actual
training IDs must match. Warmup extraction and the three shared optimization
fields, as well as label smoothing, apply to both commands; the remaining
settings must be aligned separately.
Full-data builds use a different training ID set and do not reuse this
train-partition cache. See
[warmup compatibility](caching.md#lora-head-warmup).

## Holdout token balance

`init` and `preprocess` assign one stratified grouped holdout, then exchange
whole groups with identical original class counts. At the default fraction,
validation contains approximately 20% of canonical rows and tokens. The token
criterion is `abs(validation_tokens / (all_tokens * validation_fraction) - 1)
< max_token_deviation`; row deviation must be strictly below 0.005 using the same
formula for counts. The fraction and token tolerance must be finite numbers in
`(0, 1)`. Group swaps preserve the initial class and row counts exactly.

Legacy `cross_validation`/`n_splits` settings are rejected with a migration
message. Use the new processed path and regenerate data; old artifacts remain
available on disk. See [migration](data-preparation.md#migration-from-five-folds).

Token counting uses the training conversation serializer, special tokens, and
the shared `<CACHE_PATH>/token_cache/`. It excludes padding and
truncation and loads no model weights or GPU. `tokenizer_name` accepts a Hub
name or local tokenizer/model directory. `tokenizer_revision` is optional; pin
an immutable revision for reproducible counts. The root configuration and
active presets pin
`Qwen/Qwen3-4B` to `1cfa9a7208912126459214e8b04321603b3df60c`.

These settings are independent of the optional training model sections. When
using a different training tokenizer, update the balance tokenizer as well.
Missing tokenizer files may be downloaded on preparation. If equal-class group
swaps cannot reach the target within 1,000 exchanges, preparation fails before
replacing the processed dataset. See [data preparation](data-preparation.md).

## Extraction token budgets

`extraction_budget` replaces `extraction_batch_size` in `warmup`, `qwen3_4b`,
`qwen3_4b_lora_EXPT2`, and `SR2_Qwen3_4B_lora`. It is a required positive integer
measured in padded tokens; the old key is rejected. The root configuration and
active presets use `524288` for frozen extraction and warmup, and `262144` for
both LoRA profiles' joint-validation/submission extraction. Old branch counts
such as `32` or `64` are not equivalent token budgets.

Each LoRA section's `extraction_budget` controls backbone extraction
for every joint-validation pass and submissions, including when warmup is
disabled. `warmup.extraction_budget` controls warmup extraction for all LoRA
profiles and standalone warmup. `qwen3_4b.extraction_budget` controls extraction
of missing frozen-model branches.

All three use the same deterministic best-fit decreasing packing: sort individual
A/B branches by length descending, breaking ties by original branch index, and
place each into the fitting micro-batch with the least remaining budget. Cost is
`branch_count * longest_branch_length`, including padding. Equality is allowed;
a branch longer than the budget runs alone. Branch lengths follow the existing
input rules: frozen inputs may be truncated to `max_length`, while LoRA rejects
empty or over-limit inputs. Features are written back to their original positions
to preserve ID order and A/B alignment even across odd-sized micro-batches.
Plans stay in memory, and the budget does not affect cache or checkpoint identity.
Head inference uses its own batch size after extraction completes.

For LoRA, `head_inference_batch_size` is a required positive integer replacing
`inference_batch_size`; rename the old key while preserving its value. The old
key is rejected.

## Joint-training packing

`gradient_checkpointing_threshold` is the new name for the previous
`gradient_checkpointing_budget` setting; rename the old key while preserving its
value. A positive threshold packs initial joint-training micro-batches by padded
A/B token cost and enables checkpointing above that threshold. At or below it,
checkpointing is disabled. Zero or omission keeps initial fixed micro-batches
and enables checkpointing for every joint-training batch.

`gradient_checkpointing_budget` now controls a second packing pass within each
optimizer update, after any short final group is merged. Its default `0` disables
this pass. A positive value combines only singletons that require checkpointing,
using `rows * (longest_A + longest_B)` as the padded token cost. Existing
multi-row micro-batches remain intact. Combined micro-batches cannot exceed the
budget; an individual sample above it still runs alone without truncation.
This also applies when the threshold is `0`. Both fields must be nonnegative
integers, and the budget need not exceed the threshold. Warmup and inference
disable checkpointing. The older `training_token_budget` key and
`gradient_checkpointing` boolean remain rejected.
See [packing behavior](lora.md#optional-token-budget-training-batches).

`gradient_accumulation_steps` applies only to joint training. Warmup uses
`warmup_training_batch_size` for forward batches and merges a short final batch
with the preceding full batch through gradient accumulation for one optimizer
update. Loss is normalized by the combined row count. Each nonempty warmup
epoch performs `max(1, training_rows // warmup_training_batch_size)` updates;
an epoch smaller than one batch still updates once, and exact multiples retain
one update per full batch. This also applies to standalone warmup.
Joint updates target `G = training_batch_size * gradient_accumulation_steps`
rows. An incomplete final group is merged into the preceding full group, using
the combined row count to normalize loss and performing one parameter update.
The merged joint-training group runs last, as in warmup.
An epoch smaller than `G` still performs one update. Token-budget training first
shuffles full groups, then merges the short group into the last shuffled group
and keeps the combined update last.
LoRA uses `max(1, training_rows // G)` joint updates per nonempty
epoch for scheduling and progress. See [LoRA training](lora.md).

## GPU configuration

Holdout validation and `--build` use the single CUDA device selected by
the required `gpu.device` field. The root preset selects `cuda:0`.
All models use one backbone for A and B; classifier head training uses the
same GPU. LoRA joint training encodes A and B sequentially. Its
no-gradient extraction batches individual branches from either side together,
as frozen extraction does for unique missing branches.

`--submit` splits that backbone across the two devices in `gpu.submit_devices`,
which defaults to `[cuda:0, cuda:1]` when omitted. The list must contain exactly
two distinct `cuda:N` identifiers; it is independent of the training device.
The root preset's GPU configuration is:

```yaml
gpu:
  device: cuda:0
  submit_devices: [cuda:0, cuda:1]
  precision: bf16
  attention_implementation: sdpa
  allow_tf32: false
```

The embedding and rotary modules and the first half of the decoder layers reside
on the first GPU; the remaining layers and final normalization reside on the
second. Each decoder layer stays intact, LoRA adapters follow their base layers,
and the classifier head's FP32 parameters reside on the first GPU. With
`gpu.precision: bf16`, LoRA uses BF16 head computation and pooled/CPU
features, returning FP32 logits; the frozen `qwen3-4b` head retains FP32
computation. LoRA gradient checkpointing
is disabled for submissions. Both GPUs must be available and support
`gpu.precision`; there is no automatic precision change, single-GPU fallback, or
CPU/disk offload. `ALL` still runs models sequentially, with each model split
across both GPUs.

`gpu.precision` accepts `fp16`, `bf16`, or `fp32`;
`gpu.attention_implementation` accepts `sdpa` or `eager`. Hardware validation
checks CUDA availability and BF16 support on each selected device. It runs even
when every frozen embedding is cached, so submissions still require two GPUs.

Changing device placement preserves checkpoint and cache compatibility.
Changing precision selects a different frozen embedding cache; attention
implementation and TF32 changes preserve cache compatibility. Both frozen-head
and LoRA checkpoints allow those compute changes; shared tokens remain reusable
when the tokenizer and encoding options match. See
[frozen checkpoint compatibility](models.md#checkpoint-compatibility),
[LoRA checkpoints](lora.md#checkpoints), and [caching](caching.md) for the exact
boundaries.

Complete frozen embedding-cache hits skip loading backbone weights, while still
loading the model configuration and tokenizer. Submissions are written to
`results/submit/MODEL-*/submission.csv`; historical `results/test` files remain
untouched. For frozen extraction memory errors, see
[troubleshooting](troubleshooting.md).

## Batch experiments

To validate selected model/config combinations on the fixed holdout, run:

```shell
uv run python scripts/validate.py
```

Edit the settings near the top of `scripts/validate.py`:

```python
CONFIG_DIR = "configs"
MODEL_CONFIGS: dict[str, list[str]] = {
    "SR2-Qwen3-4B-lora": ["1.yaml", "2.yaml", "3.yaml", "4.yaml"],
}
```

`CONFIG_DIR` is a string path relative to the project root. Each dictionary key
is a model selector, and its list contains the config filenames to run for that
model, relative to `CONFIG_DIR`. Supported selectors are `qwen3-4b`,
`qwen3-4b-lora-EXPT2`, and `SR2-Qwen3-4B-lora`. The current script selects only
SR2 with `1.yaml` through `4.yaml`, as shown above. Add entries to `MODEL_CONFIGS`
to run other models or presets.
Models run sequentially in dictionary order, with each model's configs running
in list order. Only listed files run; newly added YAML files are not selected
automatically. An empty list skips that model. The script resolves the project
root from its own location and runs every model from that root, regardless of
the directory from which it was invoked.

Each model's validation run publishes its own `metrics.json` and an adjacent
`config.yaml` containing the effective shared and current model sections;
`metrics.json` does not store the source config path. `config_total_time` covers
only that model's holdout validation run under the selected config.
The script continues after a failed config/model task, including a missing
config file or model section in the YAML, and summarizes successful and failed
tasks. It exits with status `1` if any task fails or no config filenames are
selected.

Each script invocation creates one UTF-8 log at
`results/logs/validate-YYYYMMDD-HHMMSS-microseconds.log`. All tasks share that
file, which records the script's messages, each model's standard output and
standard error, exit codes for failed tasks, and the final summary while output
continues to appear in the terminal. The log is created before config checks and
flushed as output arrives, retaining output on failure. Unexpected exceptions
and keyboard interruptions also save their traceback before propagating.

See [LoRA training](lora.md) for context limits, warmup, and token budgets, or the
[command reference](commands.md) for execution modes and checkpoint selection.
