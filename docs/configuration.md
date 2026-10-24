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
```

In this example, Qwen3-4B uses `D:/models/models--Qwen--Qwen3-4B/`.
Project model/config/tokenizer loaders pass `MODEL_PATH` as `cache_dir` and reuse
downloaded files across processes. Hugging Face manages refs, snapshots, blobs,
download locks, and resumable downloads in its native layout. Files remain on
disk after the process exits; no packing or unpacking occurs.
If `MODEL_PATH` is missing or blank, Hugging Face keeps its usual cache settings.
Explicit local model directories are loaded directly. See
[model cache migration](caching.md#hugging-face-model-downloads) for existing downloads.

`MODEL_PATH` controls model downloads only. `DATA_PATH` is no longer used.
Relative YAML `data.raw_dir` and `data.processed_path` values resolve from the
working directory; absolute YAML paths stay absolute. With the default YAML,
raw files use `./data/raw`, processed data uses
`./data/processed/train_validation.parquet`, training caches stay beside that
Parquet, checkpoints use `./models/`, and reports use `./results/`.

For example, from the project root:

```shell
uv run model qwen3-4b --config configs/config_aug_default.yaml
```

## Configuration sections

| Section | Controls |
| --- | --- |
| `data` | Competition name and raw/processed paths |
| `validation` | Holdout fraction, random seed, tokenizer identity, and maximum validation token deviation |
| `gpu` | Training `device`, two `submit_devices` (default `[cuda:0, cuda:1]`), precision, attention implementation, and TF32, shared by all models |
| `warmup` | Shared warmup epochs, training batch size, and learning rate for LoRA and `model warmup --build`; also configures the standalone warmup model, input limit, classifier, extraction, and optimizer |
| `qwen3_4b` | Qwen3-4B model, sequence length, batches, MLP, optimizer, and `A/B_swap` training mode (`aug` or `inf`) |
| `qwen3_4b_lora` | Live LoRA training, per-branch sequence length, training batches, shared `extraction_batch_size` for no-gradient extraction, `head_inference_batch_size` for validation/submission heads, optional `gradient_checkpointing_budget` for batch packing and automatic checkpointing, accumulation, adapter and classifier hyperparameters; `A/B_swap` training mode (`aug` or `inf`) |

Model sections are optional. Explicitly selecting a profile without its section
reports a configuration error; `ALL` requires at least one configured profile.
The retired `qwen3_4b_qlora` section is rejected as an unknown root key. Remove
it from older configurations; see [retired profiles](models.md#retired-profiles).

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
| `qwen3_4b.A/B_swap` / `qwen3_4b_lora.A/B_swap` / `warmup.A/B_swap` | `aug` |
| `qwen3_4b_lora.gradient_checkpointing_budget` | `0` |

All other fields in a present section are required, including `gpu.device`,
`warmup.warmup_epochs`, and the warmup batch sizes and learning rate. Unknown keys
are rejected. Numeric values described as presets in these guides refer to the
root [`config.yaml`](../config.yaml); files in `configs/` can use different values.

## Standalone warmup

The `warmup` section is required when `qwen3_4b_lora` is present. LoRA reads
`warmup_epochs`, `warmup_training_batch_size`, and `warmup_learning_rate` from
this section; these three keys are rejected inside `qwen3_4b_lora`. Move their
existing values into `warmup` when migrating an older configuration.
Set `warmup.warmup_epochs: 0` to disable LoRA head warmup and its cache.

The section can also be used without any model sections. A config containing
`data`, `validation`, `gpu`, and `warmup` is sufficient for:

```shell
uv run model warmup --build --config config.yaml
```

Its fields are `model_name`, `max_length`, `hidden_size`, `dropout`,
`warmup_epochs`, `extraction_batch_size`, `warmup_training_batch_size`,
`warmup_learning_rate`, `weight_decay`, `max_grad_norm`, `random_state`,
`dataloader_workers`, and `A/B_swap`. All except `A/B_swap` are required;
`A/B_swap` defaults to `aug`. `warmup_epochs` must be nonnegative;
`model warmup --build` requires it to be strictly positive. The other fields in
this section configure standalone warmup; LoRA continues to use its own model,
classifier, extraction, weight decay, clipping, seed, workers, and `A/B_swap` settings.

The root preset copies the LoRA warmup settings: Qwen3-4B, `max_length: 13514`,
a 512-unit head with dropout 0.1, one warmup epoch, extraction batch size 16,
training batch size 1024, learning rate 0.001, weight decay 0.01, clipping 1.0,
seed 42, zero workers, and `A/B_swap: aug`. These are explicit preset values,
not parser defaults. Shared GPU settings control its single training device
and precision.

To reuse the result in LoRA, its warmup compatibility fields and actual
training IDs must match. The three shared warmup optimization fields apply to
both commands; the remaining settings must be aligned separately. Full-data
builds use a different training ID set and do not reuse this train-partition cache. See
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
the shared `token_cache/` beside the processed Parquet. It excludes padding and
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

For LoRA, `head_inference_batch_size` is a required positive integer replacing
`inference_batch_size`; rename the old key while preserving its value. The old
key is rejected. `extraction_batch_size` controls backbone extraction for
warmup, every joint-validation pass, and submissions, including when warmup is
disabled. It counts individual A/B branches per backbone forward, matching the
frozen model's `extraction_batch_size`; the standalone warmup section uses the
same definition. Head inference uses its own batch size after extraction completes.

`gradient_checkpointing_budget` is a nonnegative integer replacing
`training_token_budget` and the `gradient_checkpointing` boolean; both old keys
are rejected. A positive budget packs joint-training micro-batches by padded A/B
token cost, disables checkpointing at or below the budget, and enables it only
for over-budget singletons. Zero or omission keeps fixed micro-batches and
enables checkpointing for every joint-training batch. Warmup and inference
disable checkpointing. See [budget behavior](lora.md#optional-token-budget-training-batches).

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
An epoch smaller than `G` still performs one update. Token-budget training first
shuffles full groups, then merges the short group into the last shuffled group.
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

To validate selected models on the fixed holdout for every `.yaml` or `.yml` file in `configs/`,
run:

```shell
uv run python scripts/validate_all_configs.py
```

To train selected models on all original training and validation IDs and build a separate checkpoint for each
config, run:

```shell
uv run python scripts/build_all_configs.py
```

Both scripts define `MODELS = ("qwen3-4b-lora",)` near the top. Edit each script's
tuple independently to select models and their execution order; supported model
selectors are `qwen3-4b` and `qwen3-4b-lora`. Config files run in filename order,
with the selected models running sequentially for each config. The scripts are
independent: the build script does not run or require holdout validation.
They locate `configs/` relative to the script's project root and run each model
from that root, regardless of the directory from which the script was invoked.

Each model's validation run publishes its own `metrics.json` and an adjacent
`config.yaml` containing the effective shared and current model sections;
`metrics.json` does not store the source config path. `config_total_time` covers
only that model's holdout validation run under the selected config.
The full-data build does not produce `metrics.json`. By default, it writes
`models/qwen3_4b_lora/adapter_CONFIG_STEM.pt` for each config; selecting `qwen3-4b`
also writes `models/qwen3_4b/head_CONFIG_STEM.pt`. Before starting any builds,
the script rejects config stems that are invalid checkpoint tags or duplicate
another stem case-insensitively.

Both scripts continue after a failed config/model task, including a missing
model section in the YAML, and summarize successful and failed tasks. They exit
with status `1` if any task fails, no YAML files are found, or `MODELS` is empty.
To use a tagged checkpoint for submission inference, pass its config and matching
checkpoint tag, for example:

```shell
uv run model qwen3-4b-lora --submit --config configs/config_aug_default.yaml --checkpoint-tag config_aug_default
```

See [LoRA training](lora.md) for context limits, warmup, and token budgets, or the
[command reference](commands.md) for execution modes and checkpoint selection.
