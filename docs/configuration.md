# Configuration

[Documentation index](../README.md)

## Configuration files and paths

All commands read `config.yaml` from the working directory by default. Pass a
complete YAML file with `--config PATH` to `init`, `data`, `preprocess`, `model`,
or `result`. Relative config-file paths and relative `data.raw_dir` and
`data.processed_path` values resolve from the working directory; absolute paths
stay absolute. Model artifacts and reports also remain under the working
directory. `init` and `data` load `.env` from the working directory.

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
| `warmup` | Independent non-quantized head-warmup model, input limit, classifier, optimization, and `A/B_swap` settings; used only by `model warmup --build` |
| `qwen3_4b` | Qwen3-4B model, sequence length, batches, MLP, optimizer, and `A/B_swap` training mode (`aug` or `inf`) |
| `qwen3_4b_lora` | Live LoRA training, per-branch sequence length, training batches, shared `extraction_batch_size` for no-gradient extraction, `head_inference_batch_size` for validation/submission heads, optional `training_token_budget`, accumulation, checkpointing, adapter and classifier hyperparameters; `A/B_swap` training mode (`aug` or `inf`) |
| `qwen3_4b_qlora` | Same fields as LoRA, with fixed 4-bit NF4 and double quantization for training/validation; submissions load a non-quantized base |

Model sections are optional. Explicitly selecting a profile without its section
reports a configuration error; `ALL` requires at least one configured profile.

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
| `qwen3_4b.A/B_swap` / `qwen3_4b_lora.A/B_swap` / `qwen3_4b_qlora.A/B_swap` / `warmup.A/B_swap` | `aug` |
| `qwen3_4b_lora.training_token_budget` / `qwen3_4b_qlora.training_token_budget` | `0` |

All other fields in a present section are required, including `gpu.device`,
`warmup_epochs`, and the warmup batch sizes and learning rate. Unknown keys are
rejected. Numeric values described as presets in these guides refer to the
root [`config.yaml`](../config.yaml); files in `configs/` can use different values.

## Standalone warmup

The optional `warmup` section is independent of all model sections. A config
containing `data`, `validation`, `gpu`, and `warmup` is sufficient for:

```shell
uv run model warmup --build --config config.yaml
```

Its fields are `model_name`, `max_length`, `hidden_size`, `dropout`,
`warmup_epochs`, `extraction_batch_size`, `warmup_training_batch_size`,
`warmup_learning_rate`, `weight_decay`, `max_grad_norm`, `random_state`,
`dataloader_workers`, and `A/B_swap`. All except `A/B_swap` are required;
`A/B_swap` defaults to `aug`. Validation follows the corresponding LoRA fields,
except standalone `warmup_epochs` must be strictly positive. Adapter sections
continue to allow zero epochs to disable warmup and its cache entirely.

The root preset copies the LoRA warmup settings: Qwen3-4B, `max_length: 13514`,
a 512-unit head with dropout 0.1, one warmup epoch, extraction batch size 1,
training batch size 1024, learning rate 0.001, weight decay 0.01, clipping 1.0,
seed 42, zero workers, and `A/B_swap: aug`. These are explicit preset values,
not parser defaults. Shared GPU settings control its single training device
and precision.

To reuse the result in LoRA or QLoRA, their warmup compatibility fields and
actual training IDs must match. Changing only the standalone section does not
override either adapter section. Full-data builds use a different training ID
set and do not reuse this train-partition cache. See
[warmup compatibility](caching.md#loraqlora-head-warmup).

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
an immutable revision for reproducible counts. Both active presets pin
`Qwen/Qwen3-4B` to `1cfa9a7208912126459214e8b04321603b3df60c`.

These settings are independent of the optional training model sections. When
using a different training tokenizer, update the balance tokenizer as well.
Missing tokenizer files may be downloaded on preparation. If equal-class group
swaps cannot reach the target within 1,000 exchanges, preparation fails before
replacing the processed dataset. See [data preparation](data-preparation.md).

For LoRA and QLoRA, `head_inference_batch_size` is a required positive integer replacing
`inference_batch_size`; rename the old key while preserving its value. The old
key is rejected. `extraction_batch_size` controls backbone extraction for
warmup, every joint-validation pass, and submissions, including when warmup is
disabled. It counts individual A/B branches per backbone forward, matching the
frozen model's `extraction_batch_size`; the standalone warmup section uses the
same definition. Head inference uses its own batch size after extraction completes.
`gradient_accumulation_steps` applies only to joint training; warmup performs one
optimizer update per `warmup_training_batch_size` batch.

## GPU configuration

Holdout validation and `--build` use the single CUDA device selected by
the required `gpu.device` field. The root preset selects `cuda:0`.
All models use one backbone for A and B; classifier head training uses the
same GPU. LoRA and QLoRA joint training encode A and B sequentially. Their
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
`gpu.precision: bf16`, LoRA/QLoRA use BF16 head computation and pooled/CPU
features, returning FP32 logits; the frozen `qwen3-4b` head retains FP32
computation. LoRA gradient checkpointing
is disabled for submissions. Both GPUs must be available and support
`gpu.precision`; there is no automatic precision change, single-GPU fallback, or
CPU/disk offload. `ALL` still runs models sequentially, with each model split
across both GPUs.

QLoRA uses `gpu.precision` as its quantized-layer compute dtype during training
and validation. Its submissions explicitly load a non-quantized base at that
precision and restore the saved adapter. NF4 and double quantization are fixed
profile behavior; see [QLoRA training](qlora.md).
In BF16 mode, QLoRA also restores frozen embedding and RMSNorm weights to BF16
after k-bit preparation, preserving BF16 normalization outputs and residuals.

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
selectors are `qwen3-4b`, `qwen3-4b-lora`, and `qwen3-4b-qlora`. Config files run in filename order,
with the selected models running sequentially for each config. The scripts are
independent: the build script does not run or require holdout validation.
They locate `configs/` relative to the script's project root and run each model
from that root, regardless of the directory from which the script was invoked.

Each model's validation run publishes its own `metrics.json`; `config_total_time`
covers only that model's holdout validation run under the selected config.
The full-data build does not produce `metrics.json`. By default, it writes
`models/qwen3_4b_lora/adapter_CONFIG_STEM.pt` for each config; selecting `qwen3-4b`
also writes `models/qwen3_4b/head_CONFIG_STEM.pt`; selecting `qwen3-4b-qlora` writes
`models/qwen3_4b_qlora/adapter_CONFIG_STEM.pt`. Before starting any builds,
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
