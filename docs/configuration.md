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
| `cross_validation` | Fold count and random seed |
| `gpu` | Training `device`, two `submit_devices` (default `[cuda:0, cuda:1]`), precision, attention implementation, and TF32, shared by both models |
| `qwen3_4b` | Qwen3-4B model, sequence length, batches, MLP, optimizer, and `A/B_swap` training mode (`aug` or `inf`) |
| `qwen3_4b_lora` | Live LoRA training, per-branch sequence length, training/inference batches, optional `training_token_budget`, accumulation, checkpointing, adapter and classifier hyperparameters; `A/B_swap` training mode (`aug` or `inf`) |

Model sections are optional. Explicitly selecting a profile without its section
reports a configuration error; `ALL` requires at least one configured profile.

Every command validates the entire selected YAML before doing work, including
any model sections it will not use. `data`, `cross_validation`, and `gpu` are
required even for data preparation and reporting. Each model section, when
present, must include all its required fields; config files do not inherit or
merge values from the root `config.yaml`.

Only these YAML fields have parser defaults when omitted:

| Field | Default |
| --- | --- |
| `gpu.submit_devices` | `[cuda:0, cuda:1]` |
| `qwen3_4b.A/B_swap` / `qwen3_4b_lora.A/B_swap` | `aug` |
| `qwen3_4b_lora.training_token_budget` | `0` |

All other fields in a present section are required, including `gpu.device`,
`warmup_epochs`, and the warmup batch sizes and learning rate. Unknown keys are
rejected. Numeric values described as presets in these guides refer to the
root [`config.yaml`](../config.yaml); files in `configs/` can use different values.

## GPU configuration

Cross-validation and `--build` use the single CUDA device selected by
the required `gpu.device` field. The root preset selects `cuda:0`.
Both models use one backbone for A and B; classifier head training uses the
same GPU. LoRA encodes A and B sequentially, while frozen extraction batches
unique missing branches from either side together.

`--submit` splits that backbone across the two devices in `gpu.submit_devices`,
which defaults to `[cuda:0, cuda:1]` when omitted. The list must contain exactly
two distinct `cuda:N` identifiers; it is independent of the training device.
The root preset's GPU configuration is:

```yaml
gpu:
  device: cuda:0
  submit_devices: [cuda:0, cuda:1]
  precision: fp16
  attention_implementation: sdpa
  allow_tf32: false
```

The embedding and rotary modules and the first half of the decoder layers reside
on the first GPU; the remaining layers and final normalization reside on the
second. Each decoder layer stays intact, LoRA adapters follow their base layers,
and the FP32 classifier head resides on the first GPU. LoRA gradient checkpointing
is disabled for submissions. Both GPUs must be available and support
`gpu.precision`; there is no automatic precision change, single-GPU fallback, or
CPU/disk offload. `ALL` still runs models sequentially, with each model split
across both GPUs.

`gpu.precision` accepts `fp16`, `bf16`, or `fp32`;
`gpu.attention_implementation` accepts `sdpa` or `eager`. Hardware validation
checks CUDA availability and BF16 support on each selected device. It runs even
when every frozen embedding is cached, so submissions still require two GPUs.

Changing device placement preserves checkpoint and cache compatibility.
Changing precision, attention implementation, or TF32 selects a different frozen
embedding cache and is rejected by a saved frozen-head checkpoint. LoRA
checkpoints allow those compute changes; shared tokens remain reusable when the
tokenizer and encoding options match. See
[frozen checkpoint compatibility](models.md#checkpoint-compatibility),
[LoRA checkpoints](lora.md#checkpoints), and [caching](caching.md) for the exact
boundaries.

Complete frozen embedding-cache hits skip loading backbone weights, while still
loading the model configuration and tokenizer. Submissions are written to
`results/submit/MODEL-*/submission.csv`; historical `results/test` files remain
untouched. For frozen extraction memory errors, see
[troubleshooting](troubleshooting.md).

## Batch experiments

To cross-validate selected models for every `.yaml` or `.yml` file in `configs/`,
run:

```shell
uv run python scripts/validate_all_configs.py
```

To train selected models on all folds and build a separate checkpoint for each
config, run:

```shell
uv run python scripts/build_all_configs.py
```

Both scripts define `MODELS = ("qwen3-4b-lora",)` near the top. Edit each script's
tuple independently to select models and their execution order; supported model
selectors are `qwen3-4b` and `qwen3-4b-lora`. Config files run in filename order,
with the selected models running sequentially for each config. The scripts are
independent: the build script does not run or require cross-validation.
They locate `configs/` relative to the script's project root and run each model
from that root, regardless of the directory from which the script was invoked.

Each model's validation run publishes its own `metrics.json`; `config_total_time`
covers only that model's cross-validation run under the selected config.
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
