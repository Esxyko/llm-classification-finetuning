# Command reference

[Documentation index](../README.md)

Run commands from the project root after [setup](getting-started.md).
All commands accept [`--config PATH`](configuration.md#configuration-files-and-paths).

| Command | What it does | Main output |
| --- | --- | --- |
| `uv run init` | Downloads missing raw files and prepares training data | `data/processed/train_validation.parquet` |
| `uv run data` | Downloads missing Kaggle files only | `data/raw/*.csv` |
| `uv run preprocess` | Rebuilds training data from `data/raw/train.csv` | `data/processed/train_validation.parquet` |
| `uv run model qwen3-4b` | Runs Qwen3-4B holdout validation | `results/qwen3-4b-*/` |
| `uv run model qwen3-4b-lora-EXPT2` | Runs LoRA with the equivariant head deriving ties from win representations | `results/qwen3-4b-lora-EXPT2-*/` |
| `uv run model SR2-Qwen3-4B-lora` | Runs Skywork reward-backbone LoRA with the EXPT2 head | `results/SR2-Qwen3-4B-lora-*/` |
| `uv run model warmup --build` | Builds non-quantized EXPT2 head warmup on the saved train partition | `data/processed/qwen3_4b_lora_EXPT2_warmup/<key>.pt` |
| `uv run model ALL` | Runs configured models sequentially | Each model's results directory |
| `uv run model MODEL --build` | Trains the classifier and any adapters on all training and validation IDs | `models/MODEL_SLUG/head.pt` or `adapter.pt` |
| `uv run model MODEL --submit` | Creates a submission with the selected model's checkpoint | `results/submit/MODEL-*/submission.csv` |
| `uv run result [SUBFOLDER]` | Builds reports from a validation run | `results/comprehensive/` |

Processed Parquet outputs use the listed logical path. Files at or above 100 MiB
are stored as a manifest plus volumes of at most 99 MiB and loaded automatically;
see [processed data](data-preparation.md#processed-data).

## Model selection and execution modes

Set `MODEL` to `qwen3-4b`, `qwen3-4b-lora-EXPT2`, `SR2-Qwen3-4B-lora`, `warmup`, or exact
uppercase `ALL`. `--build` and `--submit` are mutually exclusive; without either
flag, registered profiles run holdout validation. `ALL` runs configured profiles
in the order `qwen3-4b`, `qwen3-4b-lora-EXPT2`, then `SR2-Qwen3-4B-lora`, and accepts the same
`--build`, `--submit`, and `--refresh-cache` options. If one model fails, the
remaining models still run; the command exits with status `1` if any failed.
YAML validation happens before this loop; an invalid field in any present
section stops the command before any model runs.
The `qwen3_4b_lora_EXPT2` and `SR2_Qwen3_4B_lora` sections enable their respective
profiles in `ALL`. Both LoRA profiles and standalone warmup ignore `A/B_swap`
and always use original rows. The standard LoRA and
EXPT selectors are [retired](models.md#retired-profiles).

The old `--test` flag is no longer accepted.

`warmup` requires `--build` and its own [configuration section](configuration.md#standalone-warmup).
It trains only on the saved train partition, using a non-quantized frozen
backbone without adapters. It uses the EXPT2 head and writes its reusable warmup cache, then stops
before joint training, with no validation, production checkpoint, or report.
`ALL` never includes this CLI-only selector. `warmup` accepts `--config` and
`--refresh-cache`; it rejects validation mode, `--submit`, and `--checkpoint-tag`
before loading data or configuring a GPU. A cache hit skips backbone-weight
loading and reports zero executed epochs. This command writes the EXPT2 namespace;
SR2 builds its separate warmup cache during its own validation or full-data build.
See [cache reuse](caching.md#standalone-warmup-build).

## Cache refresh

Add `--refresh-cache` to rebuild requested shared token encodings and
model-specific frozen embeddings or LoRA batch plans and head-warmup
states. For adapter holdout validation/builds, this forces warmup even when a
compatible completed state exists. Token encodings are shared
by all models and by training, validation, and test inference. See
[caching](caching.md) for reuse and compatibility rules.

For `warmup --build`, refresh rebuilds train-partition LoRA warmup and its
requested tokens.

## Checkpoint selection

For builds and submission inference, `--checkpoint-tag TAG` selects
`models/MODEL_SLUG/head_TAG.pt` for the frozen model or `adapter_TAG.pt` for
LoRA; without it, the checkpoint is `head.pt` or `adapter.pt`, respectively.
Tags must begin with an ASCII letter or digit and then contain only ASCII
letters, digits, underscores, or hyphens. `--checkpoint-tag` is rejected during
holdout validation.

See [model workflows](models.md), [LoRA training](lora.md), and
[validation reports](reports.md) for detailed behavior.
