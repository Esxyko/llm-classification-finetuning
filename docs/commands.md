# Command reference

[Documentation index](../README.md)

Run commands from the project root after [setup](getting-started.md).
All commands accept [`--config PATH`](configuration.md#configuration-files-and-paths).

| Command | What it does | Main output |
| --- | --- | --- |
| `uv run init` | Downloads missing raw files and prepares training data | `data/processed/train_folds.parquet` |
| `uv run data` | Downloads missing Kaggle files only | `data/raw/*.csv` |
| `uv run preprocess` | Rebuilds training data from `data/raw/train.csv` | `data/processed/train_folds.parquet` |
| `uv run model qwen3-4b` | Runs Qwen3-4B cross-validation | `results/qwen3-4b-*/` |
| `uv run model qwen3-4b-lora` | Runs Qwen3-4B LoRA cross-validation | `results/qwen3-4b-lora-*/` |
| `uv run model ALL` | Runs configured models sequentially | Each model's results directory |
| `uv run model MODEL --build` | Trains the classifier and any adapters on all folds | `models/MODEL_SLUG/head.pt` or `adapter.pt` |
| `uv run model MODEL --submit` | Creates a submission with the selected model's checkpoint | `results/submit/MODEL-*/submission.csv` |
| `uv run result [SUBFOLDER]` | Builds reports from a validation run | `results/comprehensive/` |

## Model selection and execution modes

Set `MODEL` to `qwen3-4b`, `qwen3-4b-lora`, or exact uppercase `ALL`.
`--build` and `--submit` are mutually exclusive; without either flag, the command
runs cross-validation. `ALL` runs only profiles with a model section in the
selected YAML, runs `qwen3-4b` before `qwen3-4b-lora`, and accepts the same
`--build`, `--submit`, and `--refresh-cache` options. If one model fails, the
remaining models still run; the command exits with status `1` if any failed.
YAML validation happens before this loop; an invalid field in any present
section stops the command before either model runs.
The old `--test` flag is no longer accepted.

## Cache refresh

Add `--refresh-cache` to rebuild requested shared token encodings and
model-specific frozen embeddings or LoRA batch plans. Token encodings are shared
by both models and by training, validation, and test inference. See
[caching](caching.md) for reuse and compatibility rules.

## Checkpoint selection

For builds and submission inference, `--checkpoint-tag TAG` selects
`models/MODEL_SLUG/head_TAG.pt` for the frozen model or `adapter_TAG.pt` for
LoRA; without it, the checkpoint is `head.pt` or `adapter.pt`, respectively.
Tags must begin with an ASCII letter or digit and then contain only ASCII
letters, digits, underscores, or hyphens. `--checkpoint-tag` is rejected during
cross-validation.

See [model workflows](models.md), [LoRA training](lora.md), and
[validation reports](reports.md) for detailed behavior.
