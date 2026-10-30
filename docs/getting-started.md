# Getting started

[Documentation index](../README.md)

## Requirements

- Python **3.14 or newer**, as declared in `pyproject.toml` and `.python-version`.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
- One CUDA GPU for model validation, builds, or standalone warmup. Data
  preparation and reporting run on the CPU.
- A Kaggle account and accepted competition rules **only when downloading raw
  competition files**. The saved processed split can be used without them.

Run all commands below from the project root. For a remote runtime, start with
[Colab setup](colab.md).

## Install and configure

Install the committed dependency versions:

```shell
uv sync --locked
```

The training lock currently selects PyTorch `2.14.0+cu130` from the CUDA 13.0
index.

Copy `.env.example` to `.env` if you need credentials or custom storage paths:

```powershell
Copy-Item .env.example .env
```

On Linux, use `cp .env.example .env`. Edit the copy before downloading raw data:

```dotenv
KAGGLE_API_TOKEN=your_token_here
MODEL_PATH=
CACHE_PATH=cache
RESULT_PATH=results
```

Get a token from [Kaggle API settings](https://www.kaggle.com/settings/api).
`.env` is ignored by Git. Existing process environment variables take precedence.
`MODEL_PATH` selects the downloaded model cache; `CACHE_PATH` selects training
caches; `RESULT_PATH` selects validation runs, comprehensive reports,
and batch logs, defaulting to `./results` when missing or blank. Relative paths
resolve from the working directory. YAML data paths and production checkpoint
locations are configured independently; changing a path does not move existing
files.
See [path rules](configuration.md#configuration-files-and-paths).

## Choose the data source

| Starting point | Action |
| --- | --- |
| Use the saved split in this checkout | Keep the Parquet manifest and every referenced `.part` file; proceed directly to validation |
| Download raw data and regenerate the split | Run `uv run init` after configuring Kaggle access |
| Raw CSVs are already available | Run `uv run preprocess` |
| Download CSVs without preprocessing | Run `uv run data` |

The saved split's logical path is `data/processed/train_validation.parquet`.
The single `.parquet` file need not exist: loaders reconstruct the versioned
manifest and volumes automatically. Training still downloads the selected
backbone and tokenizer when absent; their weights are not included in Git.

Regenerating data replaces the saved logical artifact after successful checks.
Use the existing split to compare experiments on the same holdout. See
[data preparation](data-preparation.md) for its schema and balance constraints.

## Validate and report

Choose one configured profile, for example SR2:

```shell
uv run model SR2-Qwen3-4B-lora
uv run result
```

Validation writes `validation.csv`, `metrics.json`, and `config.yaml` together
under `results/SR2-Qwen3-4B-lora-<timestamp>/`. `result` selects the most recently
modified eligible run and replaces the report under `results/comprehensive/`.
Pass a run name explicitly when comparing experiments; see [reports](reports.md).

The shipped presets are experiment settings, not guaranteed memory fits for a
particular GPU. Set supported [GPU precision](configuration.md#gpu-configuration)
and use the stage-specific [memory controls](lora.md#memory-tuning) as needed.

Optional `uv run model warmup --build` trains only the SR2 head on the saved train
partition, using Skywork in the shipped presets and the SR2 warmup namespace.
It is not required before a normal LoRA run. Reuse requires matching
model/head settings and training IDs; see
[standalone warmup](configuration.md#standalone-warmup).

## Hugging Face downloads

Project initialization loads the working-directory `.env` before Hugging Face
imports. The supplied `.env.example` enables `HF_HUB_OFFLINE=1` for execution
from a complete local model cache. Populate that cache in a separate download
process with `HF_HUB_OFFLINE=0` (or use the local `download.ipynb`) before training.
Keep the full native Hub layout, including refs, snapshots, blobs, and all shards.
Offline project loaders set `local_files_only=True`; a missing required file
reports the model, revision, cache/local path, and original error.

Package initialization defaults `HF_ENDPOINT` to `https://hf-mirror.com`, preserving an
existing value. To use the official endpoint, set `HF_ENDPOINT=https://huggingface.co`
in `.env` or the process environment before starting the command.

For the mirror only, initialization defaults `HF_HUB_DISABLE_XET=1` for HTTP
weight downloads. An explicit value takes precedence; other endpoints retain
their existing Xet settings. Standard Hugging Face authentication settings are
preserved. See [model download storage](caching.md#hugging-face-model-downloads).

In a Python notebook, import `llm_classification_finetuning` before `transformers`
or `huggingface_hub` for these defaults to take effect. Restart the kernel if
those libraries were already imported with different settings.

Next: [commands](commands.md), [configuration](configuration.md), or
[troubleshooting](troubleshooting.md).
