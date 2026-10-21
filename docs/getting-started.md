# Getting started

[Project overview and documentation index](../README.md)

## Requirements

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A Kaggle account with the competition rules accepted
- One CUDA GPU for training and two CUDA GPUs for submission inference (data
  preparation and report generation do not require a GPU)
- QLoRA training requires the installed bitsandbytes CUDA backend to support
  the selected GPU and CUDA runtime; see [QLoRA training](qlora.md).

## Quick start

Run these commands from the project root.

1. Install the locked dependencies:

   ```shell
   uv sync
   ```

2. Create `.env` from the provided template:

   ```powershell
   Copy-Item .env.example .env
   ```

   On macOS or Linux, run `cp .env.example .env` instead. Generate a token from
   [Kaggle API settings](https://www.kaggle.com/settings/api), then set it in
   `.env`:

   ```dotenv
   KAGGLE_API_TOKEN=your_token_here
   ```

   `.env` is ignored by Git. An existing environment variable takes precedence.

3. Download the competition files and prepare the training data:

   ```shell
   uv run init
   ```

   The output is `data/processed/train_validation.parquet`.

## Next steps

- [Prepare training data](data-preparation.md) for the split and augmentation format.
- [Run models](models.md) for holdout validation, builds, and submissions.
- [Command reference](commands.md) for all entry points and execution flags.
- [Configuration](configuration.md) for paths, GPU settings, and batch experiments.
- [Troubleshooting](troubleshooting.md) for setup and runtime failures.
