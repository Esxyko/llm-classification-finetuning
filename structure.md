# Project Structure

## Root files

- `.env`: Gitignored local Kaggle credentials loaded by the CLI.
- `.env.example`: Safe, tracked template for configuring Kaggle credentials.
- `config.yaml`: Runtime configuration for data locations and cross-validation.
- `pyproject.toml` and `uv.lock`: Package metadata and locked dependencies.
- `data/`: Ignored local storage for downloaded raw files and generated outputs.

## Package responsibilities

- `cli.py`: Defines the complete initialization CLI.
- `config.py`: Loads and validates YAML configuration into immutable value
  objects.
- `data/`: Owns raw competition data acquisition.
  - `cli.py`: Defines the standalone raw-data download CLI.
  - `downloader.py`: Ensures official Kaggle files are locally available,
    normalizes ZIP-wrapped responses, and avoids overwriting raw data.
- `preprocess/`: Owns transformations that prepare raw data for training.
  - `augmentation.py`: Appends an A/B-swapped counterpart to each folded row.
  - `cli.py`: Defines the standalone fold and augmentation preprocessing CLI.
  - `folds.py`: Derives labels and canonical prompt groups, assigns stratified
    group folds, applies A/B augmentation, and writes the Parquet artifact.
- `pipeline.py`: Coordinates download and preprocessing and reports summary
  statistics.
- `errors.py`: Defines expected domain errors shared by the pipeline layers.
