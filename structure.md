# Project Structure

## Root files

- `.env`: Gitignored local Kaggle credentials loaded by the CLI.
- `.env.example`: Safe, tracked template for configuring Kaggle credentials.
- `config.yaml`: Runtime configuration for data locations, cross-validation,
  single-GPU resources, and baseline hyperparameters.
- `pyproject.toml` and `uv.lock`: Package metadata and locked dependencies.
- `data/`: Ignored local storage for downloaded raw files and generated outputs.
- `results/`: Per-run fold predictions, test submissions, and comprehensive
  reports.
- `models/`: Ignored generated baseline classifier-head checkpoints.
- `tests/`: Result-synthesis unit and integration coverage.

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
- `baseline/`: Owns the frozen-Qwen pairwise classification baseline.
  - `data.py`: Validates processed folds and competition test inputs and
    serializes aligned conversations.
  - `model.py`: Defines shared-backbone pooling and the pairwise MLP head.
  - `cache.py` and `extractor.py`: Extract, validate, and persist canonical pooled
    embeddings for training or test data.
  - `trainer.py`: Trains fold-specific heads or one head on all folded rows.
  - `checkpoint.py` and `predictor.py`: Persist compatible head parameters and
    run no-gradient test inference.
  - `pipeline.py` and `cli.py`: Coordinate cross-validation, full-data builds,
    test inference, and atomic datetime-named result publication.
- `result/`: Owns validation and synthesis of cross-validation predictions.
  - `cli.py`: Defines the standalone comprehensive-result CLI.
  - `synthesizer.py`: Discovers fold CSVs, validates them against processed
    folds, canonicalizes A/B augmentation, and publishes staged artifacts.
  - `writers.py`: Renders the confusion-matrix PNG and filterable Excel report.
- `pipeline.py`: Coordinates download and preprocessing and reports summary
  statistics.
- `errors.py`: Defines expected domain errors shared by the pipeline layers.
