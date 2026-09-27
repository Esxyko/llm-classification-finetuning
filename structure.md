# Codebase Reference

Use this file to locate ownership before changing code. Keep orchestration in
pipelines, user interaction in CLI modules, and domain logic in focused classes.

## Runtime flow

```text
config.yaml
  -> init -> Kaggle download -> fold assignment + A/B augmentation
     -> data/processed/train_folds.parquet

train_folds.parquet
  -> baseline (default) -> fold CSVs + metrics -> result -> PNG + XLSX report
  -> baseline --build -> models/baseline/head.pt

head.pt + data/raw/test.csv
  -> baseline --test -> results/test/.../submission.csv
```

All commands load `config.yaml` from the working directory. Configured relative
paths are resolved from the config file's directory.

## Entry points

Console scripts are declared in `pyproject.toml`.

| Command | Module | Responsibility |
| --- | --- | --- |
| `init` | `cli.py` | Download missing raw files, then preprocess training data |
| `data` | `data/cli.py` | Download missing Kaggle files only |
| `preprocess` | `preprocess/cli.py` | Rebuild folds and augmentation from local raw data |
| `baseline` | `baseline/cli.py` | Cross-validate, build a head, or run test inference |
| `result` | `result/cli.py` | Reduce fold predictions into comprehensive reports |

CLI modules parse arguments, translate expected domain errors into exit code `1`,
and print summaries. Business logic belongs below the CLI layer.

## Package map

`src/llm_classification_finetuning/`

- `config.py`: Immutable config value objects, strict YAML validation, and path
  resolution.
- `errors.py`: Shared expected-error hierarchy rooted at `DataPreparationError`.
- `pipeline.py`: Orchestrates the complete download-and-preprocess workflow.
- `data/`
  - `downloader.py`: Downloads the three required Kaggle CSVs, normalizes
    single-file ZIP responses, and preserves existing raw files.
- `preprocess/`
  - `folds.py`: Derives labels and canonical prompt hashes, assigns stratified
    group folds, invokes augmentation, and atomically writes Parquet.
  - `augmentation.py`: Appends one A/B-swapped counterpart per source row.
- `baseline/`
  - `pipeline.py`: Selects cross-validation, build, or test mode and atomically
    publishes its artifacts.
  - `data.py`: Validates training/test inputs and serializes aligned conversation
    branches.
  - `hardware.py`: Validates configured CUDA devices and selects the primary GPU.
  - `extractor.py`: Produces frozen-Qwen pooled embeddings across GPU replicas.
  - `cache.py`: Validates and persists training or test embedding caches.
  - `model.py`: Defines backbone pooling and the pairwise MLP head.
  - `trainer.py`: Trains fold-specific heads or one full-data head.
  - `checkpoint.py`: Saves and validates the production head and compatibility
    metadata.
  - `predictor.py`: Runs no-gradient inference from a saved head.
- `result/`
  - `synthesizer.py`: Discovers and validates fold CSVs, reverses swapped-row
    orientation, reduces each pair to one source record, and publishes reports.
  - `writers.py`: Renders the confusion-matrix PNG and filterable Excel workbook.

Each package `__init__.py` exposes its intended public surface. Prefer those
exports over importing private helpers across package boundaries.

## Files and generated artifacts

| Path | Role |
| --- | --- |
| `config.yaml` | Data paths, folds, GPU settings, and baseline hyperparameters |
| `.env.example` | Template for the local Kaggle token in `.env` |
| `pyproject.toml` / `uv.lock` | Package metadata, commands, and locked dependencies |
| `README.md` | Human setup and usage guide |
| `tests/` | Automated coverage, currently focused on result synthesis |
| `data/raw/` | Downloaded competition CSVs |
| `data/processed/` | Folded Parquet data and embedding caches |
| `models/baseline/` | Saved classifier head and metadata |
| `results/` | Validation runs, test submissions, and comprehensive reports |

`data/`, `models/`, and `results/` are runtime artifacts rather than source
modules. Do not make application logic depend on a particular timestamped run.

## Invariants to preserve

- Rows with the same canonical prompt must stay in one fold.
- Every source row has exactly one swapped row with the same `id`, `group_id`, and
  `fold`; A/B fields and labels `0`/`1` are reversed, while label `2` is unchanged.
- Prediction CSVs use `id`, `winner_model_a`, `winner_model_b`, and `winner_tie`
  in that order, with finite probabilities that sum to one.
- Training and test caches remain separate and are reused only when their source
  fingerprint and model/runtime compatibility keys match.
- Full-data builds save only the classifier head; cross-validation does not save
  checkpoints.
- Generated files and directories are staged before publication so failed runs do
  not expose partial outputs.
