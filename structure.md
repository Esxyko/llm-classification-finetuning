# Codebase Reference

Use this file to locate ownership before changing code. Keep orchestration in
pipelines, user interaction in CLI modules, and domain logic in focused classes.

## Runtime flow

```text
selected YAML config (defaults to config.yaml)
  -> init -> Kaggle download -> fold assignment + A/B augmentation
     -> data/processed/train_folds.parquet

train_folds.parquet
  -> model MODEL (default) -> fold CSVs + metrics -> result -> PNG + XLSX report
  -> model MODEL --build -> models/MODEL_SLUG/head.pt
  -> model MODEL --build --checkpoint-tag TAG -> models/MODEL_SLUG/head_TAG.pt

head.pt + data/raw/test.csv
  -> model MODEL --test -> results/test/MODEL-.../submission.csv
```

All commands accept `--config PATH` and default to `config.yaml` in the working
directory. Relative config-file and data paths resolve from the working
directory. `init` and `data` read `.env` there; model artifacts and reports are
also rooted there, regardless of the config file's location.

## Entry points

Console scripts are declared in `pyproject.toml`.

| Command | Module | Responsibility |
| --- | --- | --- |
| `init` | `cli.py` | Download missing raw files, then preprocess training data |
| `data` | `data/cli.py` | Download missing Kaggle files only |
| `preprocess` | `preprocess/cli.py` | Rebuild folds and augmentation from local raw data |
| `model` | `models/cli.py` | Select a model, cross-validate, build a head, or run test inference |
| `result` | `result/cli.py` | Reduce fold predictions into comprehensive reports |

CLI modules parse arguments, translate expected domain errors into exit code `1`,
and print summaries. Business logic belongs below the CLI layer.

## Package map

`src/llm_classification_finetuning/`

- `config.py`: Immutable config value objects, strict YAML validation, and path
  resolution.
- `cli_config.py`: Defines the shared `--config` argument for all CLI parsers.
- `errors.py`: Shared expected-error hierarchy rooted at `DataPreparationError`.
- `pipeline.py`: Orchestrates the complete download-and-preprocess workflow and
  returns both download and fold results without printing them.
- `cli_output.py`: Formats shared download and fold summaries for CLI commands.
- `data/`
  - `downloader.py`: Downloads the three required Kaggle CSVs, normalizes
    single-file ZIP responses, and preserves existing raw files.
- `preprocess/`
  - `folds.py`: Derives labels and canonical prompt hashes, assigns stratified
    group folds, invokes augmentation, and atomically writes Parquet.
  - `augmentation.py`: Appends one A/B-swapped counterpart per source row.
- `models/`
  - `__init__.py`: Registers each selector with its profile and pipeline.
  - `cli.py`: Selects a registered Qwen model and an execution mode.
  - `profile.py`: Maps selectors to configuration and artifact namespaces.
  - `qwen3_4b/`: Active lightweight model profile definition.
  - `archive/`: Git-ignored local copies of the retired Qwen3-1.7B model
    implementations and their configuration sections. Nothing imports these
    models from the active package.
  - `common/pipeline.py`: Selects cross-validation, build, or test behavior for
    pairwise models.
  - `common/`
    - `inputs.py`: Validates training and test tables, IDs, folds, orientation
      pairs, and source fingerprints for both model input formats.
    - `data.py`: Serializes aligned conversation branches for pairwise models.
    - `compatibility.py`: Builds shared cache and checkpoint compatibility keys.
    - `publishers.py`: Stages and publishes validation runs and test submissions.
    - `hardware.py`: Validates configured CUDA devices and selects the primary GPU.
    - `extractor.py`: Produces frozen-Qwen pooled embeddings across GPU replicas.
    - `cache.py`: Validates and persists training or test embedding caches.
    - `model.py`: Defines backbone pooling and the pairwise MLP head.
    - `orientation.py`: Averages original and swapped class probabilities and
      aligns validation predictions with processed fold rows.
    - `trainer.py`: Trains fold-specific heads or one full-data head.
    - `checkpoint.py`: Saves and validates the production head and compatibility
      metadata.
    - `predictor.py`: Runs no-gradient inference from a saved head.
- `result/`
  - `synthesizer.py`: Discovers and validates fold CSVs, reverses swapped-row
    orientation, and reduces each pair to one source record.
    Averaged Qwen3-4B and mono-input runs count one prediction per pair; older
    runs retain separate original and swapped votes.
  - `publisher.py`: Stages both report files, replaces the report directory,
    and restores the previous report on a publication failure.
  - `writers.py`: Renders the confusion-matrix PNG and filterable Excel workbook.

Each package `__init__.py` exposes its intended public surface. Prefer those
exports over importing private helpers across package boundaries.

## Files and generated artifacts

| Path | Role |
| --- | --- |
| `config.yaml` | Data paths, folds, GPU settings, and per-model hyperparameters |
| `.env.example` | Template for the local Kaggle token in `.env` |
| `pyproject.toml` / `uv.lock` | Package metadata, commands, and locked dependencies |
| `README.md` | Human setup and usage guide |
| `scripts/run_all_configs.py` | Cross-validate and build `model ALL` for each YAML file in `configs/`, with separate tagged heads |
| `configs/` | Complete YAML configurations for sequential model runs |
| `tests/` | Automated coverage, currently focused on result synthesis |
| `data/raw/` | Downloaded competition CSVs |
| `data/processed/` | Folded Parquet data and embedding caches |
| `models/qwen3_4b/` | Saved classifier head and metadata for the active model |
| `results/` | Validation runs, test submissions, and comprehensive reports |

`data/`, `models/`, and `results/` are runtime artifacts rather than source
modules. Do not make application logic depend on a particular timestamped run.

## Invariants to preserve

- Rows with the same canonical prompt must stay in one fold.
- Every source row has exactly one swapped row with the same `id`, `group_id`, and
  `fold`; A/B fields and labels `0`/`1` are reversed, while label `2` is unchanged.
- Prediction CSVs use `id`, `winner_model_a`, `winner_model_b`, and `winner_tie`
  in that order, with finite probabilities that sum to one.
- Every model's training and test caches remain separate and are reused only when
  their source fingerprint and model/runtime compatibility keys match.
- Mono-input rows are serialized as an instruction followed by an ordered JSON
  turn list; original and swapped orientations have distinct cached states.
  Overlong inputs use equal token caps for final responses and remove earlier
  whole turns only when needed to preserve the final prompt, response prefixes,
  and valid JSON.
- Mono-input head training uses original rows only. Validation and test inference
  average original and A/B-restored swapped probabilities per source ID.
- Qwen3-4B `A/B_swap: aug` trains on both orientations with per-row loss;
  `A/B_swap: inf` optimizes the averaged probability of both orientations for
  each original ID. Both modes average probabilities for validation and test.
- Full-data builds save only the classifier head; cross-validation does not save
  checkpoints.
- Generated files and directories are staged before publication. Report
  publication restores the previous directory after a recoverable promotion
  failure and reports the backup path if restoration fails.
