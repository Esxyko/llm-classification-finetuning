# LLM Classification Finetuning

Prepare leakage-safe training folds, run a frozen-Qwen pairwise baseline, and
generate reports or submissions for Kaggle's
[LLM Classification Finetuning competition](https://www.kaggle.com/competitions/llm-classification-finetuning).

## Requirements

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A Kaggle account with the competition rules accepted
- One CUDA GPU to run the baseline (data preparation and report generation do not
  require it)

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

   The output is `data/processed/train_folds.parquet`.

## Commands

| Command | What it does | Main output |
| --- | --- | --- |
| `uv run init` | Downloads missing raw files and prepares training data | `data/processed/train_folds.parquet` |
| `uv run data` | Downloads missing Kaggle files only | `data/raw/*.csv` |
| `uv run preprocess` | Rebuilds training data from `data/raw/train.csv` | `data/processed/train_folds.parquet` |
| `uv run baseline` | Runs cross-validation | `results/baseline-*/` |
| `uv run baseline --build` | Trains one head on all prepared rows | `models/baseline/head.pt` |
| `uv run baseline --test` | Creates a submission with the saved head | `results/test/baseline-*/submission.csv` |
| `uv run result [SUBFOLDER]` | Builds reports from a validation run | `results/comprehensive/` |

Add `--refresh-cache` to any `baseline` mode to rebuild that mode's embeddings.
`--build` and `--test` are mutually exclusive; without either flag, the command
runs cross-validation.

## Prepare training data

The preparation pipeline:

1. Downloads `train.csv`, `test.csv`, and `sample_submission.csv` only when they
   are missing.
2. Keeps identical canonical prompts in the same stratified fold to prevent
   leakage.
3. Adds one A/B-swapped copy of every training row to reduce position bias.
4. Writes a Zstandard-compressed Parquet file.

The processed file preserves the source columns and adds:

| Column | Meaning |
| --- | --- |
| `label` | `0` = model A wins, `1` = model B wins, `2` = tie |
| `group_id` | SHA-256 hash of the canonical prompt JSON |
| `fold` | Zero-based validation-fold index |
| `is_swapped` | Whether the row is the generated A/B-swapped copy |

Original and swapped rows share an `id`, `group_id`, and fold, so duplicate IDs
are expected. Re-running preparation preserves existing raw files and replaces
the processed file.

## Run the baseline

Prepare the data first, then run cross-validation:

```shell
uv run baseline
```

The frozen `Qwen/Qwen3-1.7B` backbone produces cached embeddings; only the
pairwise MLP head is trained. A validation run writes one competition-format CSV
per fold plus `metrics.json` to a timestamped `results/baseline-*/` directory.

To train a production head on all prepared rows and generate a submission:

```shell
uv run baseline --build
uv run baseline --test
```

Build mode replaces the saved classifier head and compatibility metadata. Test
mode validates `test.csv` against `sample_submission.csv`, loads that head, and
writes a timestamped submission. The Qwen backbone is not stored in the
checkpoint.

Training and test embeddings use separate caches in `data/processed/`. A cache
is reused only when its data, model, sequence length, serialization, precision,
attention implementation, and TF32 settings still match.

The checked-in defaults target two 16 GB NVIDIA T4 GPUs. Frozen Qwen replicas
split the flattened response branches across `cuda:0` and `cuda:1`; classifier
head training runs on the first configured device. T4 requires `fp16`, and the
default 16,384-token cap keeps one branch within each GPU's memory budget. If
extraction runs out of memory, reduce `baseline.extraction_batch_size` or
`baseline.max_length` in `config.yaml`.

## Generate validation reports

Create reports for the latest validation run:

```shell
uv run result
```

Or select a direct subfolder of `results/`:

```shell
uv run result baseline-YYYYMMDD-HHMMSS
```

Without an argument, the command selects the most recently modified result
subfolder, excluding `comprehensive` and `test`.

The selected folder must contain exactly one CSV per configured fold. Each CSV
must have these columns in order:

```text
id,winner_model_a,winner_model_b,winner_tie
```

Rows must follow the corresponding fold's exact ID order in
`train_folds.parquet`. Filenames do not need to include the fold number because
the command infers it from that sequence.

Reports replace the previous contents of `results/comprehensive/` only after
both new files are generated successfully:

| File | Contents |
| --- | --- |
| `confusion_matrix.png` | 3 × 3 counts and expected-class row percentages, after restoring swapped predictions to the original A/B orientation |
| `records.xlsx` | One row per source ID with fold, expected and actual labels, incorrect count (`0`–`2`), and pair-averaged log loss |

The workbook supports filtering; its `AVG` row recalculates over visible rows.

## Configuration

All commands read `config.yaml` from the project root. Relative data paths are
resolved from that file's directory.

| Section | Controls |
| --- | --- |
| `data` | Competition name and raw/processed paths |
| `cross_validation` | Fold count and random seed |
| `gpu` | CUDA devices, precision, attention implementation, and TF32 |
| `baseline` | Model, sequence length, batch sizes, MLP, and optimizer settings |

## Troubleshooting

- **Kaggle download fails:** Accept the competition rules, verify
  `KAGGLE_API_TOKEN`, and check network access.
- **`config.yaml` is missing:** Run the command from the project root.
- **`train.csv` is missing:** Run `uv run data`, or use `uv run init` for both
  download and preprocessing.
- **CUDA or precision is unsupported:** Select an available `cuda:N` device and
  use `fp16` or `fp32` if the GPU does not support `bf16`.
- **Embedding extraction runs out of memory:** Reduce
  `baseline.extraction_batch_size` or `baseline.max_length`.
