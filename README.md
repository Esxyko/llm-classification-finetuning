# LLM Classification Finetuning

Prepare leakage-safe, augmented training data for Kaggle's
[LLM Classification Finetuning competition](https://www.kaggle.com/competitions/llm-classification-finetuning).

## Prerequisites

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A Kaggle account with the competition rules accepted

## Quick start

1. Install the locked dependencies:

   ```shell
   uv sync
   ```

2. Create a local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

   On macOS or Linux, use `cp .env.example .env`. Generate a token from
   [Kaggle API settings](https://www.kaggle.com/settings/api) and add it to `.env`:

   ```dotenv
   KAGGLE_API_TOKEN=your_token_here
   ```

   `.env` is ignored by Git, and existing environment variables take precedence.

3. Run the complete pipeline from the project root:

   ```shell
   uv run init
   ```

The generated dataset is written to `data/processed/train_folds.parquet`.

## Commands

| Command | Purpose |
| --- | --- |
| `uv run init` | Download missing data, then preprocess the training set |
| `uv run data` | Download only the missing Kaggle files |
| `uv run preprocess` | Rebuild the processed dataset from an existing `train.csv` |
| `uv run result [SUBFOLDER]` | Synthesize k-fold prediction CSVs into comprehensive reports |

Run commands from the project root. They read [`config.yaml`](config.yaml), which
controls data paths, the fold count, and the random seed. Relative paths are
resolved from that file's location.

## Processed data

The pipeline:

1. Downloads any missing competition CSV files without replacing existing ones.
2. Groups identical canonical prompts into the same stratified fold.
3. Adds an A/B-swapped copy of each row to reduce position bias.
4. Writes a Zstandard-compressed Parquet dataset.

The output preserves the source columns and adds:

| Column | Meaning |
| --- | --- |
| `label` | `0`: model A wins, `1`: model B wins, `2`: tie |
| `group_id` | SHA-256 identifier derived from the canonical prompt JSON |
| `fold` | Zero-based validation fold index |
| `is_swapped` | Whether the row is the generated A/B-swapped copy |

Source rows and their swapped copies remain in the same fold. Augmentation
doubles the row count but retains each source `id`, so duplicate IDs are expected.
Re-running the pipeline skips existing raw files and rebuilds the processed file.

## Comprehensive results

Place one competition-format validation CSV per fold in a direct subfolder of
`results/`. Each file must contain the augmented fold rows in the same order as
they appear in `data/processed/train_folds.parquet`. Filenames do not need to
identify their folds because the command validates and infers fold membership
from the complete ID sequence.

Run the result command with a specific subfolder:

```shell
uv run result my-training-run
```

If the subfolder is omitted, the command selects the most recently modified
direct child of `results/`, excluding `comprehensive`. It writes these files to
`results/comprehensive/`, replacing the directory's previous contents only
after the new report has been generated successfully:

| File | Contents |
| --- | --- |
| `confusion_matrix.png` | Raw 3×3 counts after translating swapped predictions back to the original A/B orientation |
| `records.xlsx` | One row per source ID with its fold, canonical expected label, original/translated actual-label pair, 0–2 incorrect count, and pair-averaged log loss |

The workbook is filterable, and its `AVG` row uses Excel's `SUBTOTAL` formula so
the displayed average loss follows the currently visible records.

## Troubleshooting

- **Kaggle download fails:** confirm that you accepted the rules, set a valid
  token, and have network access.
- **`config.yaml` is not found:** run the command from the project root.
- **`train.csv` is not found:** run `uv run data`, or use `uv run init` for both
  stages.
