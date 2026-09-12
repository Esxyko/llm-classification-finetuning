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
| `uv run baseline` | Train frozen-Qwen pairwise heads and write k-fold predictions |
| `uv run result [SUBFOLDER]` | Synthesize k-fold prediction CSVs into comprehensive reports |

Run commands from the project root. They read [`config.yaml`](config.yaml), which
controls data paths, folds, GPU resources, and baseline hyperparameters. Relative
paths are resolved from that file's location.

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

## Qwen3 pairwise baseline

Run the complete five-fold baseline after preparing the processed dataset:

```shell
uv run baseline
```

Build one production head from all original and A/B-swapped rows across every
fold:

```shell
uv run baseline --build
```

Build mode does not generate validation results. It atomically replaces the
trained MLP parameters and compatibility metadata in
`models/baseline/head.pt`. The Qwen backbone remains frozen and is not copied
into the checkpoint.

Generate a competition submission from the saved head without further
training:

```shell
uv run baseline --test
```

Test mode validates `data/raw/test.csv` against `sample_submission.csv`, loads
the compatible checkpoint, and publishes
`results/test/baseline-YYYYMMDD-HHMMSS/submission.csv`. `--build` and `--test`
are mutually exclusive. With neither option, cross-validation remains the
default.

The first run downloads `Qwen/Qwen3-1.7B`, encodes each canonical response pair
with the same frozen backbone, and caches the pooled states in `data/processed/`.
Later runs reuse that cache when the processed dataset, model, sequence length,
serialization format, precision, attention implementation, and TF32 setting still
match. Training and test data use separate embedding caches. Add
`--refresh-cache` to any mode to force extraction for that mode.

Each branch is built from the aligned JSON conversation turns. A turn is formatted
as the prompt, a newline, and that branch's response; turns are separated with a
blank line. The tokenizer truncates on the right to the configured maximum length.
The last non-padding state from each branch is combined as
`[h_A, h_B, h_A - h_B, h_A * h_B]` and passed to an MLP trained with cross-entropy.
Only the MLP parameters are trained.

Successful runs are atomically published as
`results/baseline-YYYYMMDD-HHMMSS/`. The directory contains one competition-format
validation CSV per fold plus `metrics.json`; cross-validation itself does not
save checkpoints. Run the existing synthesizer without an argument to select
the latest validation result (the reserved `results/test/` directory is ignored):

```shell
uv run result
```

The checked-in GPU defaults target one 8 GB CUDA GPU. Adjust `gpu.precision`,
`baseline.extraction_batch_size`, and `baseline.max_length` for the available
hardware. `gpu.device` must identify one CUDA device, such as `cuda:0`; multi-GPU
and CPU training are not supported by this baseline.

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
| `confusion_matrix.png` | Raw 3×3 counts and expected-class row percentages after translating swapped predictions back to the original A/B orientation |
| `records.xlsx` | One row per source ID with its fold, canonical expected label, original/translated actual-label pair, 0–2 incorrect count, and pair-averaged log loss |

The workbook is filterable, and its `AVG` row uses Excel's `SUBTOTAL` formula so
the displayed average loss follows the currently visible records.

## Troubleshooting

- **Kaggle download fails:** confirm that you accepted the rules, set a valid
  token, and have network access.
- **`config.yaml` is not found:** run the command from the project root.
- **`train.csv` is not found:** run `uv run data`, or use `uv run init` for both
  stages.
- **CUDA or precision is unsupported:** select an available `cuda:N` device and
  use `fp16` or `fp32` when the GPU does not support `bf16`.
- **Baseline extraction runs out of memory:** reduce
  `baseline.extraction_batch_size` or `baseline.max_length` in `config.yaml`.
