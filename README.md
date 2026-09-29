# LLM Classification Finetuning

Prepare leakage-safe training folds, run frozen-Qwen pairwise models, and
generate reports or submissions for Kaggle's
[LLM Classification Finetuning competition](https://www.kaggle.com/competitions/llm-classification-finetuning).

## Requirements

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A Kaggle account with the competition rules accepted
- The configured CUDA GPUs to run a model (data preparation and report generation
  do not require them)

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
| `uv run model qwen3-4b` | Runs Qwen3-4B cross-validation | `results/qwen3-4b-*/` |
| `uv run model ALL` | Runs cross-validation for every registered model sequentially | Each model's results directory |
| `uv run model MODEL --build` | Trains the selected model's head on its training rows | `models/MODEL_SLUG/head.pt` |
| `uv run model MODEL --test` | Creates a submission with the selected model's saved head | `results/test/MODEL-*/submission.csv` |
| `uv run result [SUBFOLDER]` | Builds reports from a validation run | `results/comprehensive/` |

Set `MODEL` to `qwen3-4b` or exact uppercase `ALL`. Add `--refresh-cache` to any
model mode to rebuild that model and mode's embeddings.
`--build` and `--test` are mutually exclusive; without either flag, the command
runs cross-validation. `ALL` follows model registry order and accepts the same
`--build`, `--test`, and `--refresh-cache` options. If one model fails, the
remaining models still run; the command exits with status `1` if any failed.
For builds and test inference, `--checkpoint-tag TAG` selects
`models/MODEL_SLUG/head_TAG.pt`; without it, the checkpoint is `head.pt`.

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

## Run a model

Prepare the data first, then run cross-validation:

```shell
uv run model qwen3-4b
```

The frozen `Qwen/Qwen3-4B` backbone produces cached embeddings; only the
pairwise MLP head is trained. A validation run writes one competition-format
CSV per fold plus `metrics.json` to a timestamped model-named results directory.
Set `qwen3_4b.A/B_swap` to `aug` (the default) to train on both original and
A/B-swapped rows, or `inf` to train on original IDs using the loss of their
averaged original and swapped probabilities. Both modes average the two
orientations for validation predictions and loss, and for test submissions.
Changing modes requires rebuilding the saved head; embedding caches are shared.

To train a production head and generate a submission:

```shell
uv run model qwen3-4b --build
uv run model qwen3-4b --test
```

Build mode replaces the saved classifier head and compatibility metadata. Test
mode validates `test.csv` against `sample_submission.csv`, loads that head, and
writes a timestamped submission. The Qwen backbone is not stored in the
checkpoint.

The previous `qwen3-1.7b` and `qwen3-1.7b-mono-input` implementations and their
configuration sections are retained locally in
`src/llm_classification_finetuning/models/archive/`. This directory is ignored
by Git, and these models are no longer CLI choices.

Each model's training and test embeddings use separate model-named caches in
`data/processed/`. A cache is reused only when its data, model, sequence length,
serialization, precision, attention implementation, and TF32 settings still
match. Legacy `baseline_*` caches are left untouched and are not reused.

The checked-in defaults target two 16 GB NVIDIA T4 GPUs. Frozen Qwen replicas
split tokenized inputs across `cuda:0` and `cuda:1`; classifier head training
runs on the first configured device. T4 requires `fp16`, and the default
16,384-token cap keeps one input within each GPU's memory budget. If extraction
runs out of memory, reduce the selected model section's `extraction_batch_size`
or `max_length` in the selected configuration file.

## Generate validation reports

Create reports for the latest validation run:

```shell
uv run result
```

Or select a direct subfolder of `results/`:

```shell
uv run result qwen3-4b-YYYYMMDD-HHMMSS
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

Both report files are generated in a staging directory before replacing
`results/comprehensive/`. A staging or promotion failure leaves the previous
report in place or restores it. If restoration fails, the command reports the
path of the retained backup. If backup cleanup fails after promotion, the new
report remains available and the command reports the backup path. Directory
replacement uses two renames, so concurrent readers can briefly see the
directory as unavailable.

| File | Contents |
| --- | --- |
| `confusion_matrix.png` | 3 × 3 counts and expected-class row percentages, after restoring swapped predictions to the original A/B orientation |
| `records.xlsx` | One row per source ID with fold, expected and actual labels, incorrect count, and log loss |

The workbook supports filtering; its `AVG` row recalculates over visible rows.
For Qwen3-4B and averaged mono-input runs, each ID contributes one prediction,
one confusion matrix count, and the log loss of its averaged distribution.
Older runs retain their two-orientation report behavior.

## Configuration

All commands read `config.yaml` from the working directory by default. Pass a
complete YAML file with `--config PATH` to `init`, `data`, `preprocess`, `model`,
or `result`. Relative config-file paths and relative `data.raw_dir` and
`data.processed_path` values resolve from the working directory; absolute paths
stay absolute. Model artifacts and reports also remain under the working
directory. `init` and `data` load `.env` from the working directory.

For example, from the project root:

```shell
uv run model qwen3-4b --config configs/experiment.yaml
```

To cross-validate and build a separate head for every `.yaml` or `.yml` file in
`configs/`, run:

```shell
uv run python scripts/run_all_configs.py
```

The script runs cross-validation and then a full-data build for each config.
The current `config_aug.yaml` and `config_inf.yaml` write
`models/qwen3_4b/head_config_aug.pt` and
`models/qwen3_4b/head_config_inf.pt`, respectively. It continues after a failed
mode or config and exits with a nonzero status if any step fails. To use one of
these heads for test inference, pass its config and matching checkpoint tag, for
example:

```shell
uv run model qwen3-4b --test --config configs/config_inf.yaml --checkpoint-tag config_inf
```

| Section | Controls |
| --- | --- |
| `data` | Competition name and raw/processed paths |
| `cross_validation` | Fold count and random seed |
| `gpu` | CUDA devices, precision, attention implementation, and TF32 |
| `qwen3_4b` | Qwen3-4B model, sequence length, batches, MLP, optimizer, and `A/B_swap` training mode (`aug` or `inf`) |

## Troubleshooting

- **Kaggle download fails:** Accept the competition rules, verify
  `KAGGLE_API_TOKEN`, and check network access.
- **Configuration file is missing:** Run from the intended working directory or
  pass the correct file path with `--config`.
- **`train.csv` is missing:** Run `uv run data`, or use `uv run init` for both
  download and preprocessing.
- **CUDA or precision is unsupported:** Select an available `cuda:N` device and
  use `fp16` or `fp32` if the GPU does not support `bf16`.
- **Embedding extraction runs out of memory:** Reduce
  `extraction_batch_size` or `max_length` in the selected model's section.
