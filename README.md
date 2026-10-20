# LLM Classification Finetuning

Prepare leakage-safe training folds, run frozen or LoRA Qwen pairwise models, and
generate reports or submissions for Kaggle's
[LLM Classification Finetuning competition](https://www.kaggle.com/competitions/llm-classification-finetuning).

## Requirements

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A Kaggle account with the competition rules accepted
- A configured CUDA GPU to run a model (data preparation and report generation
  do not require one)

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
| `uv run model qwen3-4b-lora` | Runs Qwen3-4B LoRA cross-validation | `results/qwen3-4b-lora-*/` |
| `uv run model ALL` | Runs configured models sequentially | Each model's results directory |
| `uv run model MODEL --build` | Trains the classifier and any adapters on all folds | `models/MODEL_SLUG/head.pt` or `adapter.pt` |
| `uv run model MODEL --test` | Creates a submission with the selected model's checkpoint | `results/test/MODEL-*/submission.csv` |
| `uv run result [SUBFOLDER]` | Builds reports from a validation run | `results/comprehensive/` |

Set `MODEL` to `qwen3-4b`, `qwen3-4b-lora`, or exact uppercase `ALL`.
Add `--refresh-cache` to rebuild frozen-model embeddings; it has no effect for
LoRA, which encodes inputs during training and inference.
`--build` and `--test` are mutually exclusive; without either flag, the command
runs cross-validation. `ALL` runs only profiles with a model section in the
selected YAML, follows registry order, and accepts the same
`--build`, `--test`, and `--refresh-cache` options. If one model fails, the
remaining models still run; the command exits with status `1` if any failed.
For builds and test inference, `--checkpoint-tag TAG` selects
`models/MODEL_SLUG/head_TAG.pt` for the frozen model or `adapter_TAG.pt` for
LoRA; without it, the checkpoint is `head.pt` or `adapter.pt`, respectively.

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

The frozen model's training and test embeddings use separate model-named caches in
`data/processed/`. A cache is reused only when its data, model, sequence length,
serialization, precision, attention implementation, and TF32 settings still
match. Legacy `baseline_*` caches are left untouched and are not reused.

Both models share the single CUDA device selected by `gpu.device`, which defaults
to `cuda:0`. The frozen model loads one backbone and encodes A and B sequentially
on that device; classifier head training uses the same GPU. If extraction runs
out of memory, reduce `qwen3_4b.extraction_batch_size` or `qwen3_4b.max_length`.
Device lists, backbone replicas, and multi-GPU execution are not supported.

## Train with LoRA

Prepare the data, then use the LoRA section in the shared `config.yaml`:

```shell
uv run model qwen3-4b-lora
uv run model qwen3-4b-lora --build
uv run model qwen3-4b-lora --test
```

The model uses one shared `Qwen/Qwen3-4B` backbone to encode prompt+A and
prompt+B separately. LoRA adapters and the pairwise MLP are trained together;
original backbone parameters remain frozen. `A/B_swap: aug` trains on both
processed orientations. Validation loss, fold predictions, and test submissions
average original and A/B-restored swapped probabilities. Each validation fold
starts with fresh adapters and a fresh classifier. Cross-validation saves no
model checkpoints.

The default LoRA settings target one 32 GB CUDA GPU. The model also
supports other single NVIDIA CUDA GPUs with sufficient memory for the selected
settings. Set `gpu.device` to a single `cuda:N` identifier and choose
`gpu.precision: fp16`, `bf16`, or `fp32`. BF16 is checked against the selected
GPU's capabilities. FP16 uses gradient scaling; adapters and classifier weights
remain FP32 in all three modes. GPU model names and VRAM sizes are not restricted.

The default configuration caps **each branch** at 16,384 tokens, dynamically pads
A and B independently, processes branches sequentially, and enables non-reentrant
activation checkpointing with KV caching disabled. It uses micro-batches of one
pair and 16 accumulation steps, giving an effective batch of 16 augmented rows.
LoRA rank is 16, alpha is 32, and dropout is 0.05, targeting attention
`q_proj`, `k_proj`, `v_proj`, `o_proj` and MLP `gate_proj`, `up_proj`, `down_proj`.
The 512-unit classifier uses dropout 0.1. AdamW uses adapter learning rate
`0.0002`, head learning rate `0.001`, weight decay `0.01`, and gradient clipping
at 1.0. Training starts with one head-only warmup epoch at a fixed learning rate
of `0.001`, then jointly trains adapters and head for two epochs with learning
rates linearly decaying to zero. All these settings are configurable in
`qwen3_4b_lora`.

`warmup_epochs` is a required nonnegative integer; `0` skips head warmup.
`warmup_training_batch_size` is a required positive integer controlling only the
head-warmup micro-batch size (default preset: `1`); joint training uses
`training_batch_size`. Both stages use `gradient_accumulation_steps` and normalize
loss by the actual row count of each accumulation group, including short final
batches. The joint-training decay schedule uses only the joint-training batches.
`warmup_learning_rate` is a required finite positive number, even when warmup is
disabled. During warmup, the backbone runs in evaluation mode without gradient
tracking, while the classifier remains in training mode. Joint training restores
backbone training mode and retains the head's AdamW state. The joint-training
schedule starts at `learning_rate` / `head_learning_rate` without a learning-rate
ramp. Total training lasts `warmup_epochs + epochs` rounds (three by default),
and training histories and build summaries include both stages. Each validation
fold and full-data build follows these stages; validation still runs after each
epoch. Replace the removed `warmup_ratio` key with `warmup_epochs`,
`warmup_training_batch_size`, and `warmup_learning_rate`; old keys are rejected
as unknown configuration options.

For smaller GPUs, lower `max_length` or the relevant `warmup_training_batch_size`,
`training_batch_size`, or `inference_batch_size`. Increase
`gradient_accumulation_steps` when reducing the training batch to retain the
desired effective batch. Larger GPUs can use larger batches. Memory fit for the
16,384-token preset on a 32 GB GPU has not been measured; execution reports OOM
without silently lowering context length.

Builds atomically save adapters, the classifier, and compatibility metadata in
`models/qwen3_4b_lora/adapter.pt`. `--checkpoint-tag TAG` selects
`adapter_TAG.pt`. Base weights and optimizer state are not stored. The checkpoint
records the resolved Hub base revision so inference reloads matching weights
and tokenizer files. Architecture, serialization, LoRA settings, and context
must match; CUDA index, GPU model, precision, attention implementation, TF32,
batch sizes, and accumulation settings can change between build and inference.
There are no persistent LoRA embedding caches.

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
For frozen/LoRA Qwen3-4B and averaged mono-input runs, each ID contributes one prediction,
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

To cross-validate and build a separate checkpoint for every `.yaml` or `.yml` file in
`configs/`, run:

```shell
uv run python scripts/build_all_config.py
```

The script runs cross-validation and then a full-data build for each config,
using only its configured model sections. `config.yaml`, `config_aug.yaml`, and
`config_inf.yaml` each contain both `qwen3_4b` and `qwen3_4b_lora`; `model ALL`
runs them sequentially on the shared single GPU. The aug/inf configurations
select the frozen model's A/B policy; LoRA uses `A/B_swap: aug` in both.

For each file in `configs/`, the script writes
`models/qwen3_4b/head_CONFIG_STEM.pt` and
`models/qwen3_4b_lora/adapter_CONFIG_STEM.pt`. It continues after a failed mode
or config and exits with a nonzero status if any step fails. To use a tagged
checkpoint for test inference, pass its config and matching checkpoint tag, for
example:

```shell
uv run model qwen3-4b --test --config configs/config_inf.yaml --checkpoint-tag config_inf
```

| Section | Controls |
| --- | --- |
| `data` | Competition name and raw/processed paths |
| `cross_validation` | Fold count and random seed |
| `gpu` | One CUDA `device`, precision, attention implementation, and TF32, shared by both models |
| `qwen3_4b` | Qwen3-4B model, sequence length, batches, MLP, optimizer, and `A/B_swap` training mode (`aug` or `inf`) |
| `qwen3_4b_lora` | Live LoRA training, per-branch sequence length, training/inference batches, accumulation, checkpointing, adapter and classifier hyperparameters; `A/B_swap: aug` |

Model sections are optional. Explicitly selecting a profile without its section
reports a configuration error; `ALL` requires at least one configured profile.

## Troubleshooting

- **Kaggle download fails:** Accept the competition rules, verify
  `KAGGLE_API_TOKEN`, and check network access.
- **Configuration file is missing:** Run from the intended working directory or
  pass the correct file path with `--config`.
- **`train.csv` is missing:** Run `uv run data`, or use `uv run init` for both
  download and preprocessing.
- **CUDA or precision is unsupported:** Select an available `cuda:N` device and
  set it in `gpu.device`; use `fp16` or `fp32` if the GPU does not support `bf16`.
- **Embedding extraction runs out of memory:** Reduce
  `extraction_batch_size` or `max_length` in the selected model's section.
- **LoRA runs out of memory:** Lower `qwen3_4b_lora.training_batch_size` or
  `inference_batch_size`; at batch size 1, lower `max_length`. Enable
  `gradient_checkpointing` and use FP16/BF16 instead of FP32 where supported.
