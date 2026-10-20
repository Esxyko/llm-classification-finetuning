# LLM Classification Finetuning

Prepare leakage-safe training folds, run frozen or LoRA Qwen pairwise models, and
generate reports or submissions for Kaggle's
[LLM Classification Finetuning competition](https://www.kaggle.com/competitions/llm-classification-finetuning).

## Requirements

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A Kaggle account with the competition rules accepted
- One CUDA GPU for training and two CUDA GPUs for submission inference (data
  preparation and report generation do not require a GPU)

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
| `uv run model MODEL --submit` | Creates a submission with the selected model's checkpoint | `results/submit/MODEL-*/submission.csv` |
| `uv run result [SUBFOLDER]` | Builds reports from a validation run | `results/comprehensive/` |

Set `MODEL` to `qwen3-4b`, `qwen3-4b-lora`, or exact uppercase `ALL`.
Add `--refresh-cache` to rebuild requested shared token encodings and
model-specific frozen embeddings or LoRA batch plans. Token encodings are shared
by both models and by training, validation, and test inference.
`--build` and `--submit` are mutually exclusive; without either flag, the command
runs cross-validation. `ALL` runs only profiles with a model section in the
selected YAML, follows registry order, and accepts the same
`--build`, `--submit`, and `--refresh-cache` options. If one model fails, the
remaining models still run; the command exits with status `1` if any failed.
The old `--test` flag is no longer accepted. For builds and submission inference,
`--checkpoint-tag TAG` selects
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
For both frozen and LoRA models (including head warmup), `metrics.json` records
`training_time`, `validation_time`, and `total_time` as `HH:MM:SS` strings rounded
to whole seconds for every epoch and fold. Hours can exceed 24.
Training and validation times measure each phase's elapsed wall time after
waiting for GPU work to finish. Fold totals also include initialization,
prediction alignment, and LoRA fold cleanup; epoch totals include epoch setup.
Top-level `training_time` and `validation_time` sum that model's corresponding
phases across all folds. Top-level `total_time` measures the model's complete
run, including pipeline initialization, input loading, embedding preparation,
all folds, cleanup, and result publication. The final `config_total_time` field
records the same duration for this model under the selected config. With
`model ALL`, each model is timed separately and its metrics contain only its
own elapsed time. The command also prints each model's duration at completion.
Set `qwen3_4b.A/B_swap` to `aug` (the default) to train on both original and
A/B-swapped rows, or `inf` to train on original IDs using the loss of their
averaged original and swapped probabilities. Both modes average the two
orientations for validation predictions and loss, and for test submissions.
Changing modes requires rebuilding the saved head; embedding caches are shared.

To train a production head and generate a submission:

```shell
uv run model qwen3-4b --build
uv run model qwen3-4b --submit
```

Build mode replaces the saved classifier head and compatibility metadata. Submit
mode validates `test.csv` against `sample_submission.csv`, loads that head, and
writes a timestamped submission. The Qwen backbone is not stored in the
checkpoint.

The previous `qwen3-1.7b` and `qwen3-1.7b-mono-input` implementations and their
configuration sections are retained locally in
`src/llm_classification_finetuning/models/archive/`. This directory is ignored
by Git, and these models are no longer CLI choices.

The frozen model shares content-addressed embeddings between training and test
under `embedding_cache/` beside the processed Parquet. Each SQLite shard
identifies the model and exact Hub revision (or local checkpoint/configuration
content), backbone hidden size, precision, attention implementation, TF32,
pooling, storage format, and extraction version. Individual entries identify
the effective token IDs of one branch, after any requested truncation. Different
compute identities coexist without overwriting one another.

Changing labels, folds, IDs, row order, or dataset file encoding does not
invalidate existing vectors. Overlapping datasets and A/B branches reuse each
other's entries; only missing inputs are encoded. Changing `max_length` or the
serializer also reuses vectors when the resulting token IDs remain identical.
Cache rows are mapped back to the current canonical IDs and order. When all
inputs hit, the backbone weights are not loaded. Local checkpoint identities
require reading the selected weight files to hash their content.

Missing branches are right-padded and encoded independently; changing batch
composition can change floating-point rounding. The precision, attention, and
TF32 boundaries remain separate. Corrupt vectors are recomputed, SQLite writes
are transactional, and corrupt shards are archived before rebuilding. Older
dataset-wide `.pt`/`.json` embedding caches, including `baseline_*`, are left
untouched and are no longer read; their metadata cannot establish the new
backbone/input identities. Saved head checkpoint compatibility is unchanged.

Cross-validation and `--build` use the single CUDA device selected by
`gpu.device`, which defaults to `cuda:0`. Both models encode A and B sequentially
through one backbone; classifier head training uses the same GPU.

`--submit` splits that backbone across the two devices in `gpu.submit_devices`,
which defaults to `[cuda:0, cuda:1]` when omitted. The list must contain exactly
two distinct `cuda:N` identifiers; it is independent of the training device.
For two 16 GB GPUs, the default configuration is:

```yaml
gpu:
  device: cuda:0
  submit_devices: [cuda:0, cuda:1]
  precision: fp16
  attention_implementation: sdpa
  allow_tf32: false
```

The embedding and rotary modules and the first half of the decoder layers reside
on the first GPU; the remaining layers and final normalization reside on the
second. Each decoder layer stays intact, LoRA adapters follow their base layers,
and the FP32 classifier head resides on the first GPU. LoRA gradient checkpointing
is disabled for submissions. Both GPUs must be available and support
`gpu.precision`; there is no automatic precision change, single-GPU fallback, or
CPU/disk offload. `ALL` still runs models sequentially, with each model split
across both GPUs.

Existing checkpoints and embedding/token caches remain compatible. Complete
frozen embedding-cache hits still skip loading backbone weights. Submissions
are written to `results/submit/MODEL-*/submission.csv`; historical `results/test`
files remain untouched. If frozen extraction runs out of memory, reduce
`qwen3_4b.extraction_batch_size` or `qwen3_4b.max_length`.

## Train with LoRA

Prepare the data, then use the LoRA section in the shared `config.yaml`:

```shell
uv run model qwen3-4b-lora
uv run model qwen3-4b-lora --build
uv run model qwen3-4b-lora --submit
```

The model uses one shared `Qwen/Qwen3-4B` backbone to encode prompt+A and
prompt+B separately. LoRA adapters and the pairwise MLP are trained together;
original backbone parameters remain frozen. `A/B_swap: aug` (the default) trains
on both processed orientations with per-row cross-entropy. Set
`qwen3_4b_lora.A/B_swap` to `inf` to train once per original ID using the negative
log-likelihood of the averaged original and A/B-restored swapped probabilities.
Both head-only warmup and joint training use the selected objective. Each branch
is encoded once per sample and reused for both head orientations; joint training
retains gradients through both encodings. In `inf` mode, training batch sizes,
accumulation counts, and build row counts refer to original samples.
Validation loss, fold predictions, and test submissions
average original and A/B-restored swapped probabilities. Each validation fold
starts with fresh adapters and a fresh classifier. Cross-validation saves no
model checkpoints.

The default LoRA training settings target one 32 GB CUDA GPU. Training also
supports other single NVIDIA CUDA GPUs with sufficient memory for the selected
settings; submission inference uses the two-GPU split described above.
Set `gpu.device` to a single `cuda:N` identifier for training and choose
`gpu.precision: fp16`, `bf16`, or `fp32`. BF16 is checked against the selected
GPU's capabilities. FP16 uses gradient scaling; adapters and classifier weights
remain FP32 in all three modes. GPU model names and VRAM sizes are not restricted.

The default configuration allows **each branch** up to 16,384 tokens, dynamically
pads A and B independently, processes branches sequentially, and enables non-reentrant
activation checkpointing with KV caching disabled. It uses micro-batches of one
pair and 16 accumulation steps, giving an effective batch of 16 augmented rows.
LoRA never truncates inputs: all canonical branches are tokenized and checked
before training, and inference checks its inputs before encoding them. An input
longer than `max_length` raises an error with its ID, branch, actual token count,
and required minimum setting. Empty token sequences are also rejected.

Tokenization uses a shared cache keyed by text content and tokenizer content,
with compact int32 CPU views shared across epochs and folds. Identical texts
reuse encodings across IDs, dataset revisions, A/B modes, model profiles, and
training/test inputs. Swapped rows select the opposite branch views. LoRA training
preparation resolves a fixed base revision before loading weights and uses it for
both tokenizer and models. A model revision change can still reuse tokens when
the actual tokenizer is unchanged. Pooling
positions are computed on the CPU; branches without padding omit the attention
mask. Encoding releases the full hidden-state tensor after pooling, while joint
training still retains both branches' gradient graphs. Training loss statistics
stay on the GPU until logging or epoch completion; non-finite losses are checked
at each update boundary before the optimizer can modify weights.

### Measured input lengths

The local processed training data was inspected on 2026-10-20 using the actual
`aligned-prompt-newline-response-v1` serialization and the `Qwen/Qwen3-4B`
tokenizer at revision `1cfa9a7208912126459214e8b04321603b3df60c`, with no truncation
and the tokenizer's default special-token behavior.

| Training input statistic | Value |
| --- | ---: |
| Original source IDs | 57,477 IDs |
| Median branch length | 283 tokens |
| 99th-percentile branch length | 2,193 tokens |
| Longest branch | **13,514 tokens** |

The longest branch is A for ID `2995198009` (the corresponding B branch has
4,225 tokens). The processed Parquet file's SHA-256 fingerprint is
`0db037aac8f94bf68d3f6e02b2bf4214e6bf4817ed01d4c475af5c2f41fab4ff`.
The 16,384-token default therefore preserves every training branch in this
snapshot. New training or test data is checked independently; these measurements
do not establish its maximum length. The 43,035,297 canonical tokens occupy about
164 MiB of int32 payload, excluding offsets and other runtime objects. At batch
size one, lowering the configured ceiling does not reduce work for inputs already
below that ceiling.

### Optimization and head warmup

LoRA rank is 16, alpha is 32, and dropout is 0.05, targeting attention
`q_proj`, `k_proj`, `v_proj`, `o_proj` and MLP `gate_proj`, `up_proj`, `down_proj`.
The 512-unit classifier uses dropout 0.1. AdamW uses adapter learning rate
`0.0002`, head learning rate `0.001`, weight decay `0.01`, and gradient clipping
at 1.0. Training starts with one head-only warmup epoch at a fixed learning rate
of `0.001`, then jointly trains adapters and head for two epochs with learning
rates linearly decaying to zero. All these settings are configurable in
`qwen3_4b_lora`.

`warmup_epochs` is a required nonnegative integer; `0` skips head warmup.
`warmup_extraction_batch_size` is a required positive integer controlling the
no-gradient backbone extraction batch size (default: `1`). Before the first
warmup epoch, each unique ID needed by training or validation is encoded once
in its canonical orientation with the backbone in evaluation mode. Its paired
FP32 features remain in CPU memory; training rows select the appropriate A/B
orientation and retain their original labels and row order without storing
augmented feature copies. Warmup validation averages the two head orientations
over these same frozen features. All warmup epochs reuse the features; both
training and validation views are released when joint training starts, and
features are never written to disk or reused between folds. The full 57,477-ID
feature pool occupies about 1.10 GiB. With `warmup_epochs: 0`, feature extraction
is skipped; the shared training token cache is still used.
`warmup_training_batch_size` is a required positive integer controlling only the
head-warmup micro-batch size over extracted features (default config: `1`).
Joint training uses `training_batch_size` and the optional token budget described
below. Both stages use `gradient_accumulation_steps` and normalize loss by the
actual row count of each update group, including short final batches. The
joint-training decay schedule uses only joint optimizer updates.
`warmup_learning_rate` is a required finite positive number, even when warmup is
disabled. During warmup, only the classifier runs in training mode; validation
uses frozen features in `inference_batch_size` batches. Joint training restores
live validation encodings and backbone training mode while retaining the head's
AdamW state. The joint-training schedule starts at `learning_rate` /
`head_learning_rate` without a learning-rate
ramp. Total training lasts `warmup_epochs + epochs` rounds (three by default),
and training histories and build summaries include both stages. Each validation
fold and full-data build follows these stages; validation still runs after each
epoch. Replace the removed `warmup_ratio` key with `warmup_epochs`,
`warmup_extraction_batch_size`, `warmup_training_batch_size`, and
`warmup_learning_rate`; old keys are rejected as unknown configuration options.

### Optional token-budget training batches

`training_token_budget` is an optional nonnegative integer, defaulting to `0`.
Zero preserves fixed micro-batches. A positive value enables dynamic micro-batches
for joint training only; warmup and inference keep their configured batch sizes.
For example, add this key to the existing `qwen3_4b_lora` section:

```yaml
training_token_budget: 4096
```

Each fold's training rows are globally planned before joint training, excluding
its validation fold and selecting the configured A/B training orientations.
Preparation tries stable sorts by `(A_length, B_length)`,
`(B_length, A_length)`, and `(A_length + B_length, A_length, B_length)`, with
`(id, is_swapped)` as the tie-breaker. Each candidate is divided into update
groups of `G = training_batch_size * gradient_accumulation_steps` rows, with a
possibly shorter final group. Inside each group, longest-total-length rows are
placed first into the existing batch with the least remaining token budget;
a new batch is opened when none fits. Batch cost is
`rows * (longest_A + longest_B)`. The chosen candidate minimizes micro-batch
count, then total padded tokens; this heuristic does not guarantee a global
optimum. A full-data build applies the same planning to all training folds.

Group membership stays fixed across epochs. A zero-based joint epoch uses
`random_state + fold + joint_epoch` to shuffle update groups and their
micro-batches; builds omit the fold term. Dynamic micro-batches can contain more
than `training_batch_size` rows, up to `G`. A single over-budget pair is processed
alone without truncation. Losses are divided by each group's actual row count,
even when the short group is shuffled into the middle of an epoch. Gradient
clipping and optimizer/scaler/scheduler handling occur once per group.
The planned joint schedule still has
`ceil(training_rows / (training_batch_size * gradient_accumulation_steps))`
updates per epoch; FP16 overflow can still skip an update as before.

The effective batch size and schedule length are preserved; positive budgets
change update-group membership from the earlier random grouping to global
length-based grouping. Floating-point accumulation order and dropout draws can
differ. The default configuration leaves the budget at `0`, retaining the
original fixed-batch seeded shuffle. The example budget of `4096` is a starting
point to try, not a measured recommendation for a particular GPU. Padded batches
can change SDPA kernel selection, so dynamic batching is not guaranteed to be
faster or use less peak memory. No throughput or peak-VRAM benchmark has been
run on the target 32 GB virtual GPU.

### Shared token cache and LoRA batch plans

`token_cache/` beside the processed Parquet is shared by frozen-model extraction,
LoRA training/warmup/validation, and LoRA test inference. Each SQLite shard
identifies the tokenizer's encoding content, implementation/library versions,
encoding options, and cache schema; entries identify individual UTF-8 texts by
SHA-256. IDs, labels, folds, row order, dataset fingerprints, model names, model
weight revisions, and training settings are excluded. Only changed or missing
texts are encoded; duplicates and overlapping datasets reuse existing entries.
The tokenizer is loaded to fingerprint its actual content, but cache hits do not
run tokenization. Padding is applied by each caller after cache lookup.

Full untruncated encodings are the common reusable representation. Frozen-model
extraction additionally caches the tokenizer's exact truncated output for texts
exceeding its selected length limit, preserving special-token behavior. LoRA
always validates the full cached lengths against `max_length` and rejects empty
or over-limit branches without truncation. Changing length limits therefore
keeps full tokens reusable. Corrupt entries are recomputed; corrupt SQLite
shards are archived and rebuilt. SQLite transactions publish complete entries.

LoRA's `qwen3_4b_lora_batches/plans/*.pt` files remain separate and identify the
sorted selected `(id, is_swapped, A_length, B_length)` rows, budget, update group
size, and planning algorithm version. Labels, fold numbers, reference order,
text/token content beyond lengths, and unselected validation inputs do not enter
the plan key. Cached groups store sample identities and are mapped to current
dataset indices on every read; current labels are always used for training.
A/B mode and fold selection determine membership before lookup. Changing
budget or effective batch size rebuilds plans while reusing tokens. With
`training_token_budget: 0`, tokens are cached but no global plan is created.
Earlier LoRA `tokens/*.pt` files are left untouched and are no longer read.

`--refresh-cache` replaces requested token entries and model-specific caches;
shared training encodings remain reusable between folds in that run. Warmup
features and live LoRA encodings are never persisted in these token caches.

The independent text API accepts arbitrary serialized strings and a shared
cache directory; an already loaded tokenizer can also be passed to
`TokenizationCache(cache_dir, tokenizer)`:

```python
from pathlib import Path

from llm_classification_finetuning.models.common import TokenizationCache

cache = TokenizationCache.from_pretrained(
    Path("data/processed/token_cache"), "Qwen/Qwen3-4B"
)
encoded = cache.encode(["first input", "second input"])
first_ids = encoded[0]  # Unpadded int32 token IDs, in requested input order.
```

The Python API can prepare a CPU loader without loading model weights or
initializing CUDA:

```python
from pathlib import Path

from llm_classification_finetuning.config import AppConfig
from llm_classification_finetuning.models.qwen3_4b_lora import (
    LoRATrainingBatchRepository,
)

config = AppConfig.load(Path("config.yaml"))
batches = LoRATrainingBatchRepository.from_config(config)
training = batches.get_training(fold=0)  # None selects a full-data build.
training.set_epoch(0)  # Zero-based joint epoch; fixed batching keeps its RNG stream.
loader = training.loader  # Yields PairwiseBatch; training.tokens is shared.
```

For smaller GPUs, lower an enabled token budget or the relevant
`warmup_extraction_batch_size`, `warmup_training_batch_size`,
`training_batch_size`, or `inference_batch_size`. Increase
`gradient_accumulation_steps` when reducing the training batch to retain the
desired effective batch. Keep `max_length` at least as large as the longest
required input. Larger GPUs can use larger batches. Memory fit for the
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
subfolder, excluding `comprehensive`, `submit`, and historical `test` directories.

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

To cross-validate selected models for every `.yaml` or `.yml` file in `configs/`,
run:

```shell
uv run python scripts/validate_all_configs.py
```

To train selected models on all folds and build a separate checkpoint for each
config, run:

```shell
uv run python scripts/build_all_configs.py
```

Both scripts define `MODELS = ("qwen3-4b-lora",)` near the top. Edit each script's
tuple independently to select models and their execution order; supported model
selectors are `qwen3-4b` and `qwen3-4b-lora`. Config files run in filename order,
with the selected models running sequentially for each config. The scripts are
independent: the build script does not run or require cross-validation.

Each model's validation run publishes its own `metrics.json`; `config_total_time`
covers only that model's cross-validation run under the selected config.
The full-data build does not produce `metrics.json`. By default, it writes
`models/qwen3_4b_lora/adapter_CONFIG_STEM.pt` for each config; selecting `qwen3-4b`
also writes `models/qwen3_4b/head_CONFIG_STEM.pt`. Before starting any builds,
the script rejects config stems that are invalid checkpoint tags or duplicate
another stem case-insensitively.

Both scripts continue after a failed config/model task, including a missing
model section in the YAML, and summarize successful and failed tasks. They exit
with status `1` if any task fails, no YAML files are found, or `MODELS` is empty.
To use a tagged checkpoint for submission inference, pass its config and matching
checkpoint tag, for example:

```shell
uv run model qwen3-4b-lora --submit --config configs/config_aug.yaml --checkpoint-tag config_aug
```

| Section | Controls |
| --- | --- |
| `data` | Competition name and raw/processed paths |
| `cross_validation` | Fold count and random seed |
| `gpu` | Training `device`, two `submit_devices` (default `[cuda:0, cuda:1]`), precision, attention implementation, and TF32, shared by both models |
| `qwen3_4b` | Qwen3-4B model, sequence length, batches, MLP, optimizer, and `A/B_swap` training mode (`aug` or `inf`) |
| `qwen3_4b_lora` | Live LoRA training, per-branch sequence length, training/inference batches, optional `training_token_budget`, accumulation, checkpointing, adapter and classifier hyperparameters; `A/B_swap` training mode (`aug` or `inf`) |

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
  set it in `gpu.device` for training, or select two distinct available devices
  in `gpu.submit_devices` for submissions. Both submission GPUs must support the
  selected precision; use `fp16` or `fp32` if either does not support `bf16`.
- **Embedding extraction runs out of memory:** Reduce
  `extraction_batch_size` or `max_length` in the selected model's section.
- **LoRA runs out of memory:** Lower `qwen3_4b_lora.warmup_extraction_batch_size`,
  `warmup_training_batch_size`, `training_batch_size`, or `inference_batch_size`
  for the affected stage, or lower `training_token_budget` when enabled. Keep
  `max_length` at least as large as the longest required branch. Enable
  `gradient_checkpointing` and use FP16/BF16 instead of FP32 where supported.
- **LoRA input exceeds `max_length`:** Use the reported ID, branch, and actual
  token count to raise the limit before rerunning. Inputs are never silently
  truncated; lowering batch sizes or the token budget preserves their content.
