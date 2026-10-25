# Model workflows

[Documentation index](../README.md)

All profiles use a shared Qwen3-4B backbone to compare two response branches.
The frozen profile trains only a pairwise classifier; LoRA also trains
adapters. See [LoRA training](lora.md) for its objectives and settings.
Training uses one GPU; submissions use the
[two-GPU configuration](configuration.md#gpu-configuration).

The frozen profile uses a pairwise MLP head. It concatenates the pooled branch
embeddings as `[A, B, |A - B|]`, where the absolute difference is elementwise
and the head input has width `3H` for backbone hidden size `H`.

The `qwen3-4b-lora-EXPT2` profile trains adapters with a swap-equivariant
[head that derives ties from win representations](lora.md#expt2-equivariant-head).
It uses the `qwen3_4b_lora_EXPT2` configuration section and trains and predicts
only original rows, without orientation averaging. Standalone warmup builds
this same EXPT2 head.

## Frozen-model holdout validation

Prepare the [training data](data-preparation.md) first, then run holdout validation:

```shell
uv run model qwen3-4b
```

The frozen `Qwen/Qwen3-4B` backbone produces cached embeddings; only the
pairwise MLP head is trained. A validation run writes one competition-format
`validation.csv`, `metrics.json`, and a `config.yaml` containing the shared and
current model sections to a timestamped model-named results directory.
All active profiles consume the same saved 80/20 split and initialize one model
per default profile run. See [validation reports and timing metrics](reports.md)
for their interpretation.

Each [serialized conversation branch](data-preparation.md) is tokenized with
special tokens enabled and right-truncated to `qwen3_4b.max_length` when needed.
The final non-padding token supplies its pooled feature. The
[LoRA profile](lora.md#hardware-and-input-limits) rejects over-limit inputs.

## A/B training modes

Set `qwen3_4b.A/B_swap` to `aug` (the default) to train on both original and
A/B-swapped rows, or `inf` to train on original IDs using the loss of their
averaged original and swapped probabilities. Both modes average the two
orientations for validation predictions and loss, and for test submissions.
Changing modes requires rebuilding the saved head;
[embedding caches](caching.md#frozen-embeddings) are shared.

## Training progress

Holdout validation and full-data builds print progress at 5% milestones. One
counter spans all executed stages and epochs of the selected model. Each entry
identifies model, mode, stage, and epoch on the first line. The second line shows
`progress: percentage`, `work: processed/total`, and `time: elapsed/estimated total`.
Both durations use `HH:MM:SS`, with hours allowed to exceed 99. Elapsed time starts
when the progress counter is created; estimated total time extrapolates elapsed
time using the completed fraction of weighted work.

For the frozen model, one training or validation batch counts as one work unit.
A validation run uses
`epochs * (ceil(training_rows / training_batch_size) + ceil(validation_rows / training_batch_size))`;
a build uses only `epochs * ceil(training_rows / training_batch_size)`.
Training counts use both orientations for `aug` and originals for `inf`;
validation counts use original IDs with probability averaging inside each batch.
Short final batches are included. [LoRA progress](lora.md#training-progress)
uses update groups; warmup updates and warmup validation batches count as half
a unit each. Reused warmup contributes no historical work.

The final epoch's validation completes the same counter. Extraction, preparation,
cache operations, checkpoint saving, and publication are excluded. A large
work unit crossing multiple milestones prints one entry with the actual percentage.
Each model prints 100% once on completion; failure does not force completion.
Counters stay on the CPU without adding GPU synchronization.

## Build and submit

To train a production head and generate a submission:

```shell
uv run model qwen3-4b --build
uv run model qwen3-4b --submit
```

Build mode uses every original ID from both partitions and replaces the saved classifier head and compatibility metadata. Submit
mode validates `test.csv` against `sample_submission.csv`, loads that head, and
writes a timestamped submission. The Qwen backbone is not stored in the
checkpoint.

Both test files must have the exact column order expected by the loader:

```text
test.csv: id,prompt,response_a,response_b
sample_submission.csv: id,winner_model_a,winner_model_b,winner_tie
```

IDs must be unique finite integers and match in the same order across the two
files; the loader does not reorder either file. Submission rows preserve that
order. Frozen head training, validation, and submission inference all use
`qwen3_4b.training_batch_size`; backbone extraction uses
`qwen3_4b.extraction_budget`, a required positive padded-token budget (root and
active presets: `524288`). It replaces the old `extraction_batch_size` branch
count. Missing branches are packed by descending length and best fit, with cost
`branch_count * longest_branch_length <= extraction_budget`; an oversized branch
runs alone. Extracted features return to their original positions. See
[extraction budgets](configuration.md#extraction-token-budgets).

For tagged checkpoints, multi-model execution, and cache refresh options, see
the [command reference](commands.md). To run every configuration, see
[batch experiments](configuration.md#batch-experiments).

## Checkpoint compatibility

The frozen head requires matching model name, serialization format, classifier
`hidden_size`, schema version, and class count. Inference also checks the
backbone hidden size. `max_length`, `gpu.precision`,
`gpu.attention_implementation`, `gpu.allow_tf32`, and `A/B_swap` remain recorded
in checkpoint metadata but are not checked when loading. Batch sizes and GPU
indices can also change.
An incompatible configuration requires rebuilding the head with that config.

Frozen head schema v3 identifies the `[A, B, |A - B|]` architecture. Earlier
checkpoints with a `2H` head input require a new `--build`; existing per-branch
embedding caches remain reusable.

The head checkpoint does not store a pinned backbone revision or local weight
fingerprint. Frozen extraction resolves the currently available backbone and
keys its embeddings by that identity; it does not recover the build's backbone
revision from the head. [LoRA checkpoints](lora.md#checkpoints) store the Hub
base revision used during training.

## Retired profiles

The retired `qwen3-1.7b`, `qwen3-1.7b-mono-input`, `qwen3-4b-qlora`,
`qwen3-4b-lora`, and `qwen3-4b-lora-EXPT` implementations are retained locally
in the Git-ignored `src/llm_classification_finetuning/models/archive/` directory.
Standard LoRA and EXPT were retired on 2026-10-25. Their subdirectories contain
source snapshots, configuration/preset snapshots, archival notices, and
historical workflow/configuration documentation. QLoRA retains its historical
README and the archive-level configuration.

These profiles are no longer CLI choices or members of `ALL`. The old
`qwen3_4b_lora`, `qwen3_4b_lora_EXPT`, and `qwen3_4b_qlora` configuration keys
are rejected. To run EXPT2 with an older configuration, rename the shared
`qwen3_4b_lora` section to `qwen3_4b_lora_EXPT2`; there is no compatibility alias.
Historical results, checkpoints, caches, and archived configurations remain at
their original locations. The archive is not included in a fresh clone.
Shared legacy head and quantization definitions remain available within the
EXPT2 implementation, along with the quantization dependencies.
Source-module ownership is covered by
[structure.md](../structure.md#package-map).
