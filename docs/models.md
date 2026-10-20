# Model workflows

[Documentation index](../README.md)

Both profiles use a shared Qwen3-4B backbone to compare two response branches.
The frozen profile trains only a pairwise classifier; the LoRA profile also
trains adapters. See [LoRA training](lora.md) for its objectives and settings.
Training uses one GPU; submissions use the
[two-GPU configuration](configuration.md#gpu-configuration).

## Frozen-model cross-validation

Prepare the [training data](data-preparation.md) first, then run cross-validation:

```shell
uv run model qwen3-4b
```

The frozen `Qwen/Qwen3-4B` backbone produces cached embeddings; only the
pairwise MLP head is trained. A validation run writes one competition-format
CSV per fold plus `metrics.json` to a timestamped model-named results directory.
See [validation reports and timing metrics](reports.md) for their interpretation.

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

Cross-validation and full-data builds print progress whenever processed work
reaches or crosses the next 10% milestone. Each model uses one counter across all
stages and epochs, including all folds in cross-validation. Each line identifies
the model, mode, stage, fold or build, current epoch, actual percentage, and
processed/total weighted work units. Output is flushed immediately:

```text
qwen3_4b build head training Build epoch 1/2: progress=10.0% work=110.0/1,100.0
```

For the frozen model, each training batch and each validation batch count as one
work unit. A cross-validation fold contributes
`epochs * (ceil(training_rows / training_batch_size) + ceil(validation_rows / training_batch_size))`;
builds contribute only `epochs * ceil(training_rows / training_batch_size)`.
Training rows follow `A/B_swap`: both orientations for `aug`, original IDs only
for `inf`. Validation always uses original IDs, with both probability orientations
averaged inside each batch. Rounding each fold separately accounts for uneven
folds and short final batches. [LoRA progress](lora.md#training-progress) includes
gradient accumulation and merges warmup at half weight into the same counter.

Training update groups and the validation batches after every epoch advance the
same counter. Feature extraction, token/batch-plan preparation, cache operations,
and checkpoint saving are excluded. Thus 100% covers the final epoch's validation
when present; saving and result publication may still follow. Existing epoch loss
and extraction logs remain available. With few work units, one group or batch may
cross several milestones; they produce one line with the actual percentage.
Each model prints 100% once when its final work unit is processed, with no forced
completion log on failure. The counters run on the CPU without adding GPU
synchronization.

## Build and submit

To train a production head and generate a submission:

```shell
uv run model qwen3-4b --build
uv run model qwen3-4b --submit
```

Build mode replaces the saved classifier head and compatibility metadata. Submit
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
`qwen3_4b.extraction_batch_size`.

For tagged checkpoints, multi-model execution, and cache refresh options, see
the [command reference](commands.md). To run every configuration, see
[batch experiments](configuration.md#batch-experiments).

## Checkpoint compatibility

The frozen head requires matching model name, `max_length`, serialization
format, `gpu.precision`, `gpu.attention_implementation`, `gpu.allow_tf32`,
classifier `hidden_size`, `A/B_swap`, schema version, and class count. Inference
also checks the backbone hidden size. Batch sizes and GPU indices can change.
An incompatible configuration requires rebuilding the head with that config.

The head checkpoint does not store a pinned backbone revision or local weight
fingerprint. Frozen extraction resolves the currently available backbone and
keys its embeddings by that identity; it does not recover the build's backbone
revision from the head. [LoRA checkpoints](lora.md#checkpoints) store the Hub
base revision used during training.

## Retired profiles

The previous `qwen3-1.7b` and `qwen3-1.7b-mono-input` implementations and their
configuration sections may be retained locally in the Git-ignored
`src/llm_classification_finetuning/models/archive/` directory. These models are
no longer CLI choices, and that archive is not included in a fresh clone.
Source-module ownership is covered by
[structure.md](../structure.md#package-map).
