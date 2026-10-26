# Model workflows

[Documentation index](../README.md)

Use the [saved processed split or prepare data](getting-started.md#choose-the-data-source),
then choose a [configured model profile](../README.md#model-profiles).
All profiles use the same saved holdout; validation/build uses one GPU and
submission uses [two GPUs](configuration.md#gpu-configuration).

## Validate, build, or submit

| Mode | Training data | Main output | Does it save a production checkpoint? |
| --- | --- | --- | --- |
| `model MODEL` | Saved train partition; evaluate the saved validation partition after each epoch | Final-epoch `validation.csv`, `metrics.json`, `config.yaml` | No |
| `model MODEL --build` | All original IDs from both partitions, applying the profile's A/B policy | `head[_TAG].pt` or `adapter[_TAG].pt` | Yes |
| `model MODEL --submit` | No training; use test inputs and a compatible checkpoint | `submission.csv` | No |
| `model warmup --build` | Train partition only, original rows | Optional completed EXPT2 warmup cache | No |

Builds have no validation set. Validation publishes final-epoch predictions;
there is no best-epoch selection. Warmup caches and production checkpoints serve
different purposes. See [LoRA stages](lora.md#optimization-and-head-warmup).

## Frozen-model holdout validation

```shell
uv run model qwen3-4b
```

The frozen backbone produces [cached branch embeddings](caching.md#frozen-embeddings),
and only the pairwise MLP is trained. The head uses pooled features
`[A, B, abs(A - B)]` with width `3H` for backbone hidden size `H`.

Each [serialized conversation branch](data-preparation.md#processed-data) is
tokenized with special tokens and right-truncated to `qwen3_4b.max_length` when
needed. Last-nonpadding-token pooling supplies its feature. Frozen pooled vectors
are stored as FP16 and used by the head in FP32. LoRA instead
[rejects over-limit inputs](lora.md#hardware-and-input-limits).

`qwen3_4b.extraction_budget` controls missing backbone branch extraction;
`qwen3_4b.training_batch_size` controls head training, validation, and submission
inference. Complete embedding hits skip backbone-weight loading, while still
checking the hardware and loading config/tokenizer files.

## A/B training modes

Only the frozen profile accepts `A/B_swap`:

| Setting | Training objective | Validation and submission |
| --- | --- | --- |
| `aug` (default) | Per-row loss on original and swapped training rows | Average the two orientations' class probabilities after restoring A/B order |
| `inf` | Loss of the averaged original/swapped probabilities for each original ID | The same probability averaging |

A/B averaging operates on probabilities, not logits. Both modes reuse the same
branch embeddings. Rebuild a head when intentionally changing its training
objective; changing the YAML flag at inference does not retrain its weights.

EXPT2/SR2 use a [swap-equivariant head](lora.md#expt2-equivariant-head), original
training rows, and one prediction per ID. They reject `A/B_swap`.

## Skywork reward backbone with LoRA

```shell
uv run model SR2-Qwen3-4B-lora
```

SR2 loads `Skywork/Skywork-Reward-V2-Qwen3-4B` through `AutoModel`, discards its
scalar reward head, and trains LoRA plus the EXPT2 three-class head. Its
serialization and pooling remain the same as EXPT2; it does not add a chat
template or regenerate the split. It has its own model, checkpoint, warmup-cache,
and result namespace. See [LoRA training](lora.md) for both profiles.

## Build and submit

Build a tagged frozen head on all IDs:

```shell
uv run model qwen3-4b --build --checkpoint-tag experiment
```

Then, in the [submission environment](getting-started.md#submission-environment-pytorch-211--cuda-128),
load the corresponding checkpoint with a compatible submission configuration:

```shell
model qwen3-4b --submit --checkpoint-tag experiment --config path/to/submit.yaml
```

For LoRA, replace the selector with EXPT2 or SR2; the checkpoint contains
adapters and head. All checkpoint types omit base weights; LoRA production
checkpoints also omit optimizer state. Transfer the checkpoint when moving to
another runtime. [Model downloads](caching.md#hugging-face-model-downloads) are
not the trained checkpoint.

Submission requires these raw files under `data.raw_dir`, with exact column order:

| File | Columns |
| --- | --- |
| `test.csv` | `id,prompt,response_a,response_b` |
| `sample_submission.csv` | `id,winner_model_a,winner_model_b,winner_tie` |

IDs must be unique finite integers and match in the same order across both
files. The loader does not reorder them; output preserves that order.
Submissions are written under `results/submit/MODEL-<timestamp>/`.

Without a tag, a build replaces `models/SECTION/head.pt` or `adapter.pt`.
See [checkpoint flags](commands.md#model-selection-and-execution-modes),
[batch validation](commands.md#batch-validation), and [reports](reports.md).

## Checkpoint compatibility

Frozen head schema v3 identifies the `3H` architecture. Loading checks schema,
model name, serializer, classifier `hidden_size`, and class count; inference
also checks backbone hidden size. Earlier `2H` heads require a rebuild, while
per-branch embeddings can remain reusable.

Frozen metadata records context, precision, attention, TF32, and A/B training
mode but does not enforce them when loading. GPU placement and batch sizes may
also change. Such compatibility does not guarantee unchanged predictions.

The frozen head stores neither a pinned backbone revision nor a local-weight
fingerprint. Extraction resolves the current backbone identity; it cannot recover
the build's revision from the head. LoRA
[checkpoints](lora.md#checkpoints) retain the Hub base revision and have stricter
adapter/head compatibility checks.

## Training progress

One CPU counter spans all executed training and validation epochs, logging
roughly 5% milestones with model/mode/stage/epoch, percentage, work count, and
elapsed/estimated total time. Durations use unbounded `HH:MM:SS`; estimates
extrapolate weighted work rather than predicting exact remaining runtime.

For the frozen profile, each training or validation batch counts as one unit:

- Holdout: `epochs * (ceil(training_rows / batch_size) + ceil(validation_rows / batch_size))`.
- Build: `epochs * ceil(training_rows / batch_size)`.

Here `batch_size = training_batch_size`; training counts both orientations for
`aug` and originals for `inf`. Validation counts original IDs, with averaging
inside each batch. Short final batches count. [LoRA progress](lora.md#training-progress)
uses optimizer groups, with half-weight warmup work.

Extraction, preparation, caches, saving, and publication are excluded. Thus the
progress estimate differs from [complete-run timing](reports.md#timing-metrics).
Successful completion prints 100% once; failures do not force completion.

## Retired profiles

Older standard LoRA, EXPT, QLoRA, and Qwen3-1.7B profiles are not registered.
For old names, local archives, configuration changes, and artifact compatibility,
see [migration](migration.md). Source ownership is mapped in
[structure.md](../structure.md#package-map).
