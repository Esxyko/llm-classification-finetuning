# Model workflows

[Documentation index](../README.md)

Use the [saved processed split or prepare data](getting-started.md#choose-the-data-source),
then choose a [configured model profile](../README.md#model-profiles).
SR2 uses the saved holdout; validation/build uses one GPU and
submission uses [two GPUs](configuration.md#gpu-configuration).

## Validate, build, or submit

| Mode | Training data | Main output | Does it save a production checkpoint? |
| --- | --- | --- | --- |
| `model MODEL` | Saved train partition; evaluate the saved validation partition after each epoch | Final-epoch `validation.csv`, `metrics.json`, `config.yaml` | No |
| `model MODEL --build` | All original IDs from both partitions | `adapter[_TAG].pt` | Yes |
| `model MODEL --submit` | No training; use test inputs and a compatible checkpoint | `submission.csv` | No |
| `model warmup --build` | Train partition only, original rows | Optional completed SR2 warmup cache | No |

Builds have no validation set. Validation publishes final-epoch predictions;
there is no best-epoch selection. Warmup caches and production checkpoints serve
different purposes. See [LoRA stages](lora.md#optimization-and-head-warmup).

## Skywork reward backbone with LoRA

```shell
uv run model SR2-Qwen3-4B-lora
```

SR2 loads `Skywork/Skywork-Reward-V2-Qwen3-4B` through `AutoModel`, discards its
scalar reward head, and trains LoRA plus a
[swap-equivariant three-class head](lora.md#swap-equivariant-head). It uses the
aligned conversation serializer and last-real-token pooling, without a chat
template or regenerating the split. Training uses original rows and produces
one prediction per ID; `A/B_swap` is rejected. Its model, checkpoint,
warmup-cache, and result namespace is SR2. See [LoRA training](lora.md).

## Build and submit

Build a tagged adapter/head checkpoint on all IDs:

```shell
uv run model SR2-Qwen3-4B-lora --build --checkpoint-tag experiment
```

Then, in the [submission environment](getting-started.md#submission-environment-pytorch-211--cuda-128),
load the corresponding checkpoint with a compatible submission configuration:

```shell
model SR2-Qwen3-4B-lora --submit --checkpoint-tag experiment --config path/to/submit.yaml
```

The checkpoint contains adapters and the head, omitting base weights and
optimizer state. Transfer the checkpoint when moving to
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

Without a tag, a build replaces `models/SR2_Qwen3_4B_lora/adapter.pt`.
See [checkpoint flags](commands.md#model-selection-and-execution-modes),
[batch validation](commands.md#batch-validation), and [reports](reports.md).

## Checkpoint compatibility

SR2 retains adapter/head schema v2 and uses the
`swap_equivariant_harmonic_discrepancy_v2` architecture with shared cross
projections, harmonic fusion, an absolute-difference correction, and one scalar
tie bias. Artifact paths remain unchanged. The architecture marker and exact
head tensor structure reject earlier tie-from-wins v1/v2 checkpoints; the
architecture marker also rejects harmonic/discrepancy v1 checkpoints, which
must be rebuilt with the matching configuration. Existing warmup files are
preserved; the new architecture selects new cache keys and regenerates warmup.
See [adapter/head checks](lora.md#checkpoints) for the enforced fields and pinned
Hub revision. Historical Qwen checkpoints are not converted into SR2 checkpoints.

## Training progress

One CPU counter spans all executed training and validation epochs, logging
roughly 5% milestones with model/mode/stage/epoch, percentage, work count, and
elapsed/estimated total time. Durations use unbounded `HH:MM:SS`; estimates
extrapolate weighted work rather than predicting exact remaining runtime.

[LoRA progress](lora.md#training-progress) uses optimizer groups, with half-weight
warmup work. Reused warmup contributes no executed work; dynamic micro-batches
do not change the finalized update-group count.

Extraction, preparation, caches, saving, and publication are excluded. Thus the
progress estimate differs from [complete-run timing](reports.md#timing-metrics).
Successful completion prints 100% once; failures do not force completion.

## Retired profiles

Frozen Qwen3-4B, EXPT2, older standard LoRA, EXPT, QLoRA, and Qwen3-1.7B
profiles are not registered.
For old names, local archives, configuration changes, and artifact compatibility,
see [migration](migration.md). Source ownership is mapped in
[structure.md](../structure.md#package-map).
