# Model workflows

[Documentation index](../README.md)

Use the [saved processed split or prepare data](getting-started.md#choose-the-data-source),
then choose a [configured model profile](../README.md#model-profiles).
Both SR2 profiles use the saved holdout; validation/build uses one GPU and
LoRA submission uses [two GPUs](configuration.md#gpu-configuration).

## Validate, build, or submit

| Mode | Training data | Main output | Does it save a production checkpoint? |
| --- | --- | --- | --- |
| `model MODEL` | Saved train partition; evaluate the saved validation partition after each epoch | Final-epoch `validation.csv`, `metrics.json`, `config.yaml` | No |
| `model SR2-Qwen3-4B --build` | All original IDs from both partitions | `head[_TAG].pt` | Yes |
| `model SR2-Qwen3-4B-lora --build` | All original IDs from both partitions | `adapter[_TAG].pt` and `merged[_TAG]/` | Yes |
| `model SR2-Qwen3-4B-lora --submit` | No training; use test inputs and a compatible checkpoint | `submission.csv` | No |
| `model warmup --build` | Train partition only, original rows; evaluate train and validation | Completed SR2 warmup cache and both losses | No |

Full-data model builds have no validation set. Standalone warmup evaluates the
saved validation partition, including after restoring a cache hit.
Validation publishes final-epoch predictions, or current head evaluation after
an exact frozen-SR2 cache hit;
there is no best-epoch selection. Warmup caches and production checkpoints serve
different purposes. See [LoRA stages](lora.md#optimization-and-head-warmup).

## Frozen Skywork reward backbone

```shell
uv run model SR2-Qwen3-4B
uv run model SR2-Qwen3-4B --build --checkpoint-tag experiment
```

`SR2-Qwen3-4B` freezes the same Skywork backbone and trains the same current
SR2 swap-equivariant head without creating adapters. Its independent
`SR2_Qwen3_4B` section uses ordinary `epochs`, `training_batch_size`, and
`learning_rate` fields, plus `min_lr_ratio` (default `0.125`), and does not require
`warmup` or LoRA sections. The optimizer, cosine learning-rate decay, clipping,
precision, full-input serialization, and pooling follow head warmup. Training
and inference use original rows only;
the backbone stays in evaluation mode during feature extraction/head training.

Validation trains on the saved train partition and publishes current holdout
predictions, metrics, and the independent configuration snapshot. Each executed
epoch evaluates the whole training set with final weights, dropout disabled,
and configured label smoothing; validation log loss remains unsmoothed.
Compatible tokens, frozen features, and completed head/optimizer/RNG states are
[shared with warmup](caching.md#frozen-sr2-head-training). An exact head-cache hit
evaluates current data without training and publishes `epochs: []`,
`executed_training_epochs: 0`, and `cached_head_evaluation` losses/times. A
lower-epoch hit is supported only with `min_lr_ratio: 1`; it executes only the
remaining epochs, retaining their epoch numbers. With decay enabled, a changed
target epoch count requires the complete new curve to be trained.

Build uses every original ID from both partitions, without holdout evaluation.
It prepares CPU inputs and checks the destination before configuring CUDA.
It writes `models/SR2_Qwen3_4B/head[_TAG].pt`, containing FP32 head parameters,
the head architecture, immutable backbone/tokenizer/serializer identities,
training-source fingerprint, and effective settings. A staged file is reloaded
on CPU and its metadata, exact tensor names/shapes, dtype, and finite values
are checked before atomic replacement. It omits base weights and optimizer
state; the frozen backbone needs no merged export. Build reports only executed
epochs, including zero for an exact full-data head-cache hit.

This profile does not support `--submit`. `ALL` runs it before LoRA for
validation/build when configured; `ALL --submit` prints a skip message and
executes only configured profiles supporting submission.

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

Build a tagged adapter/head checkpoint and merged backbone on all IDs:

```shell
uv run model SR2-Qwen3-4B-lora --build --checkpoint-tag experiment
```

Build prepares the saved data, full conversation tokens, and full-data batch
plan on CPU, then probes both output destinations before configuring CUDA
or loading model weights. Every original ID from both partitions contributes
one training row; there is no holdout evaluation during a build. Compatible token, batch,
and frozen-feature caches are reused. The full-data head-warmup cache is keyed
by all training IDs, so a train-partition-only head cache is not a complete hit.
Exact warmup hits skip that stage. With `warmup_min_lr_ratio: 1`, compatible
lower-epoch hits execute only the remaining warmup epochs before joint training;
cosine decay requires an exact target-epoch hit or a complete new curve.

The build summary reports actual executed epochs separately for head warmup
and joint training, followed by the final training loss, checkpoint path, and
merged backbone directory.
The total epoch count excludes cached warmup epochs. For example, an exact
warmup hit with two configured joint epochs reports zero executed warmup epochs
and two executed joint epochs.

Saving writes a temporary checkpoint and reloads it on CPU using the same
metadata, finite-tensor, and head-architecture checks used by submission.
Only a validated temporary file replaces the selected checkpoint. A failed
write or validation leaves the previous checkpoint intact and cleans up the
temporary file; this check does not reload base weights or run GPU inference.

After saving the adapter/head checkpoint, build moves the trained backbone to
CPU, merges its LoRA weights, and exports a full Hugging Face model directory.
The export contains the model configuration, Safetensors weights split at 5 GB,
and the training tokenizer. It preserves the build's backbone precision. A
tagged build writes `models/SR2_Qwen3_4B_lora/merged_TAG/`; an untagged build
writes `models/SR2_Qwen3_4B_lora/merged/`.

The export is staged and checked locally before replacing an existing directory,
with backup/restore handling during replacement. If export fails, the command
reports failure and the saved adapter/head checkpoint remains available; the
previous export is preserved at its original location or the backup path named
in the error. Build reports success only after both outputs have been saved.
See [export checks and metadata](lora.md#merged-backbone-export).

Load an exported backbone independently of the original Hub cache:

```python
from transformers import AutoModel, AutoTokenizer

path = "models/SR2_Qwen3_4B_lora/merged_experiment"
backbone = AutoModel.from_pretrained(path, dtype="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
```

This is the `AutoModel` backbone used for training. The trained classification
head remains in the corresponding adapter/head checkpoint.

Then, in the [submission environment](getting-started.md#submission-environment-pytorch-211--cuda-128),
load the corresponding checkpoint with a compatible submission configuration:

```shell
model SR2-Qwen3-4B-lora --submit --checkpoint-tag experiment --config path/to/submit.yaml
```

Submission continues to load the original base revision plus the adapters and
head from the checkpoint; it does not select the merged export. The checkpoint
omits base weights and optimizer state. Transfer it when moving submission to
another runtime, and preserve the merged directory separately when retaining
the complete trained backbone. [Model downloads](caching.md#hugging-face-model-downloads)
contain the original base weights.

Submission requires these raw files under `data.raw_dir`, with exact column order:

| File | Columns |
| --- | --- |
| `test.csv` | `id,prompt,response_a,response_b` |
| `sample_submission.csv` | `id,winner_model_a,winner_model_b,winner_tie` |

IDs must be unique finite integers and match in the same order across both
files. The loader does not reorder them; output preserves that order.
Submissions are written under `results/submit/MODEL-<timestamp>/`.

Without a tag, a build replaces `models/SR2_Qwen3_4B_lora/adapter.pt` and
`models/SR2_Qwen3_4B_lora/merged/`. Different tags retain separate outputs.
See [checkpoint flags](commands.md#model-selection-and-execution-modes),
[batch validation](commands.md#batch-validation), and [reports](reports.md).

## Checkpoint compatibility

SR2 retains adapter/head schema v2 and uses the
`swap_equivariant_harmonic_discrepancy_v3` architecture with fused L/M win
projections, independent linear-difference X/Y features for harmonic fusion and
the discrepancy correction, and one scalar tie bias. The same correction is
added to both win logits and subtracted from the tie logit. Artifact paths remain
unchanged. The architecture marker rejects harmonic/discrepancy v1/v2 and earlier
tie-from-wins v1/v2 checkpoints; exact head tensor checks additionally enforce the
fused projection and new B3 bias. Rebuild earlier checkpoints with the matching
configuration. Existing warmup files are preserved; the new architecture selects
new cache keys and regenerates warmup.
See [adapter/head checks](lora.md#checkpoints) for the enforced fields and pinned
Hub revision. Historical Qwen checkpoints are not converted into SR2 checkpoints.

## Training progress

Progress logs include model/mode/stage/epoch, percentage, work count, and
`time: elapsed/estimated_total`. Durations use unbounded `HH:MM:SS`; the right-hand
duration is the current counter's estimated total time, calculated as
`elapsed * total_work / completed_work`, rather than remaining time.

[LoRA progress](lora.md#training-progress) has independent Warmup feature
extraction, Head warmup, and Joint training stages. Work and timing reset between
stages and accumulate across epochs within a stage. Each optimizer group,
extraction branch batch, and head inference batch counts as one work unit. Warmup
stages log approximately every 20%; Joint training logs approximately every 10%
and includes validation feature extraction. Dynamic training micro-batches do
not change the finalized update-group count.

Complete inline head-warmup hits log reuse without progress for skipped work. A
complete standalone head hit reports a separate Cached head evaluation stage,
executing zero training epochs. Warmup feature extraction runs only for missing
unique branches; complete feature hits log reuse without an extraction stage.
Partial head hits in either mode report Head warmup progress only for remaining
epochs, keeping epoch numbers continuous with the source cache. CPU preparation, cache I/O,
saving, and publication remain outside the stage boundaries, so stage estimates
differ from [complete-run timing](reports.md#timing-metrics). Each successfully
completed stage prints 100% once with its actual duration on both sides;
failures do not force completion.

The retained frozen-model infrastructure keeps its legacy shared CPU counter and
5% interval. Existing `TrainingProgress(total_steps=...)` callers retain that
behavior; LoRA explicitly starts and finishes each independent stage.

## Retired profiles

Frozen Qwen3-4B, EXPT2, older standard LoRA, EXPT, QLoRA, and Qwen3-1.7B
profiles are not registered.
For old names, local archives, configuration changes, and artifact compatibility,
see [migration](migration.md). Source ownership is mapped in
[structure.md](../structure.md#package-map).
