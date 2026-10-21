# Train with LoRA

[Documentation index](../README.md)

The [QLoRA profile](qlora.md) reuses this training workflow with a 4-bit NF4
training/validation base and a non-quantized submission base. Unless stated
otherwise, the objectives, batching, input limits, and schedules below also
apply to its independent `qwen3_4b_qlora` configuration section.

Prepare the [training data](data-preparation.md), then use the LoRA section in
the shared [`config.yaml`](../config.yaml):

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
is encoded once per selected row during joint training. In `aug` mode, original
and swapped rows are separate samples and are encoded separately. In `inf`
mode, the two branch encodings are reused for both head orientations; joint
training retains gradients through both encodings. In `inf` mode, training
batch sizes, accumulation counts, and build row counts refer to original samples.
Validation loss, fold predictions, and test submissions
average original and A/B-restored swapped probabilities. Each validation fold
starts with fresh adapters and initializes the classifier from completed
compatible warmup when available. Cross-validation saves warmup cache states,
but no production adapter checkpoints.

## Hardware and input limits

Training uses one NVIDIA CUDA GPU with sufficient memory for the selected
settings; GPU model names and VRAM sizes are not restricted. Submission
inference uses the [two-GPU split](configuration.md#gpu-configuration).
Set `gpu.device` to a single `cuda:N` identifier for training and choose
`gpu.precision: fp16`, `bf16`, or `fp32`. BF16 is checked against the selected
GPU's capabilities. FP16 uses gradient scaling; adapters and classifier weights
remain FP32 in all three modes.

With `gpu.precision: bf16`, residual states, activation-checkpoint layer inputs,
pooled features, and temporary CPU features remain BF16. The classifier uses
BF16 autocast for every head forward pass, including warmup and both inference
orientations, then returns FP32 logits for loss calculation and probability
averaging. Trainable parameters, gradients, and AdamW moment tensors remain
FP32. RMSNorm retains FP32 variance calculation with BF16 weights and outputs.
FP16 and FP32 modes retain FP32 pooled/CPU features and classifier computation.

With `gpu.attention_implementation: sdpa`, LoRA permits only Flash, memory-efficient,
and cuDNN attention kernels. The `math` backend is disabled during every backbone
encoding, including warmup, validation, submission, and gradient-checkpoint
recomputation. If none of these fused kernels supports the current GPU, precision,
and inputs, execution fails instead of falling back to `math`. Backend flags are
restored after each scoped call. Explicit `eager` attention remains available.
This uses PyTorch's [SDPA backend context](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
and checkpoint [forward/recomputation contexts](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint).

The root [`config.yaml`](../config.yaml) preset allows **each branch** up to
16,384 tokens, dynamically pads A and B independently, processes branches
sequentially, and enables non-reentrant activation checkpointing with KV caching
disabled. It uses micro-batches of one pair and 16 accumulation steps, giving an
effective batch of 16 augmented rows.
LoRA never truncates inputs: all canonical branches are tokenized and checked
before training, and inference checks its inputs before encoding them. An input
longer than `max_length` raises an error with its ID, branch, actual token count,
and required minimum setting. Empty token sequences are also rejected.

Tokenization uses a shared cache keyed by text content and tokenizer content,
with compact int32 CPU views shared across epochs and folds. Identical texts
reuse encodings across IDs, dataset revisions, A/B modes, model profiles, and
training/test inputs. Swapped rows select the opposite branch views. For Hub
backbones, training preparation resolves a fixed base revision before loading
weights and uses it for both tokenizer and models. A model revision change can
still reuse tokens when the actual tokenizer is unchanged. All backbone stages
right-pad each branch independently without constructing or passing padding masks.
Qwen's causal attention prevents real tokens from attending to right padding;
pooling selects the last real token using positions computed on the CPU. With
SDPA and standard Qwen3-4B, omitting the padding mask enables Transformers' native
GQA path without explicitly repeating K/V heads. Kernel selection and
floating-point rounding can differ from masked encoding. Encoding releases the
full hidden-state tensor after pooling, while joint training still retains both
branches' gradient graphs. Training loss statistics stay on the GPU until logging
or epoch completion; non-finite losses are checked at each update boundary before
the optimizer can modify weights.

## Measured input lengths

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
The root preset's 16,384-token limit preserves every training branch in this
snapshot. New training or test data is checked independently; these measurements
do not establish its maximum length. The 43,035,297 canonical tokens occupy about
164 MiB of int32 payload, excluding offsets and other runtime objects. At batch
size one, lowering the configured ceiling does not reduce work for inputs already
below that ceiling.

## Optimization and head warmup

In the root preset, LoRA rank is 16, alpha is 32, and dropout is 0.05. Adapters
target attention `q_proj`, `k_proj`, `v_proj`, `o_proj` and MLP `gate_proj`,
`up_proj`, `down_proj`.
The 512-unit classifier uses dropout 0.1. AdamW uses adapter learning rate
`0.0002`, head learning rate `0.001`, weight decay `0.01`, and gradient clipping
at 1.0. Training starts with one head-only warmup epoch at a fixed learning rate
of `0.001`, then jointly trains adapters and head for two epochs with learning
rates linearly decaying to zero. All these settings are configurable in
`qwen3_4b_lora`.

`warmup_epochs` is a required nonnegative integer; `0` skips head warmup and
does not read or write its cache. Otherwise a compatible completed
[head-warmup cache](caching.md#loraqlora-head-warmup) restores weights, head
AdamW moments, and random states, skipping warmup extraction, training, and
validation. It matches the current original training/validation ID sets (all
original IDs for builds), warmup optimization settings, and core model/input
identities. Compute precision, attention/TF32, extraction batch size, LoRA
settings, local model paths with identical contents, and joint-training
learning rates, epochs, or batching may change. LoRA/QLoRA and each split/build
are isolated; changing assignments among other folds does not invalidate this
split, and builds ignore fold assignments.
`--refresh-cache` forces a fresh warmup. Across precision changes, the original
warmup result is reused; only FP16-to-FP16 restores the gradient scaler.

On a cache miss, warmup uses the extraction and training settings below.

`warmup_extraction_batch_size` is a required positive integer controlling the
no-gradient backbone extraction batch size for warmup, joint validation, and
submissions (root preset: `1`). Before the first
warmup epoch, each unique ID needed by training or validation is encoded once
in its canonical orientation with the backbone in evaluation mode. Its paired
features remain in CPU memory, using BF16 for `gpu.precision: bf16` and FP32
otherwise; training rows select the appropriate A/B
orientation and retain their original labels and row order without storing
augmented feature copies. Warmup validation averages the two head orientations
over these same frozen features. All warmup epochs reuse the features; both
training and validation views are released when joint training starts, and
features are never written to disk or reused between folds. Only the completed
head and its training state are persisted. The full 57,477-ID
Qwen3-4B feature tensor occupies about 0.55 GiB in BF16 or 1.10 GiB in FP32,
excluding loader batches and metadata. With `warmup_epochs: 0`, warmup feature
extraction is skipped; joint validation and submission still extract features,
and the shared training token cache is still used.
`warmup_training_batch_size` is a required positive integer controlling only the
head-warmup batch size over extracted features (root preset: `8`). Each warmup
batch performs one optimizer update without gradient accumulation, independently
of `gradient_accumulation_steps`. Joint training uses `training_batch_size`,
`gradient_accumulation_steps`, and the
[optional token budget](#optional-token-budget-training-batches).
Both stages normalize loss by the actual row count of each update, including
short final warmup batches and incomplete joint accumulation groups. The
joint-training decay schedule uses only joint optimizer updates.
`warmup_learning_rate` is a required finite positive number, even when warmup is
disabled. During warmup, only the classifier runs in training mode; validation
reuses frozen features in `head_inference_batch_size` batches. Joint training
restores backbone training mode while retaining the head's AdamW state.
The joint-training schedule starts at the adapter `learning_rate`
and classifier `head_learning_rate`, without a learning-rate ramp.
Without cache reuse, training lasts `warmup_epochs + epochs` rounds and
training histories and build summaries include both stages. A reused warmup
leaves only the joint epochs in this run's history and build epoch count;
epoch numbers retain their configured warmup offset. Old warmup losses and
timings are not replayed.
Each validation fold and full-data build follows these stages. Cross-validation
evaluates its held-out fold after every executed warmup and joint epoch, and
publishes the final epoch's predictions. Full-data builds have no validation set and do
not run validation. Replace the removed `warmup_ratio` key with `warmup_epochs`,
`warmup_extraction_batch_size`, `warmup_training_batch_size`, and
`warmup_learning_rate`; old keys are rejected as unknown configuration options.

## Two-stage validation and submission

`head_inference_batch_size` is a required positive integer controlling the
classifier batch size for all validation and submission predictions. Rename the
old `inference_batch_size` key while preserving its value; the old key is rejected.
This setting does not control backbone extraction or checkpoint compatibility.

After every joint-training epoch, validation first extracts each needed ID once
in canonical A/B orientation using the current adapters. Submission extracts test
features after restoring the checkpoint. Both use `warmup_extraction_batch_size`,
evaluation mode, and no gradients, even with `warmup_epochs: 0`. Extraction
finishes before head inference starts. Paired features stay in temporary CPU
memory in BF16 for `gpu.precision: bf16` and FP32 otherwise; the head selects
reference-ordered feature views, averages original and
A/B-restored swapped probabilities, and computes validation loss when labels
exist. Submission inputs do not require labels.

Warmup validation reuses its existing features. Each joint-validation call and
submission extracts fresh features and releases them on return; features are
never persisted or reused across joint epochs or folds. Joint-validation timing
includes both extraction and head inference. Extraction logs identify warmup,
joint validation, or submission separately.

## Training progress

Head warmup, joint training, and the validation batches after every epoch share
one progress counter across all epochs and folds. Progress is printed at
approximately 5% intervals, with the current stage identified in each line.
Each warmup training batch and warmup validation batch counts as **0.5**
work units; each joint-training update group and joint validation batch counts as
**1**. With `warmup_epochs: 0` or reused warmup, the total contains only joint
training and its validation for that fold/build.

For each fold, let `W` be
`ceil(training_rows / warmup_training_batch_size)`,
`J` be `ceil(training_rows / (training_batch_size * gradient_accumulation_steps))`,
and `V` be `ceil(validation_rows / head_inference_batch_size)` using original validation
IDs. The fold contributes `0.5 * warmup_epochs * (W + V) + epochs * (J + V)` work
units when warmup runs, or `epochs * (J + V)` when warmup is reused.
Cross-validation sums the separately rounded fold counts; a build uses
all selected training rows with `V = 0`. Training counts respect `A/B_swap`,
including original IDs only for `inf`.

Training advances the counter after each warmup batch or complete joint
accumulation group, including short batches/groups and skipped FP16 updates.
Evaluation advances it once per head-validation batch, using the corresponding
stage's weight.
Dynamic token batches retain the same group counts regardless of how many
micro-batches each group contains.
Percentage logs replace the previous first-batch/every-100-batches training logs;
epoch loss summaries and stage-specific feature-extraction logs are retained. Feature
extraction, cache work, and saving are excluded, as described in the
[shared progress behavior](models.md#training-progress).

Python callers of `LoRATrainer.fit()` may supply one `progress` object from
`llm_classification_finetuning.models.common` to share the total across calls.
If omitted, a counter is created for that fit's training rows, optional
validation rows, and both stages' epochs.

## Optional token-budget training batches

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
The budget counts padded A/B input tokens, not GPU memory bytes. An
over-budget singleton is allowed, so lowering the budget cannot reduce the
tokens or memory needed by that individual input.

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
differ. The root preset leaves the budget at `0`, retaining the
original fixed-batch seeded shuffle. The example budget of `4096` is a starting
point to try, not a measured recommendation for a particular GPU. Padded batches
can change SDPA kernel selection, so dynamic batching is not guaranteed to be
faster or use less peak memory. No throughput or peak-VRAM benchmark has been
run on the target 32 GB virtual GPU.

## Memory tuning

For smaller GPUs, lower an enabled token budget or the relevant
`warmup_extraction_batch_size`, `warmup_training_batch_size`,
`training_batch_size`, or `head_inference_batch_size`. The extraction setting
controls backbone memory for warmup, joint validation, and submission; the head
setting controls only classifier batches over CPU features. Increase
`gradient_accumulation_steps` when reducing the training batch to retain the
desired effective batch. Keep `max_length` at least as large as the longest
required input. Larger GPUs can use larger batches. Memory fit for the
16,384-token preset on a 32 GB GPU has not been measured; execution reports OOM
without silently lowering context length.

## Checkpoints

Builds atomically save adapters, the classifier, and compatibility metadata in
`models/qwen3_4b_lora/adapter.pt`. `--checkpoint-tag TAG` selects
`adapter_TAG.pt`. Base weights and optimizer state are not stored. The checkpoint
records the resolved Hub base revision so inference reloads matching weights
and tokenizer files. Model name, classifier size and dropout, serialization,
and LoRA settings must match. `max_length` and `A/B_swap` remain recorded in
checkpoint metadata. The `training_quantization` marker must also match: `none`
for LoRA (including older checkpoints without the marker), or `bnb-nf4-double`
for QLoRA. `max_length` and `A/B_swap` are not checked when loading. CUDA index,
GPU model, precision, attention implementation, TF32, batch sizes, and accumulation settings
can also change between build and inference. Inputs must still fit the current
`max_length`. Hub revision pinning does not verify local directory contents;
the checkpoint neither includes nor fingerprints local base weights.
There are no persistent LoRA embedding caches.
BF16 head computation preserves the classifier's state-dict keys and FP32
checkpoint parameters, so existing adapter/head checkpoints remain loadable.

See [caching and Python APIs](caching.md) for shared token encodings and batch-plan
reuse, [configuration](configuration.md) for batch experiments, and
[troubleshooting](troubleshooting.md) for context-limit and memory errors.
