# EXPT2 and SR2 LoRA training

[Documentation index](../README.md)

Both active LoRA profiles train adapters and a three-class comparison head over
one shared Qwen3-family backbone. Original base parameters remain frozen;
training is non-quantized on one GPU. Each pair is serialized as separate A and B
branches and pooled at the last real token. Both stages train on **original rows
only**; validation/submission make one head prediction per ID.

```shell
uv run model qwen3-4b-lora-EXPT2
uv run model qwen3-4b-lora-EXPT2 --build --checkpoint-tag experiment
```

Select `SR2-Qwen3-4B-lora` instead for SR2. Both require their own model section
and the shared `warmup` section. See [commands](commands.md) for execution modes
and [configuration](configuration.md) for the complete fields and parser defaults.

## SR2 reward-backbone profile

| Profile | Model section | Backbone in shipped presets | Checkpoint / warmup namespace |
| --- | --- | --- | --- |
| EXPT2 | `qwen3_4b_lora_EXPT2` | `Qwen/Qwen3-4B` | `qwen3_4b_lora_EXPT2` / `qwen3_4b_lora_EXPT2_warmup` |
| SR2 | `SR2_Qwen3_4B_lora` | `Skywork/Skywork-Reward-V2-Qwen3-4B` | `SR2_Qwen3_4B_lora` / `SR2_Qwen3_4B_lora_warmup` |

SR2 loads Skywork's Transformer through `AutoModel`, omitting its scalar reward
head. It uses the EXPT2 head and the same serialization, pooling, batching,
precision, and training rules below. The two YAML sections are independent;
head size, rank, smoothing, rates, and epochs may differ. Read their actual
preset values rather than assuming only `model_name` changes.

SR2 warms up on Skywork features in its own namespace. The standalone `warmup`
command uses the EXPT2 namespace; it does not prepare SR2's inline warmup.

## EXPT2 equivariant head

For pooled column vectors $A,B$ of width $D$ and classifier width $h$:

$$
\begin{aligned}
u &= W_s[A+B;\,|A-B|] + b_1, & W_s &\in \mathbb{R}^{h\times 2D},\\
v &= W_d(A-B), & W_d &\in \mathbb{R}^{h\times D},\\
p_A &= \operatorname{GELU}(u+v), & p_B &= \operatorname{GELU}(u-v),\\
t &= \operatorname{GELU}\left(W_t\frac{p_A+p_B}{2}+b_2\right),
& W_t &\in \mathbb{R}^{h\times h}.
\end{aligned}
$$

The directional projection has no bias. The tie projection acts on the mean of
the activated win representations **before dropout**. Win branches share a
dropout mask; the tie branch has an independent mask. One shared bias-free
`Linear(h, 1)` scores the three representations, returning FP32 logits in
A-win, B-win, tie order. Swapping inputs exchanges win logits and preserves the
tie logit under the same masks; evaluation disables dropout.

These weights are trained directly. The architecture marker is
`swap_equivariant_tie_from_wins_v1`; adapter/head checkpoint schema is v2 and
warmup schema is v5. Active profiles reject retired-profile head states.
There is no `A/B_swap` setting or orientation averaging for EXPT2/SR2.

## Hardware and input limits

Validation/build uses `gpu.device`; submission uses the
[two-GPU layer split](configuration.md#gpu-configuration). All branches are
right-padded independently without padding masks. Qwen's causal attention
protects real tokens from right padding, and CPU-computed last-real-token
positions select pooled features. KV caching is disabled. Joint training encodes
A then B while retaining both gradient graphs.

LoRA **never truncates**. Token preparation validates full canonical branches
before training, and inference checks its inputs before encoding. Empty or
`max_length`-exceeding branches fail with the ID, branch, actual count, and
required minimum limit. Lowering a configured ceiling does not reduce work for
inputs already below it. Shared full-token encodings remain reusable when
limits change, but length checks still run.

For Hub backbones, preparation resolves a fixed revision and uses it for the
tokenizer and weights. Token reuse depends on actual tokenizer content, not
just the model name/revision.

| Precision | Backbone / temporary pooled CPU features | Trainable state and head computation |
| --- | --- | --- |
| `bf16` | BF16 residuals, checkpoint inputs, and features | FP32 parameters, gradients, and AdamW moments; BF16 head autocast; FP32 logits |
| `fp16` | FP16 backbone compute; FP32 pooled features | FP32 head/state with gradient scaling |
| `fp32` | FP32 | FP32 head/state |

BF16 support is checked on each selected GPU. BF16 adapter encoding and
checkpoint recomputation disable PEFT's input dtype upcasting while preserving
FP32 adapter parameters. RMSNorm retains FP32 variance calculation.

With `sdpa`, LoRA permits Flash, memory-efficient, and cuDNN backends and disables
math fallback during warmup, joint training, validation, submission, and
checkpoint recomputation. Unsupported fused inputs fail explicitly. Scoped flags
are restored on exit; explicit `eager` attention is available. Submission also
applies the [T4 compatibility path](getting-started.md#submission-environment-pytorch-211--cuda-128)
that expands K/V heads. Training/validation keep their native GQA policy.

## Measured input lengths

These are **historical measurements**, not a guarantee for the current saved
split or test inputs. A 2026-10-20 snapshot used
`aligned-prompt-newline-response-v1`, the Qwen3-4B tokenizer at revision
`1cfa9a7208912126459214e8b04321603b3df60c`, default special tokens, and no truncation.

| Statistic | Historical value |
| --- | ---: |
| Original IDs | 57,477 |
| Median branch | 283 tokens |
| 99th percentile | 2,193 tokens |
| Longest branch | 13,514 tokens |
| Canonical A+B token total | 43,035,297 |

The longest branch was A for ID `2995198009`; B had 4,225 tokens. The snapshot's
Parquet SHA-256 was
`0db037aac8f94bf68d3f6e02b2bf4214e6bf4817ed01d4c475af5c2f41fab4ff`, which differs
from the checked-in manifest. Its int32 token payload was about 164 MiB before
offsets/metadata. Current train/test inputs are checked independently.

## Optimization and head warmup

| Stage | Trainable work | Validation / persistence |
| --- | --- | --- |
| Head warmup, cache miss | Extract frozen eval-mode features with adapter forwards disabled; train only the head at fixed `warmup_learning_rate` | Holdout validates after each epoch using the same temporary features; save completed head/optimizer/RNG state |
| Head warmup, cache hit | Restore compatible head and training state | Skip warmup extraction, training, and validation |
| Joint training | Train adapters and head at fixed `learning_rate` and `head_learning_rate`; retain the head's AdamW state | Fresh backbone extraction and head validation after each joint epoch |
| Full-data build | Same training stages on all original IDs from both partitions | No validation; save adapters/head after the final epoch |

The shared warmup section supplies epochs, head batch size, learning rate,
extraction budget, and smoothing. Inline warmup's remaining model/head settings
come from the selected LoRA section. `warmup_epochs: 0` skips warmup extraction
and cache I/O. See [standalone versus inline settings](configuration.md#standalone-warmup).

Warmup features are extracted once for needed train/validation IDs, stay in CPU
memory, and are released before joint training. They are not persisted. For a
57,477-ID, 2,560-dimensional backbone, both branches' raw feature tensor is about
0.55 GiB in BF16 or 1.10 GiB in FP32, excluding loader batches and metadata.

Both stages normalize loss by the actual rows in each optimizer update, then
clip gradients and apply optimizer/scaler handling once per update. A short
final group merges into the previous full group; a smaller-than-one-group epoch
still updates once. For example, 27 warmup rows with batch size 8 use forward
batches of 8, 8, 8, and 3, but update groups of 8, 8, and 11 rows.

| Stage | Target rows per update | Updates per nonempty epoch |
| --- | --- | --- |
| Warmup | `warmup_training_batch_size` | `max(1, rows // warmup_training_batch_size)` |
| Joint | `G = training_batch_size * gradient_accumulation_steps` | `max(1, rows // G)` |

`gradient_accumulation_steps` applies only to joint training. Both stages use
configured label smoothing for training and unsmoothed validation loss.
There is no early-stopping checkpoint selection: holdout publication uses the
final epoch. Reused warmup contributes no old losses, timings, or executed epochs;
joint epoch numbers retain their configured warmup offset.

For warmup identity and restoration details, see [warmup caches](caching.md#lora-head-warmup).

## Two-stage validation and submission

Each joint-validation call extracts canonical A/B features using the current
adapters, then runs the head over original IDs. Submission does the same after
restoring its checkpoint. Backbone extraction uses the selected LoRA section's
`extraction_budget`; head inference uses `head_inference_batch_size`.

Extraction completes before head inference, stores temporary CPU features, and
releases them on return. Features are never reused across joint epochs/runs.
Warmup validation instead reuses that stage's frozen features. Logs distinguish
warmup, joint validation, and submission; joint-validation timing includes both
extraction and head inference. Validation/submission still work when warmup is
disabled. See [extraction packing](configuration.md#extraction-token-budgets).

## Optional token-budget training batches

Two passes control joint micro-batches. Both use padded A/B token cost
`rows * (longest_A + longest_B)`, independently of extraction budgets.

### Initial plan and checkpointing threshold

With `gradient_checkpointing_threshold: 0`, initial batches use
`training_batch_size` and all joint batches use non-reentrant gradient
checkpointing. A positive threshold enables a reusable global plan:

1. Select original training IDs, or all original IDs for a build.
2. Compare stable sorts by `(A_length, B_length)`, `(B_length, A_length)`, and
   `(A_length + B_length, A_length, B_length)`, using sample identity as tie-breaker.
3. Divide each candidate into groups of `G` rows and a possible short tail.
4. Within each group, best-fit pack longest-total-length rows first under the
   threshold. Choose the candidate with fewest micro-batches, then least padding.

This is a heuristic, not a global optimum. A single pair above the threshold
remains alone. At or below a positive threshold, checkpointing is disabled;
above it, checkpointing is enabled for **both branches through backward**.
Warmup and inference disable checkpointing.

### Epoch groups and the second packing pass

For positive thresholds, base group membership is cached. Each zero-based joint
epoch shuffles full groups with `random_state + joint_epoch`, then merges any
short tail into the last shuffled full group. That merged update stays last;
a lone short group runs alone.

A positive `gradient_checkpointing_budget` then combines only checkpointed
**singleton** micro-batches inside each finalized update. It uses descending
total length, identity tie-breaks, and best fit. Existing multi-row batches and
non-checkpointed singletons remain intact. Rows never cross update boundaries.
Combined cost may equal the budget; an oversized pair still runs alone. The
budget need not exceed the threshold.

The second pass also works with threshold `0`: initial fixed batches are seeded
by `random_state + joint_epoch` and repacked in memory. A zero budget disables
this pass and preserves initial within-update batch ordering. The merged update
remains last. Epoch batches are memoized so length queries match iteration
without consuming extra shuffle state.

Dynamic micro-batches can exceed `training_batch_size`. Update counts and loss
normalization use finalized row groups, not micro-batch counts. Only initial
plans are persisted; second-pass packing is not part of the plan key. See
[plan compatibility](caching.md#lora-batch-plans).

Packing can change accumulation order, dropout draws, and attention kernels.
Throughput and peak-memory improvements depend on the workload and GPU; preset
budgets are not benchmark guarantees.

## Training progress

One counter spans executed warmup/joint epochs and their validation batches,
logging approximately 5% milestones. Warmup updates and warmup validation
batches each count as **0.5** work units; joint updates and validation batches
count as **1**. Dynamic micro-batches do not change update-group work counts;
FP16-skipped updates still advance progress.

For a holdout, let `W` and `J` be the warmup/joint update counts above and
`V = ceil(original_validation_rows / head_inference_batch_size)`.
Work is `0.5 * executed_warmup_epochs * (W + V) + epochs * (J + V)`.
Builds use all training IDs and `V = 0`; standalone warmup has only half-weight
warmup updates. Reused warmup contributes zero work.

Extraction, preparation, caches, saving, and publication are excluded. See
[shared progress behavior](models.md#training-progress) and
[timing metrics](reports.md#timing-metrics). Python callers may pass a shared
`TrainingProgress` to `LoRATrainer.fit()`; otherwise one is created for that call.

## Memory tuning

Adjust the setting for the stage that actually failed:

| Stage | Main controls |
| --- | --- |
| Warmup backbone extraction | Lower `warmup.extraction_budget` |
| Warmup head training | Lower `warmup.warmup_training_batch_size` |
| Joint backbone training | Lower a positive checkpointing threshold; lower the second-pass budget or set it to `0` |
| Fixed initial joint batches, threshold `0` | Lower `training_batch_size`; adjust accumulation if retaining effective batch size |
| Joint-validation / submission extraction | Lower the selected LoRA section's `extraction_budget` |
| Head validation / submission | Lower `head_inference_batch_size` |

With a positive threshold, keeping `training_batch_size * gradient_accumulation_steps`
fixed does not lower packing limits; change the token controls instead. A budget
is not an input cap: oversized single samples still run alone. Keep `max_length`
large enough for required inputs. FP16/BF16 can reduce memory where supported;
checkpointing affects joint training only. The implementation reports OOM
rather than silently changing precision or shortening inputs.

## Checkpoints

`--build` atomically writes `models/SECTION/adapter.pt`, or `adapter_TAG.pt` for
a checkpoint tag. It contains adapters, the FP32 head, and compatibility metadata;
**base weights and optimizer state are not stored**. It is an inference artifact,
not a checkpoint for resuming interrupted joint training.

Loading checks model name, serializer, classifier width/dropout, head
architecture, LoRA rank/alpha/dropout/targets, schema/classes, pooling/truncation
policy, bias, and `training_quantization: none`. Head tensor names/shapes are
checked before backbone loading. Hub inference reloads the saved base revision.
Local base weights are neither embedded nor content-fingerprinted by the
production checkpoint.

`max_length` and internal `ab_swap` are recorded but not checked. Device,
precision, attention, TF32, batches, and accumulation may change; inputs must
still fit the current length limit. Historical quantized or retired-head
checkpoints are incompatible. See [migration](migration.md).
