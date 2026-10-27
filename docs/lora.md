# SR2 LoRA training

[Documentation index](../README.md)

SR2 trains adapters and a three-class comparison head over
one Skywork reward-trained Qwen3 backbone. Original base parameters remain frozen;
training is non-quantized on one GPU. Each pair is serialized as separate A and B
branches and pooled at the last real token. Both stages train on **original rows
only**; validation/submission make one head prediction per ID.

```shell
uv run model SR2-Qwen3-4B-lora
uv run model SR2-Qwen3-4B-lora --build --checkpoint-tag experiment
```

SR2 requires the `SR2_Qwen3_4B_lora` model section
and the shared `warmup` section. See [commands](commands.md) for execution modes
and [configuration](configuration.md) for the complete fields and parser defaults.

## SR2 reward-backbone profile

| Profile | Model section | Backbone in shipped presets | Checkpoint / warmup namespace |
| --- | --- | --- | --- |
| SR2 | `SR2_Qwen3_4B_lora` | `Skywork/Skywork-Reward-V2-Qwen3-4B` | `SR2_Qwen3_4B_lora` / `SR2_Qwen3_4B_lora_warmup` |

SR2 loads Skywork's Transformer through `AutoModel`, omitting its scalar reward
head. Its comparison head extends the architecture introduced by the retired
EXPT2 profile with separate win and tie readout biases. The implementation and
public Python APIs now live in `models/SR2_Qwen3_4B_lora/`. Head size, rank,
smoothing, rates, and epochs may differ between presets; read their actual values.

SR2 and the standalone `warmup` command use the same SR2 cache namespace.
The shipped presets select Skywork for both; reuse additionally requires
matching training IDs and effective model/head settings.

## Swap-equivariant head

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
`Linear(h, 1)` scores the three representations. After converting these scores
to FP32, the head adds `readout_bias`, a trainable FP32 vector of shape `[2]`
initialized to zero: A-win and B-win share its first element, and tie uses its
second element. The FP32 logits are returned in A-win, B-win, tie order. The
relative bias learns the win-versus-tie prior. Swapping inputs exchanges win
logits and preserves the tie logit under the same masks; evaluation disables
dropout. Biases use the existing head learning rate and weight decay.

These weights are trained directly. The architecture marker is
`swap_equivariant_tie_from_wins_v2`; adapter/head checkpoint schema is v2 and
warmup schema is v5. The new marker selects a different warmup cache key and
rejects historical v1 checkpoints without converting their weights. Rebuild
those checkpoints with the matching configuration; warmup is regenerated.
There is no `A/B_swap` setting or orientation averaging for SR2.

## Hardware and input limits

Validation/build uses `gpu.device`; submission uses the
[two-GPU layer split](configuration.md#gpu-configuration). All branches are
right-padded independently without padding masks. Qwen's causal attention
protects real tokens from right padding, and CPU-computed last-real-token
positions select pooled features. KV caching is disabled. Joint training uses
single-branch recompute: A is encoded without a graph (keeping only its features
and the pre-forward CPU/CUDA RNG state), B and the head run normally, and the
first backward yields head, B, and A-feature gradients. B's graph is then freed,
A is recomputed with its RNG state replayed (the training RNG is restored
afterwards), and the cached feature gradient completes A's backward into the
shared adapters. Draw order and masks match a two-graph pass; AMP, SDPA scopes,
and checkpointing are identical across the recompute. A costs one extra forward.

LoRA **never truncates**. Token preparation validates full canonical branches
before training, and inference checks its inputs before encoding. Empty or
`max_length`-exceeding branches fail with the ID, branch, actual count, and
required minimum limit. Lowering a configured ceiling does not reduce work for
inputs already below it. Shared full-token encodings remain reusable when
limits change, but length checks still run.

For Hub backbones, preparation resolves a fixed revision and uses it for the
tokenizer and weights. Token reuse depends on actual tokenizer content, not
just the model name/revision.

All stages use [SR2 chat-template inputs](data-preparation.md#conversation-serialization)
with no system message. Template output includes its terminal newline and is
encoded without additional special tokens. Submission checks checkpoint
compatibility first, then loads the tokenizer at the saved base revision and
validates full input lengths before loading backbone weights. Plain-text
checkpoints require rebuilding.

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

The 2026-10-27 SR2 rebuild uses `sr2-qwen3-chat-template-v1` and the official
Skywork tokenizer at revision `fd958fef475f323f4e6b195930e3dd918485c668`, encoded
with `add_special_tokens=False` and no truncation. These values cover both A/B
branches of all original training rows, before augmentation:

| Statistic | SR2 rebuild value |
| --- | ---: |
| Original IDs | 57,477 |
| Median branch | 297 tokens |
| 99th percentile | 2,231.94 tokens |
| Longest branch | 13,866 tokens |
| Canonical A+B token total | 44,873,727 |

The longest branch is A for ID `2995198009`. The local three-row test CSV has a
maximum branch length of 851 tokens and a total of 2,328 A+B tokens. All current
presets set both warmup and SR2 `max_length` to 13,866. New competition test data
must still pass the normal length checks. The matching saved split and its
fingerprint are documented under [data preparation](data-preparation.md#saved-sr2-split-2026-10-27).

### Historical plain-text inputs

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

Two passes control joint micro-batches, independently of extraction budgets.
For `n` rows with independently padded branch lengths `L_A` and `L_B`, both use
the same two costs:

- Peak branch load `M = n * max(L_A, L_B)` constrains packing and determines
  checkpointing.
- Total padded tokens `P = n * (L_A + L_B)` measures padding and computation
  across both branches and ranks packing choices.

Planning, cached-plan validation, and runtime checkpointing share these
definitions. Runtime costs use each branch's actual padded tensor size.

### Initial plan and checkpointing threshold

With `gradient_checkpointing_threshold: 0`, initial batches use
`training_batch_size` and all joint batches use non-reentrant gradient
checkpointing. A positive threshold enables a reusable global plan:

1. Select original training IDs, or all original IDs for a build.
2. Compare four stable ascending sorts by `A_length`, `B_length`,
   `A_length + B_length`, and `max(A_length, B_length)`, breaking ties by sample
   ID and then orientation.
3. Divide each candidate into groups of `G` rows and a possible short tail.
4. Within each group, pack under `M <= threshold` using the shared packing rule
   below. Choose the candidate with the lowest total `P`, then the fewest
   micro-batches. Exact ties retain the first candidate in the order above.

Both packing passes process rows by descending `max(A_length, B_length)`, then
descending total length, then ascending sample identity. For each row, consider
existing micro-batches whose resulting `M` fits the applicable budget. Choose
the smallest increase in `P`; ties prefer the smallest remaining branch budget,
then the earliest-created micro-batch. If none fits, create a new micro-batch.

This is a heuristic, not a global optimum. A single pair above the threshold
remains alone. When `M` is at or below a positive threshold, checkpointing is
disabled; above it, checkpointing is enabled for **both branches through backward**,
including A's no-grad forward, B's backward, and A's recompute. Because non-checkpointed
batches store full autograd activations across all backbone layers for one branch
at a time, setting the threshold too high can exceed GPU memory. Keep the positive
threshold low enough for non-checkpointed single-branch loads, or set it to `0`
to force gradient checkpointing on all joint batches.
Warmup and inference disable checkpointing.

### Epoch groups and the second packing pass

For positive thresholds, base group membership is cached. Each zero-based joint
epoch shuffles full groups with `random_state + joint_epoch`, then merges any
short tail into the last shuffled full group. That merged update stays last;
a lone short group runs alone.

A positive `gradient_checkpointing_budget` then combines only checkpointed
**singleton** micro-batches inside each finalized update, using the shared
packing rule with `M <= budget`. For positive thresholds, eligible singletons
have `max(A_length, B_length) > threshold`; combining them keeps checkpointing
enabled. Existing multi-row batches and non-checkpointed singletons remain
intact. Neither pass crosses a finalized update boundary. Combined `M` may equal
the budget; an oversized pair still runs alone without truncation. For positive
thresholds, the budget must exceed `2 * threshold` to combine any two singletons;
a smaller budget leaves all eligible singletons running alone.

The second pass also works with threshold `0`: all micro-batches run with
checkpointing enabled. In this mode, initial micro-batches of size 1 (e.g. from
`training_batch_size: 1`) are treated as singletons and repacked in memory under
`M <= budget`; existing multi-row initial batches (`training_batch_size > 1`)
remain intact. A zero budget disables only this repacking pass. Positive-threshold
plans still shuffle micro-batches inside each update after repacking. The merged
update remains last. Epoch batches are memoized so length queries match iteration
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
checkpointing affects joint training only. Single-branch recompute lowers peak
memory (one branch graph at a time) at the cost of an extra A forward, so
thresholds and budgets can be raised accordingly. The implementation reports OOM
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
