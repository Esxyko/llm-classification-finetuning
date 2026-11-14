# SR2 LoRA training

[Documentation index](../README.md)

SR2 trains adapters and a three-class comparison head over
one Skywork reward-trained Qwen3 backbone. Original base parameters remain frozen;
training is non-quantized on one GPU. Each pair is serialized as separate A and B
branches and pooled at the last real token. Both stages train on **original rows
only**; validation makes one head prediction per ID.

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
head. Its comparison head shares L/M projections across wins and reads the signed
tie vector `(A-B)(L+M)` before taking the scalar readout's absolute value.
The implementation and public Python APIs now live in
`models/SR2_Qwen3_4B_lora/`. Head size, rank, smoothing, rates, and epochs may
differ between presets; read their actual values.

SR2 and the standalone `warmup` command use the same SR2 cache namespace.
The shipped presets select Skywork for both; reuse additionally requires
matching training IDs, labels, and effective model/head settings.

## Swap-equivariant head

For pooled row vectors $A,B$ of width $D$ and classifier width $h$:

$$
\begin{aligned}
P_A &= \operatorname{GELU}(AL-BM+B_1),\\
P_B &= \operatorname{GELU}(BL-AM+B_1),\\
d &= (AL+AM)-(BL+BM) = (A-B)(L+M),\\
z_A &= (P_A\odot m_P)R,\\
z_B &= (P_B\odot m_P)R,\\
b_{\mathrm{tie}} &= \operatorname{softplus}(\mathrm{raw\_b\_tie}),\\
z_{\mathrm{tie}} &= b_{\mathrm{tie}}-|du|.
\end{aligned}
$$

Here $L,M\in\mathbb{R}^{D\times h}$ and
$R,u\in\mathbb{R}^{h\times 1}$. The trainable vector $u$ has no sign constraint.
PyTorch stores the transposed linear weights; `u` has shape `[1, h]`
and `raw_b_tie` is a scalar. L/M/R/u are sampled independently from zero-mean uniform
distributions $U(-\sqrt{3}\sigma,\sqrt{3}\sigma)$ with these standard deviations:

| Matrices | Standard deviation $\sigma$ |
| --- | --- |
| $L,M$ (two `input_projection.weight` blocks) | $1/\sqrt{2D}$ |
| $R$ (`win_readout`) | $0.1/\sqrt{h}$ |
| $u$ (`u`) | $0.1/\sqrt{h}$ |

Each bound is computed from the corresponding input width, $D$ or $h$.
$L$ and $M$ are shared across win and tie branches. The trainable hidden bias
$B_1\in\mathbb{R}^h$ (`win_projection_bias`) is initialized to zero.
For the tie readout, sample $u\sim U(-a,a)$ with $a=0.1\sqrt{3/h}$ and retain
the sampled values directly, without an absolute value, magnitude floor, or
softplus mapping. The effective scalar bias starts at $10^{-4}$; initialize
`raw_b_tie` using the stable inverse softplus
$\operatorname{invsoftplus}(x)=x+\log(-\operatorname{expm1}(-x))$ at that value.
There is no tie projection bias.
The head has $2Dh+3h+1$ trainable parameters.
The tie vector, raw tie bias, and win bias use the existing head learning rate
and weight decay.

The tie feature $d$ is the signed difference of the win preactivations; their
common bias cancels. Win mask multiplication acts elementwise.
Each sample's win branches share the standard dropout mask $m_P$ with
$1/(1-p)$ scaling during training; evaluation disables it.
Training and inference both read the signed $d$ with $u$, then subtract the
absolute scalar from $b_{\mathrm{tie}}$, without GELU or dropout in the tie branch.
Components can cancel during the scalar readout, and the tie logit is at most
$b_{\mathrm{tie}}$.
Swapping inputs exchanges win logits and preserves the tie logit under the same
masks.

The implementation batches A/B as `[batch, 2, D]` and stores $L^\top,M^\top$
as contiguous blocks of one bias-free `Linear(D, 2h)`. One call computes both
projections for both inputs; M's branch axis is reversed before subtraction. $B_1$
is added in the projection output dtype. GELU, shared-mask multiplication, and
the R readout operate on both win branches together.

Every forward computes $L+M$ from the FP32 parameter blocks and $A-B$ from
FP32 inputs with autocast disabled. One bias-free `F.linear` then projects
the input difference using the existing head mixed-precision policy, retaining
the signed output $d$ for the scalar readout. Gradients
flow through the weight sum and input difference to both weight blocks and
both inputs.
This algebraic simplification halves the tie projection's input rows compared
with projecting A/B separately, with possible floating-point rounding differences.
The four linear calls compute the packed L/M projection, the L+M projection of
A-B, the shared win readout, and the tie readout. Weight views avoid concatenating
full L/M weights per call.

All head parameters remain FP32. Every forward computes the effective tie bias
in FP32 with autocast disabled, retaining gradients to `raw_b_tie`. Both readouts
follow the existing head mixed-precision policy. The tie scalar is converted to
FP32 before its absolute value is subtracted from the FP32 bias.
Returned logits are FP32 in the order A-win, B-win, tie. Swapping A/B preserves
the tie logit because $d$ changes sign; identical inputs give a tie logit of
$b_{\mathrm{tie}}$.

L/M/R/u and the win bias are trained directly; the tie bias trains and saves
`raw_b_tie`. The architecture marker is
`swap_equivariant_abs_diff_v8`, with embedding combination
`cross_projected_signed_diff_abs_readout`; adapter/head checkpoint schema is v2,
frozen-head schema is v1, and warmup schema is v5. The marker selects different
warmup cache keys and rejects earlier checkpoints without converting their
weights. Exact parameter-name/shape checks additionally reject earlier layouts,
including v7's `raw_w_tie`, v5's `tie_readout.weight` and `tie_bias`, and v4's
tie projection bias.
Rebuild those checkpoints with the matching configuration; warmup is regenerated.
The initialization policy is versioned separately as
`scaled_uniform_signed_tie_readout_v4` in the SR2 warmup identity: L/M/R retain
their scales, $B_1$ starts at zero, $u$ retains its signed uniform samples, and
the raw tie bias encodes the effective initial value above. The new architecture
marker isolates old warmup states without a storage schema change.
See [warmup identity](caching.md#identity) and the
[v7-to-v8 migration](migration.md#current-sr2-v8-head-2026-11-14).
There is no `A/B_swap` setting or orientation averaging for SR2.

## Hardware and input limits

Validation/build uses [`gpu.device`](configuration.md#gpu-configuration). All
branches are right-padded independently without padding masks. Qwen's causal attention
protects real tokens from right padding, and CPU-computed last-real-token
positions select pooled features. KV caching is disabled. Joint training encodes
A then B while retaining both gradient graphs. The head consumes both features,
and one backward pass computes gradients for the head and shared adapters.

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
encoded without additional special tokens. Full-data builds validate complete
input lengths before loading backbone weights. Plain-text checkpoints require
rebuilding.

| Precision | Backbone / pooled CPU features | Trainable state and head computation |
| --- | --- | --- |
| `bf16` | BF16 residuals, checkpoint inputs, and features | FP32 parameters, gradients, and AdamW moments; BF16 head autocast; FP32 logits |
| `fp16` | FP16 backbone compute; FP32 pooled features | FP32 head/state with gradient scaling |
| `fp32` | FP32 | FP32 head/state |

BF16 support is checked on each selected GPU. BF16 LoRA models disable PEFT's
input dtype upcasting once after adapter initialization, fixing the policy for
the model instance across encoding and checkpoint recomputation without repeated
model traversal. Adapter parameters remain FP32. Other precisions and standalone
warmup retain their existing input dtype behavior. RMSNorm retains FP32 variance
calculation.
Absolute values, Softplus transforms, and final logit arithmetic remain FP32 in every mode.

With `sdpa`, LoRA permits Flash, memory-efficient, and cuDNN backends and disables
math fallback during warmup, joint training, validation, and
checkpoint recomputation. Unsupported fused inputs fail explicitly. Scoped flags
are restored on exit; explicit `eager` attention is available. Training/validation
keep their native GQA policy.

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

The longest branch is A for ID `2995198009`. All current presets set both warmup
and SR2 `max_length` to 13,866. New training inputs must still pass the normal
length checks. The matching saved split and its fingerprint are documented under
[data preparation](data-preparation.md#saved-sr2-split-2026-10-27).

### Historical plain-text inputs

These are **historical measurements**, not a guarantee for the current saved
split. A 2026-10-20 snapshot used
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
offsets/metadata. Current training inputs are checked independently.

## Optimization and head warmup

| Stage | Trainable work | Validation / persistence |
| --- | --- | --- |
| Head warmup, cache miss | Load cached frozen features and extract missing branches with adapter forwards disabled; train only the head with epoch-based exponential decay from `warmup.learning_rate` to `warmup.build_min_lr` across `warmup.build_epochs`, keeping the rate fixed within each epoch | Evaluate the full training set and, for holdout runs, validation after each epoch using shared features; save completed head/optimizer/RNG state |
| Head warmup, complete cache hit | Restore compatible head and training state | Inline LoRA skips all warmup work; standalone warmup loads cached features, extracts missing branches, and recomputes both losses without training |
| Head warmup, lower-epoch cache hit (fixed rate only) | Load/extract frozen features, reconstruct shuffle state, and restore compatible head/optimizer/RNG state; train only remaining epochs | Evaluate after each remaining epoch and save a separate target-epoch cache, preserving the source |
| Joint training | Clear AdamW moments and step counters once at entry, then train adapters and head with one shared cosine-decayed `learning_rate` | Fresh backbone extraction and head validation after each joint epoch |
| Full-data build | Same training stages on all original IDs from both partitions | No validation; save adapters/head after the final epoch |

The shared warmup section supplies build epochs, head batch size, initial and
minimum learning rates, extraction budget, and smoothing.
Inline warmup's remaining model/head settings come from the selected LoRA section.
`warmup.build_epochs: 0` skips warmup extraction
and cache I/O. See [standalone versus inline settings](configuration.md#standalone-warmup).
Exponential decay reuses only an exact target-epoch cache. Fixed-rate warmup
(`warmup.build_min_lr` equal to `warmup.learning_rate`) can also continue from the largest compatible lower
epoch; all other cache identity fields must match.

Every path into joint training clears the optimizer state before its first update,
including fresh warmup, complete cache hits, and partial-cache continuation.
The trained head weights, optimizer parameter groups, GradScaler, and RNG state
are preserved at this transition. Subsequent joint epochs retain their accumulated
AdamW state. Warmup caches still save optimizer state for warmup continuation.

Adapter-free warmup features are persisted by branch token content and reused
across standalone/inline runs and head configurations. Only missing unique
branches are extracted. The current CPU tensor is shared by the shuffled
training loader and fixed-order train/validation inference loaders; its views
are released before joint training. See [warmup features](caching.md#warmup-features)
for compatibility and refresh rules. For a 57,477-ID, 2,560-dimensional backbone,
both branches' raw feature tensor is about
0.55 GiB in BF16 or 1.10 GiB in FP32, excluding loader batches and metadata.

Both stages normalize loss by the actual rows in each optimizer update, then
clip gradients and apply optimizer/scaler handling once per update. A short
final group merges into the previous full group; a smaller-than-one-group epoch
still updates once. For example, 27 warmup rows with batch size 8 use forward
batches of 8, 8, 8, and 3, but update groups of 8, 8, and 11 rows.

| Stage | Target rows per update | Updates per nonempty epoch |
| --- | --- | --- |
| Warmup | `warmup.training_batch_size` | `max(1, rows // warmup.training_batch_size)` |
| Joint | `G = training_batch_size * gradient_accumulation_steps` | `max(1, rows // G)` |

`gradient_accumulation_steps` applies only to joint training. Both stages use
configured label smoothing for training and unsmoothed validation loss.
Warmup `train_loss` is recomputed over the complete training set after all
optimizer updates in each epoch, with the epoch's final weights, dropout disabled,
and warmup label smoothing. It sums losses across all rows and divides by the
actual row count, including a short final inference batch. Joint `train_loss`
remains the row-weighted average observed during optimization.
There is no early-stopping checkpoint selection: holdout publication uses the
final epoch. Reused warmup contributes no old losses, timings, or executed epochs;
joint epoch numbers retain their configured warmup offset.

`LoRATrainingHistory.executed_warmup_epochs` and `executed_joint_epochs` count
only epochs executed by the current call, including zero warmup epochs for
an exact hit or disabled warmup. Full-data builds expose both counters through
`ModelBuildResult` and the CLI summary; `ModelBuildResult.epochs` remains their
sum. Its optional stage counters default to `None` for other callers.

For warmup identity and restoration details, see [warmup caches](caching.md#lora-head-warmup).

### Learning-rate decay

Frozen SR2 builds and standalone/inline warmup use exponential decay
once per epoch. LoRA joint training uses an independent cosine curve once per
planned optimizer update.

For head training, let `N = build_epochs`, `L = learning_rate`, and
`M = build_min_lr` in either the frozen SR2 or warmup section. For `N > 1`:

```text
epoch_multiplier = (M / L) ** (1 / (N - 1))
decay_rate = (ln(L) - ln(M)) / (N - 1)
lr = L * exp(-decay_rate * epoch_index)
```

The epoch index spans `0` through `N - 1`. The scheduler explicitly sets the first
rate to `L` and the final rate to `M`. Each epoch keeps that rate throughout,
including FP16-skipped updates. A single epoch uses `L`; `M = L` keeps every epoch
fixed. For example, `L = 0.001`, `M = 0.0002`, and `N = 4` yield approximately
`0.001`, `0.000584804`, `0.000341995`, and `0.0002`.

`build_min_lr` defaults to the corresponding initial head rate divided by five
and must be finite and in `(0, L]`. Frozen SR2 builds map `build_epochs`,
`training_batch_size`, `learning_rate`, and `build_min_lr` to the shared internal
`warmup_build_epochs`, `warmup_training_batch_size`, `warmup_learning_rate`, and
`warmup_build_min_lr` settings. Standalone and inline warmup share the same
mapping. Configurations, snapshots, checkpoint training metadata, and decaying
cache identities store the minimum rate; only the scheduler derives its
coefficient. `warmup.build_epochs: 0` skips the stage without creating a scheduler.

Frozen SR2 validation uses ReduceLROnPlateau on the full unsmoothed validation
loss after each epoch, with `factor: 0.95` and `patience: 2` as defaults and fixed
`threshold=0` and `min_lr=0`. Reductions apply to the following epoch. Training
continues until `early_stopping` consecutive epochs do not lower validation
loss, independently of the scheduler's counter. It ignores the build schedule
and uses no head-state cache; see
[frozen SR2 validation](models.md#frozen-skywork-reward-backbone).

Decaying builds/warmup reuse only exact target-epoch caches. Changing the minimum
rate or target epoch count changes the full curve and trains from epoch index
`0` on a cache miss. Lower-epoch continuation remains supported only when
`build_min_lr = learning_rate`.

Joint training uses `N = epochs * max(1, rows // G)`, using the joint
optimizer-group size `G` above. For planned update index `i` from `0` through
`N - 1`, when `N > 1`, its formula is unchanged:

```text
p = i / (N - 1)
lr = learning_rate * (min_lr_ratio + (1 - min_lr_ratio) * (1 + cos(pi * p)) / 2)
```

For joint training, the first update uses the initial rate and the last uses
the minimum. Its `min_lr_ratio` defaults to `0.125`; `0` reaches zero and `1`
keeps the rate fixed. A run with only one update uses the initial rate. The
schedule applies immediately before each complete optimizer group's update attempt and
advances once for that consumed group, including when FP16 gradient scaling
skips the parameter update. Micro-batches and evaluation do not advance a
schedule; warmup does not advance the joint schedule. Packing and short-tail
merging retain the update counts above; epochs do not restart decay.

At the start of joint training, both groups switch to `learning_rate` and begin
a fresh cosine curve, whether warmup was executed, restored from cache, or
disabled. Adapters and the classifier head retain separate optimizer parameter
groups with the same joint learning rate. The classifier's AdamW state is
retained. These joint-only settings do not change warmup cache identity or
production checkpoint compatibility.

## Two-stage validation

Each joint-validation call extracts canonical A/B features using the current
adapters, then runs the head over original IDs. Backbone extraction uses the
selected LoRA section's `extraction_budget`; head inference uses `head_inference_batch_size`.

Extraction completes before head inference, stores temporary CPU features, and
releases them on return. Features are never reused across joint epochs/runs.
Warmup train and validation inference instead reuse that stage's frozen features.
Inline warmup uses `head_inference_batch_size` for both evaluations; standalone
warmup uses `warmup.training_batch_size`. Logs distinguish warmup training
evaluation, warmup validation, and joint validation; joint-validation timing
includes both extraction and head inference. Validation still works when warmup is
disabled. See [extraction packing](configuration.md#extraction-token-budgets).

## Optional token-budget training batches

Two passes control joint micro-batches, independently of extraction budgets.
For `n` rows with independently padded branch lengths `L_A` and `L_B`, both use
the total padded A/B token cost `P = n * (L_A + L_B)`.
Planning, cached-plan validation, and runtime checkpointing share this cost.
Runtime adds the two branches' actual padded tensor sizes.

### Initial plan and checkpointing threshold

With `gradient_checkpointing_threshold: 0`, initial batches use
`training_batch_size` and all joint batches use non-reentrant gradient
checkpointing. A positive threshold enables a reusable global plan:

1. Select original training IDs, or all original IDs for a build.
2. Compare three stable ascending sorts by `(A_length, B_length)`,
   `(B_length, A_length)`, and `(A_length + B_length, A_length, B_length)`,
   breaking ties by sample ID and then orientation.
3. Divide each candidate into groups of `G` rows and a possible short tail.
4. Within each group, pack under `P <= threshold` using the shared packing rule
   below. Choose the candidate with the fewest micro-batches, then the lowest
   total `P`. Exact ties retain the first candidate in the order above.

Both passes compare two candidates within each group, choosing the fewest
micro-batches, then the lowest total `P`:

- A suffix dynamic program finds the optimal contiguous partition of rows that
  individually fit the applicable limit. The initial pass preserves the current
  global sort's order within each group; the second pass uses descending
  `A_length + B_length`, then ascending sample identity. Rows individually above
  the limit stay alone and do not interrupt the eligible sequence. Interval
  maxima are updated incrementally; extension stops as soon as `P` exceeds the
  limit. Equal DP scores retain the first enumerated cut, shortest prefix first.
- The retained best-fit candidate processes rows by descending
  `A_length + B_length`, then ascending sample identity. Each row joins the
  feasible micro-batch with the smallest remaining budget, breaking ties by
  creation order; if none fits, it starts a new micro-batch. Exact ties between
  the two candidates retain best-fit.

DP is optimal for contiguous cuts in the specified order; the overall packing
remains a heuristic for unrestricted assignments. Both candidates use the same
`P`, with extra DP space linear in group size. The DP and best-fit each take
quadratic time in the worst case; over-limit intervals and permanently closed
best-fit batches are skipped. When all eligible rows fit in one micro-batch,
the packer returns that result directly in best-fit order.

A single pair above the threshold remains alone. When `P` is at or below a
positive threshold, checkpointing is disabled; above it, checkpointing is enabled
for **both branches through backward**.
Both branch graphs remain live until the single backward pass. Setting the
threshold too high can exceed GPU memory because non-checkpointed batches keep
both branches' full activations. Lower the positive threshold, or set it to `0`
to force gradient checkpointing on all joint batches.
Warmup and inference disable checkpointing.

### Epoch groups and the second packing pass

For positive thresholds, base group membership is cached. Each zero-based joint
epoch shuffles full groups with `random_state + joint_epoch`, then merges any
short tail into the last shuffled full group. That merged update stays last;
a lone short group runs alone.

A positive `gradient_checkpointing_budget` then combines only checkpointed
**singleton** micro-batches inside each finalized update, using the shared
packing rule with `P <= budget`. For positive thresholds, eligible singletons
have `A_length + B_length > threshold`; combining them keeps checkpointing
enabled. Existing multi-row batches and non-checkpointed singletons remain
intact. Neither pass crosses a finalized update boundary. Combined `P` may equal
the budget; an oversized pair still runs alone without truncation. For positive
thresholds, the budget must exceed `2 * threshold` to combine any two singletons;
a smaller budget leaves all eligible singletons running alone.

The second pass also works with threshold `0`: all micro-batches run with
checkpointing enabled. In this mode, initial micro-batches of size 1 (e.g. from
`training_batch_size: 1`) are treated as singletons and repacked in memory under
`P <= budget`; existing multi-row initial batches (`training_batch_size > 1`)
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

Progress has three independent stages. Each starts its own timer and work counter,
accumulates across all of its epochs, and finishes with one 100% log after its
work and result checks succeed. Failures do not force completion.

| Stage | Included work | Logging interval |
| --- | --- | --- |
| **Warmup feature extraction** | All branch batches for the deduplicated train/validation IDs, extracted once before head training | Approximately 20% |
| **Head warmup** | All warmup epochs, including optimizer groups, full-training-set evaluation, and validation | Approximately 20% |
| **Joint training** | All joint epochs, including optimizer groups, validation feature extraction, and validation head inference | Approximately 10% |

Each completed optimizer group, extraction branch batch, or head inference batch
counts as **1** work unit. There are no warmup/joint weighting factors.
Dynamic training micro-batches do not change update-group work counts;
FP16-skipped updates still advance progress. Extraction milestones use the actual
loader length: 46 branch batches log at batches 10, 19, 28, 37, and 46, including
the final 100% log.

For an inline holdout run, let `W` and `J` be the warmup/joint update counts above,
`T = ceil(training_rows / head_inference_batch_size)`, and
`V = ceil(original_validation_rows / head_inference_batch_size)`, and `E` be the
number of branch batches in each joint-validation extraction loader. Warmup
extraction counts its own loader's batches; Head warmup totals
`remaining_warmup_epochs * (W + T + V)`, where remaining epochs exclude those
restored from cache; Joint training totals `epochs * (J + E + V)`.
Full-data model builds use all training IDs and `E = V = 0`.

Standalone warmup uses `warmup.training_batch_size` for the denominators of
both `T` and `V`. A head miss runs Head warmup, without Joint training. Warmup
feature extraction precedes it only when unique branches are missing. A complete
head hit logs cache reuse and runs a separate **Cached head evaluation** stage with
`T + V` work and approximately 20% milestones. It executes zero training epochs.
Complete inline cache hits log reuse without creating progress records for either
skipped warmup stage. Partial head hits in either mode run Head warmup for only
the remaining epochs, with epoch numbers continuing from the source; extraction
counts only missing branches. Complete feature hits have no extraction stage.
Disabled warmup also enters Joint training directly.

Logs retain `time: HH:MM:SS/HH:MM:SS`, meaning **elapsed/estimated_total**. The
right-hand duration is the current stage's estimated total, calculated as
`stage_elapsed * stage_total_work / stage_completed_work`, rather than time
remaining. At 100%, both durations equal that stage's actual elapsed time. Stage
transitions reset all work and timing; neither earlier stages nor cache reuse
contribute to the estimate. CPU preparation, cache I/O, saving, and publication
are outside the stage boundaries. Warmup extraction starts after its loader is
prepared; Head warmup starts after its feature views and loaders are ready and
ends before saving the warmup cache; Joint training starts after transition setup.

See [shared progress behavior](models.md#training-progress) and
[timing metrics](reports.md#timing-metrics). Python callers may still pass a
`TrainingProgress` to `LoRATrainer.fit()`; it is reset at each stage boundary.
Otherwise an initially inactive counter is created for that call. The shared
counter supports `start_stage(stage=..., total_steps=..., milestone_percent=...)`
and `finish_stage(description)`. The optional `update_interval_seconds` argument
on `start_stage` selects time-based updates with millisecond precision instead of
percentage milestones. `finish_stage(description, emit=False)` verifies all work
completed without printing. Frozen SR2 validation uses silent per-epoch counters
and emits summaries every `early_stopping` epochs, plus the stopping epoch.
Existing construction with `total_steps` retains
the legacy 5% logging interval and automatic completion.

## Memory tuning

Adjust the setting for the stage that actually failed:

| Stage | Main controls |
| --- | --- |
| Warmup backbone extraction | Lower `warmup.extraction_budget` |
| Warmup head training | Lower `warmup.training_batch_size` |
| Joint backbone training | Lower a positive checkpointing threshold; lower the second-pass budget or set it to `0` |
| Fixed initial joint batches, threshold `0` | Lower `training_batch_size`; adjust accumulation if retaining effective batch size |
| Joint-validation extraction | Lower the selected LoRA section's `extraction_budget` |
| Head validation | Lower `head_inference_batch_size` |

With a positive threshold, keeping `training_batch_size * gradient_accumulation_steps`
fixed does not lower packing limits; change the token controls instead. A budget
is not an input cap: oversized single samples still run alone. Keep `max_length`
large enough for required inputs. FP16/BF16 can reduce memory where supported;
checkpointing affects joint training only. The implementation reports OOM
rather than silently changing precision or shortening inputs.

## Checkpoints

`--build` atomically writes `models/SECTION/adapter.pt`, or `adapter_TAG.pt` for
a checkpoint tag. It contains adapters, the FP32 head, and compatibility metadata;
base weights are exported separately as described below. Optimizer state is
not stored. These are inference artifacts, not checkpoints for resuming
interrupted joint training.

Before initializing CUDA, full-data builds prepare their data/tokens/batch plan
and probe both the adapter/head and merged-backbone destinations for writability.
Saving reloads the staged file on CPU to validate its metadata and tensors,
then atomically replaces the destination. Failed writes or staged
validation preserve the previous checkpoint and clean up the temporary file.
The CPU snapshot used for serialization is released before readback; neither
the probe nor readback loads base weights or runs inference.

CPU readback checks model name, serializer, classifier width/dropout, head
architecture, LoRA rank/alpha/dropout/targets, schema/classes, pooling/truncation
policy, bias, and `training_quantization: none`. It also checks head tensor
names/shapes and records the saved Hub base revision. Local base weights are
neither embedded nor content-fingerprinted by the production checkpoint.

`max_length` and internal `ab_swap` are recorded but excluded from readback
compatibility comparisons. Device, precision, attention, TF32, batches, and
accumulation are also outside these comparisons. Historical quantized or
retired-head checkpoints are incompatible. See [migration](migration.md).

### Merged backbone export

Every SR2 `--build` also writes `models/SECTION/merged/`, or `merged_TAG/` with
the same checkpoint tag. The adapter/head checkpoint is saved and validated
first, before the PEFT model is consumed by export. Build switches the model to
evaluation mode, moves the existing backbone to CPU, and calls
`merge_and_unload(safe_merge=True)`. The returned plain `AutoModel` is saved in
Safetensors format with `max_shard_size="5GB"`, preserving the backbone's build
precision and using its own parameter names rather than the source reward
model's prefixes. The training tokenizer is saved alongside the model config.
The classification head remains in `adapter[_TAG].pt`.

`export_metadata.json` records `model_name`, `base_revision`,
`training_source_sha256`, the corresponding `checkpoint_file` and its
`checkpoint_sha256`, `dtype`, and a UTC `created_at` timestamp. The checkpoint
digest identifies the exact adapter/head artifact paired with the export.

All files are written in a sibling staging directory. Validation reloads the
configuration and tokenizer using local files only, checks the tokenizer
vocabulary/special tokens/chat template, and opens each Safetensors file to
check its structure and tensor names, shapes, and dtypes against the merged
backbone. Shard-index coverage must be complete, and adapter layers/tensor names
must be absent. Validation does not load another copy of the backbone.

Replacement moves the previous export to `.merged-backup` (or
`.merged_TAG-backup`), promotes the staged directory, and removes the backup
after success. A failed promotion restores the previous directory when possible;
an unsuccessful restore reports the backup location. An interrupted promotion
that left only a backup is recovered during the next write preflight. When both
the destination and a leftover backup exist, preflight reports both paths for
review instead of discarding either. Temporary files are cleaned up after
failure, and cleanup errors do not replace the export error.

An export failure returns a command failure while retaining the already saved
adapter/head checkpoint. Both outputs must be saved before build reports success.
