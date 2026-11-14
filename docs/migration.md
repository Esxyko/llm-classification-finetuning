# Migration from older workflows

[Documentation index](../README.md)

Use the current [configuration reference](configuration.md) for new runs.
These notes explain historical names and artifacts; they do not change saved
experiments or make old checkpoints compatible with a new architecture.

## Removed submission workflow

The project no longer provides submission inference. `model MODEL --submit` and
the older `--test` option have no current replacement command. Validation,
full-data builds, and standalone warmup remain available; build still exports
the frozen head or LoRA adapter/head plus merged backbone and tokenizer.

Remove `gpu.submit_devices` from custom YAML files: it is now an unknown key
and is rejected by the strict configuration loader. Current presets and new
snapshots omit it. `--checkpoint-tag` requires `--build`, and remains unsupported
for standalone warmup. Submission-only Python interfaces and the separate
`requirements-submit.txt` installation are removed.

`data` and `init` now download only missing `train.csv`. Existing test CSVs,
checkpoints, caches, results, and historical configuration snapshots remain
untouched. Reporting continues to exclude historical `submit` and `test`
directories from validation-run discovery.

## Five folds to one holdout

Replace `cross_validation` with `validation`, remove `n_splits`, set the desired
`validation_fraction` and `max_token_deviation`, and point `data.processed_path`
to `data/processed/train_validation.parquet`. Run `uv run preprocess` from raw
training data, or use the saved current manifest and volumes from Git.

Models share the saved grouped holdout. Old fold datasets/configurations are
rejected rather than interpreted as a holdout. Report synthesis rejects old
five-fold result directories; retain them for the corresponding historical code.

## Retired profiles

The active model selectors are `SR2-Qwen3-4B` (frozen head training, validation/build)
and `SR2-Qwen3-4B-lora` (validation/build); `warmup` is a separate
build-only utility. The new frozen SR2 profile uses the current Skywork backbone
and SR2 head, independently of the retired frozen Qwen profile.
On 2026-10-27, frozen `qwen3-4b` and
`qwen3-4b-lora-EXPT2` were retired. Other retired selectors include `qwen3-1.7b`,
`qwen3-1.7b-mono-input`, `qwen3-4b-qlora`, `qwen3-4b-lora`, and
`qwen3-4b-lora-EXPT`. They are not CLI choices or members of `ALL`.

The old `qwen3_4b`, `qwen3_4b_lora_EXPT2`, `qwen3_4b_lora`,
`qwen3_4b_lora_EXPT`, and `qwen3_4b_qlora` sections are unknown keys. The
corresponding `AppConfig` fields and public Qwen profile exports are removed,
with no aliases. Start from a current preset when moving to SR2; renaming
a section alone does not migrate its objective, warmup fields, or saved head.

Local archives, when present, live under the Git-ignored
`src/llm_classification_finetuning/models/archive/`. They are not included in a
fresh clone and are never imported by active profiles. The new `qwen3_4b/`
and `qwen3_4b_lora_EXPT2/` archives preserve original profile source, root
configuration, seven presets, and documentation snapshots. Their README files
describe retirement; the snapshots are reference material, not self-contained
runnable packages. Shared legacy head and quantization branches remain in the
SR2 implementation, without active registrations for those historical profiles.

## SR2 implementation and standalone warmup

The retained LoRA implementation and public exports moved from
`llm_classification_finetuning.models.qwen3_4b_lora_EXPT2` to
`llm_classification_finetuning.models.SR2_Qwen3_4B_lora`. Update Python imports
of `LoRAModelPipeline`, `LoRATrainer`, `LoRATrainingBatchRepository`,
`PreparedTrainingSplit`, `TrainingBatchPlan`, and `WarmupCacheContext` to that
package. The default Python profile is now SR2; its identity lives in `profile.py`.

That package move preserved the SR2 selector, config key, result prefix,
checkpoint/cache directories, head tensors, architecture marker
`swap_equivariant_tie_from_wins_v1`, adapter/head schema v2, batch-plan schema v4,
and warmup schema v5. The EXPT2 head definition was retained even though its
profile was retired; compatible checkpoints required no conversion at that stage.

The subsequent readout-bias update changes the SR2 head marker to
`swap_equivariant_tie_from_wins_v2` and adds `readout_bias`, with one shared
win bias and one tie bias. Storage schemas and artifact namespaces stay the
same. The v2 head rejects v1 checkpoints; rebuild them with the matching config.
Warmup uses new cache keys and is regenerated, leaving historical v1 files intact.

The harmonic/discrepancy head replacement introduced `swap_equivariant_harmonic_discrepancy_v1`:
shared cross projections feed harmonic fusion and an absolute-difference
correction, with separate R/S/T readouts and only one scalar output bias. All six
matrices use Xavier uniform with gain 1.0; hidden and output biases start at zero.
The adapter/head schema v2, warmup schema v5, configuration fields, and artifact
namespaces remain unchanged. Earlier tie-from-wins v1/v2 checkpoints are rejected
and require rebuilding. Warmup selects new cache keys; exact parameter/moment
structure checks also reject old states with relabeled metadata. Historical
weights, caches, results, and configuration snapshots are not converted or edited.
See the [current head definition](lora.md#swap-equivariant-head).

The next marker was `swap_equivariant_harmonic_discrepancy_v2`. Batched A/B
projections and the shared win readout are retained, while the relative tie
correction changes from `s - 2q + b` to `s - q + b`; win logits contain no `q`.
Parameter names/shapes and storage schemas are unchanged, but predictions for
the same weights change. V2 selects new warmup keys and rejects v1 production
checkpoints through the architecture marker. Rebuild for v2; historical v1
artifacts are preserved without automatic weight conversion.

The subsequent marker was `swap_equivariant_harmonic_discrepancy_v3`. L/M occupy
contiguous blocks of one bias-free input projection, while X/Y use a separate
projection of the FP32 parameter difference L-M plus the new shared B3. X/Y have
no GELU; their undropped values feed harmonic fusion and the discrepancy branch.
The single correction q now contributes to all outputs as
`[r_A + q, r_B + q, s - q + b]`. The only scalar output bias remains b.
V3 retains the scaled-uniform initialization version
`scaled_uniform_zero_discrepancy_v1` and initializes B3 to zero. Adapter/head
schema v2, warmup schema v5, configuration fields, and artifact namespaces remain
unchanged. Earlier harmonic/discrepancy v1/v2 and tie-from-wins checkpoints require
rebuilding; v3 selects new warmup keys and its exact tensor checks reject the
old layout even with relabeled metadata. Historical artifacts remain intact,
and token, adapter-free feature, and batch-plan caches retain their existing
compatibility rules.

### Archived SR2 v4 head (2026-11-09)

The historical architecture `swap_equivariant_penalized_harmonic_v4` used
`cross_projected_linear_diff_penalized_harmonic` features. Its complete formula,
for pooled row vectors $A,B\in\mathbb{R}^D$ and head width $h$, was:

$$
\begin{aligned}
P_A &= \operatorname{GELU}(AL-BM+B_1),\\
P_B &= \operatorname{GELU}(BL-AM+B_1),\\
X &= A(L-M)+B_3,\\
Y &= B(L-M)+B_3,\\
H &= \frac{4\left(X\odot Y-(X-Y)^{\odot 2}\right)}{|X|+|Y|+10^{-8}},\\
z_A &= (P_A\odot m_P)R,\\
z_B &= (P_B\odot m_P)R,\\
z_{\mathrm{tie}} &= (H\odot m_H)S+b.
\end{aligned}
$$

All products, powers, absolute values, and masks within H were elementwise.
The parameter mapping was:

| Formula | Parameter and shape |
| --- | --- |
| $L,M\in\mathbb{R}^{D\times h}$ | Two contiguous transposed blocks of `input_projection.weight`, each `[h, D]` |
| $B_1\in\mathbb{R}^h$ | `win_projection_bias` |
| $B_3\in\mathbb{R}^h$ | `tie_projection_bias` |
| $R,S\in\mathbb{R}^{h\times 1}$ | `win_readout.weight`, `tie_readout.weight`, each stored `[1, h]` |
| $b\in\mathbb{R}$ | FP32 scalar `tie_bias` |

The head had `2Dh + 4h + 1` trainable parameters and no q/discrepancy branch.
Initialization version `scaled_uniform_v2` sampled L/M/R/S independently from
zero-mean uniform distributions $U(-\sqrt{3}\sigma,\sqrt{3}\sigma)$, with
$\sigma=1/\sqrt{2D}$ for L/M and $\sigma=0.1/\sqrt{h}$ for R/S, using actual
layer input dimensions. B1, B3, and b started at zero. Biases used the existing
head learning rate and weight decay.

Each sample's wins shared the standard dropout mask m_P; H used an independent
m_H. Both masks used `1/(1-p)` scaling during training and were disabled at
evaluation. X/Y had no GELU or dropout, so H consumed their undropped values.
Swapping A/B exchanged win logits and preserved the tie logit under the same masks.

The implementation batched A/B as `[batch, 2, D]`, using one packed L/M
projection, one L-M projection of both inputs with shared B3, one shared win
readout, and one tie readout: four linear calls. L-M was computed from FP32
parameter blocks with autocast disabled, retaining gradients, before the X/Y
projection followed the head mixed-precision policy. B1 was added in the win
projection output dtype. X/Y were then converted to FP32; all H arithmetic,
including the fixed epsilon, ran with autocast disabled. Tie dropout preserved FP32.
Readouts followed the head mixed-precision policy and were converted to FP32
before b was added. Returned logits were FP32 `[A-win, B-win, tie]`.

Frozen SR2, LoRA SR2, standalone/inline warmup, and `submit.ipynb` used this
architecture. Its storage schemas were adapter/head v2, frozen-head v1, and
warmup v5. This is a documentation-only archive of v4; historical artifacts
remain intact.

### Archived SR2 v5 head (2026-11-09)

The v5 marker was `swap_equivariant_abs_diff_v5`, with embedding combination
`cross_projected_linear_abs_diff`. Wins retained the v4 branches and shared dropout
mask. The tie used `d = abs((A-B)(L-M))` and
`z_tie = tie_readout(d) + tie_bias` directly in both training and inference,
without tie GELU or dropout. Removing `tie_projection_bias` and the harmonic formula
reduced the head to `2Dh + 3h + 1` parameters.

The implementation forms A-B and L-M in FP32 with autocast disabled, then uses
one bias-free mixed-precision projection and FP32 elementwise absolute values.
This halves the tie projection's input rows, while retaining gradients and four
linear calls overall. Algebraically equivalent separate projections may round
differently. Frozen SR2, LoRA SR2, standalone/inline warmup, and `submit.ipynb`
all used v5; the notebook computed the head entirely in FP32.

Initialization used `scaled_uniform_v2` for the remaining weights and zero
B1/tie bias. Adapter/head schema v2, frozen-head schema v1, warmup schema v5,
configuration fields, and artifact namespaces remain unchanged. Earlier
checkpoints require rebuilding; the new architecture selects fresh warmup keys
without editing or migrating historical artifacts. Exact head/optimizer tensor
checks also reject the removed tie projection bias, even with relabeled metadata.
Token, adapter-free feature, and batch-plan caches retain their existing reuse
rules. See the [current head definition](lora.md#swap-equivariant-head).

### Archived SR2 v6 head (2026-11-09)

The v6 marker was `swap_equivariant_abs_diff_v6`, with embedding combination
`cross_projected_linear_abs_diff`. It used the v5 win branches and tie feature
`d = abs((A-B)(L-M))`. The tie readout used
`w_tie = -softplus(raw_w_tie)`, `b_tie = softplus(raw_b_tie)`, and
`z_tie = d @ w_tie + b_tie` (with the readout weight stored transposed).
Effective tie parameters are computed in FP32 with autocast disabled; the
readout follows the head mixed-precision policy before adding the FP32 bias.
The head retains four linear calls and `2Dh + 3h + 1` trainable parameters.

Initialization version `scaled_uniform_softplus_tie_v3` retains the L/M/R
uniform scales and zero B1. Effective tie weights start at the negative absolute
values of `U(-0.1*sqrt(3/h), 0.1*sqrt(3/h))`, floored in magnitude at FP32
epsilon, and the effective bias starts at `1e-4`. Stable inverse softplus
`x + log(-expm1(-x))` encodes these initial values in the raw parameters.

Frozen SR2, LoRA SR2, standalone/inline warmup, and the FP32 submission notebook
used v6. Checkpoints saved `raw_w_tie` (`[1, h]`) and `raw_b_tie` (scalar).
Architecture and exact tensor checks rejected v5 and earlier layouts.

### Archived SR2 v7 head

`swap_equivariant_abs_diff_v7` computes `d = abs((A-B)(L+M))`, the elementwise
absolute difference of the two win preactivations. It reads that nonnegative
vector with `w_tie = -softplus(raw_w_tie)` and adds
`b_tie = softplus(raw_b_tie)`. Both SR2 profiles, warmup, and `submit.ipynb` used
this head. Its embedding combination was `cross_projected_linear_abs_diff`;
initialization remained `scaled_uniform_softplus_tie_v3`, as in v6.

The v7 marker selected fresh head warmup keys and required rebuilding v6 and
earlier checkpoints. Its schemas remained adapter/head v2, frozen-head v1, and
warmup v5. Historical artifacts remain intact.

### Current SR2 v8 head (2026-11-14)

`swap_equivariant_abs_diff_v8` retains the v7 win branches, shared win dropout,
L/M/R initialization, and zero win projection bias. The tie now forms the signed
vector `d = (A-B)(L+M)`, reads it with an unconstrained trainable `u`, and only
then takes the scalar absolute value:

```text
v7: d = abs((A-B)(L+M)); z_tie = d @ (-softplus(raw_w_tie)) + softplus(raw_b_tie)
v8: d =     (A-B)(L+M);  z_tie = softplus(raw_b_tie) - abs(d @ u)
```

Components can cancel in the new signed readout. There is no tie GELU or
dropout. A-B and L+M are constructed in FP32 with autocast disabled; both tie
linear computations follow the existing head autocast policy. The scalar is
converted to FP32 before its absolute value and subtraction from the FP32
softplus bias. The submission notebook continues to run the entire head in FP32.

The new embedding combination is `cross_projected_signed_diff_abs_readout` and
the initialization version is `scaled_uniform_signed_tie_readout_v4`. Parameter
`u` replaces `raw_w_tie`, retains the `[1, h]` transposed readout shape, and keeps
its direct samples from `U(-0.1*sqrt(3/h), 0.1*sqrt(3/h))`. It has no softplus
mapping, absolute-value initialization, or magnitude floor. The effective tie
bias still starts at `1e-4`, encoded in scalar `raw_b_tie` with stable inverse
softplus. All parameters remain FP32, the count stays `2Dh + 3h + 1`, and output
order remains `[A-win, B-win, tie]`.

Frozen SR2, LoRA SR2, standalone/inline warmup, and `submit.ipynb` use v8.
Rebuild v7 and earlier checkpoints for the new head; no weights are converted.
Architecture checks reject old checkpoint metadata, while exact parameter and
optimizer-state checks also reject old layouts with relabeled metadata. The
architecture and initialization version select fresh warmup keys within the
existing namespace. Storage schemas, artifact paths, and old files are unchanged.
Token, adapter-free frozen-feature, and batch-plan caches retain their existing
identities and reuse rules. New reports use the v8 markers; historical reports
keep their original metadata. See the [current head definition](lora.md#swap-equivariant-head).

### Standalone SR2 warmup profile

`model warmup --build` now uses the SR2 profile and writes
`SR2_Qwen3_4B_lora_warmup/`. Current presets select
`Skywork/Skywork-Reward-V2-Qwen3-4B` in `warmup.model_name` and align standalone
model/head settings with their SR2 validation settings. Update custom standalone
settings when switching from the original Qwen backbone. Old
`qwen3_4b_lora_EXPT2_warmup/` entries stay in place and are not read or relabeled
as SR2 caches. Matching original training IDs, labels, and effective model/head
settings remain required; full-data builds use both partitions and therefore a different
warmup entry. See [warmup configuration](configuration.md#standalone-warmup).

## SR2 chat-template inputs

The active serializer is now `sr2-qwen3-chat-template-v1`, replacing
`aligned-prompt-newline-response-v1`. It renders alternating user/assistant
messages with the official SR2 tokenizer, including the final assistant's
default reasoning block, and encodes without additional special tokens.
System messages are not supplied. See [conversation serialization](data-preparation.md#conversation-serialization).

Use the current presets and run `uv run preprocess` to regenerate the grouped
holdout and token cache. The SR2 tokenizer is pinned for preprocessing; new
token counts may change partition membership. Keep a run's original processed
split when comparing or reporting historical validation predictions.

Existing token shards and batch plans coexist with newly keyed inputs/plans.
The changed serializer marker makes previous warmup entries miss and rejects
plain-text production checkpoints, even when their head architecture matches.
Rebuild those checkpoints with `uv run model SR2-Qwen3-4B-lora --build` and the
matching configuration when producing current checkpoints. Historical artifacts are not converted.
Storage schemas and model/head architecture are unchanged.

Python callers must supply a tokenizer to `ConversationSerializer` and pass the
result as `serializer=` when constructing `ModelDataRepository`. Resolve
the tokenizer revision before serializing, then encode using that same tokenizer
with `add_special_tokens=False`.

## Historical frozen model

This section describes the retired `qwen3-4b` profile, not a current CLI option.
It trained a pairwise MLP on pooled `[A, B, abs(A - B)]` features of width `3H`.
Branches used the aligned conversation serializer, right truncation to the old
`qwen3_4b.max_length`, and last-real-token pooling. Cached pooled vectors were
stored as FP16 and consumed by the head in FP32. The historical
`extraction_budget` controlled missing branch extraction and
`training_batch_size` controlled head training and inference.

The old `A/B_swap: aug` objective used per-row losses for original/swapped
training pairs; `inf` used loss on their averaged probabilities. Both averaged
probabilities after restoring A/B order at validation/submission. SR2 accepts
neither setting and uses original rows only.

Frozen head schema v3 identifies the `3H` head. Historical loading checked model
name, serializer, classifier width, class count, schema, and backbone hidden size
at inference. Earlier `2H` heads needed rebuilding. Context, precision,
attention, TF32, and A/B training mode were recorded but not enforced on load.
Frozen checkpoints did not pin the Hub revision or fingerprint local weights.
These artifacts remain at their original paths and require historical code to
execute; persistent branch-cache details remain in
[frozen embeddings](caching.md#frozen-embeddings).

## Renamed and removed options

| Old setting | Current setting / action |
| --- | --- |
| `model MODEL --test` / `--submit` | Removed; submission inference is no longer provided |
| `gpu.submit_devices` | Remove; model workflows use `gpu.device` |
| `DATA_PATH` | Set `data.raw_dir` and `data.processed_path` in YAML |
| `extraction_batch_size` | `extraction_budget`, measured in padded tokens; old branch counts are not equivalent values |
| LoRA `inference_batch_size` | `head_inference_batch_size`; keep the head batch count |
| LoRA `head_learning_rate` | Remove; `learning_rate` now controls both adapters and the classifier head during joint training |
| Frozen SR2 `min_lr_ratio` / `lr_decay_power` | Remove both; builds use `build_epochs` and absolute `build_min_lr`; validation uses ReduceLROnPlateau with `factor` and `patience` |
| Frozen SR2 / warmup `epochs` | Rename to `build_epochs`; LoRA joint `epochs` is unchanged |
| Frozen SR2 / warmup `lr_decay_base` | Replace with absolute `build_min_lr`, defaulting to that head stage's `learning_rate / 5`; the numeric base is not a learning rate |
| `warmup.warmup_min_lr_ratio` / `warmup.warmup_lr_decay_power` | Remove both; configure `warmup.build_epochs` and `warmup.build_min_lr` |
| `warmup.warmup_epochs`, `warmup.warmup_training_batch_size`, `warmup.warmup_learning_rate`, `warmup.warmup_lr_decay_base` | Use `warmup.build_epochs`, `warmup.training_batch_size`, `warmup.learning_rate`, and a chosen absolute `warmup.build_min_lr` |
| LoRA-local `warmup_epochs`, `warmup_training_batch_size`, `warmup_learning_rate` | Move to the required top-level `warmup` section as `build_epochs`, `training_batch_size`, and `learning_rate` |
| `warmup_ratio` | Choose explicit `warmup.build_epochs`, batch size, learning rate, and extraction budget |
| LoRA/warmup `A/B_swap` | Remove; active LoRA and standalone warmup always use original rows |
| `training_token_budget` or Boolean `gradient_checkpointing` | Use the two current checkpointing/packing controls below |

Joint training applies one cosine curve across all joint epochs, with
`min_lr_ratio` defaulting to `0.125`. Set `min_lr_ratio: 1` to keep its shared
learning rate fixed. Its `epochs` field is unchanged. Remove `head_learning_rate`
before using current commands; it fails strict unknown-key validation.

Frozen SR2 and `warmup` now accept `build_epochs` and absolute `build_min_lr`.
The old `epochs` and `lr_decay_base` names are rejected in those two sections,
including when supplied alongside the new names. Omitted `build_min_lr` defaults
to the corresponding head initial rate divided by five and must satisfy
`0 < build_min_lr <= learning_rate`. Current presets explicitly set that
one-fifth value and retain their previous epoch counts. Remove older head
minimum-ratio and polynomial-power fields, including any `e` string values.

For `N = build_epochs > 1`, the scheduler derives
`decay_rate = (ln(initial_lr) - ln(build_min_lr)) / (N - 1)` and uses
`lr = initial_lr * exp(-decay_rate * epoch_index)` at the start of each
zero-based epoch. The first and final epochs explicitly use the initial and
minimum rates, with no change within an epoch. One epoch uses the initial rate;
`warmup.build_epochs: 0` disables inline warmup without creating a scheduler.
To preserve an old base-controlled curve for `N > 1`, choose
`build_min_lr = learning_rate / old_base ** (N - 1)` when that value is positive
and representable. Set the minimum equal to the initial rate for fixed-rate
training. Configuration objects use `warmup_build_epochs` and
`warmup_build_min_lr` internally; public snapshots and frozen-head checkpoint
training metadata use the unprefixed `build_epochs` and `build_min_lr`.

Frozen SR2 validation uses ReduceLROnPlateau with `factor: 0.95`, `patience: 2`,
and `early_stopping: 7` as parser defaults. Threshold and minimum rate are fixed
at zero. The scheduler consumes the unrounded unsmoothed validation loss once
after each epoch; a reduction affects the next epoch and does not reset the
early-stopping counter. Build fields do not control validation, which permits
omitted or null `build_epochs`.

Decaying head caches now include `exponential_endpoint_epoch_v2` and
`warmup_build_min_lr`, with the target count still stored under the persistent
`warmup_epochs` key. Only an exact target is reusable: changing the count or
minimum trains a complete new run on a miss. Earlier `exponential_epoch_v1`
base-controlled caches remain on disk without migration or reuse, even when a
new curve is mathematically equivalent. Fixed-rate training omits the schedule
and minimum metadata fields, preserving legacy fixed-rate cache identities and
lower-epoch continuation. Cache schema v5, token/feature/plan identities, and
production checkpoint compatibility remain unchanged. Existing caches and
historical configuration snapshots are preserved. See
[learning-rate decay](lora.md#learning-rate-decay).

The historical `gradient_checkpointing_budget` used to name what is now
`gradient_checkpointing_threshold`. The current **budget** is a separate second
packing pass. When migrating that historical format, move its old value to the
threshold and explicitly choose the new budget (`0` disables the second pass).
Do not mechanically rename an already-current configuration. See
[packing controls](configuration.md#joint-training-packing).

Internal `warmup_extraction_budget`, `warmup_label_smoothing`, and `ab_swap` are
not supported YAML keys. Configure their documented public fields instead.

## Cache and checkpoint migration

| Artifact | Current behavior |
| --- | --- |
| Caches formerly under `data/processed/` | Move `token_cache/`, `embedding_cache/`, `adapter_batches/`, and relevant `*_warmup/` directories into `CACHE_PATH`, preserving manifests and all parts |
| Single-file caches / processed Parquet | Still readable; adopt the multipart policy on a later write if large enough |
| Old dataset-wide frozen `.pt` / `.json` embeddings | Not reused by current per-branch SQLite caching |
| Legacy per-model tokens or batch-plan directories | Not read by current shared token/plan caches |
| LoRA batch-plan schemas before v4 | Rebuilt when needed |
| Warmup schemas before v5 or incompatible historical model/head identities | Not reused by active SR2 |
| Historical frozen head checkpoints, including schemas before v3 | Retained for historical code; incompatible with the new frozen SR2 head and not converted |
| Adapter/head schema v1 or another head architecture | Rebuild for the current schema-v2 SR2 head |
| Plain-text SR2 warmup / adapter-head checkpoint | New serializer keys for warmup; retrain production checkpoint for chat-template input |
| Historical quantized adapter/head checkpoints | Incompatible with active non-quantized LoRA profiles |

Model downloads use a native Hub cache under optional `MODEL_PATH`; they are
separate from `CACHE_PATH` and the training-cache multipart policy. See
[caching](caching.md) for current identities and recovery rules.

## Preserve historical results

Keep each historical `validation.csv`, `metrics.json`, and `config.yaml` together.
Do not rewrite saved section names, parameter values, resolved paths, or
aggregation markers to resemble current presets. Historical holdout runs remain
reportable where their existing aggregation policy is supported. Supply a
separate current-format config containing `data`, `validation`, and `gpu`,
pointing at the run's original processed split and matching validation fraction:

```shell
uv run result RUN_NAME --config path/to/report.yaml
```

Do not pass a snapshot with retired model sections directly to the current
loader. Executing the retired model itself requires matching historical code;
reporting its supported holdout outputs does not. See [reports](reports.md).
