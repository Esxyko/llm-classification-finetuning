# Migration from older workflows

[Documentation index](../README.md)

Use the current [configuration reference](configuration.md) for new runs.
These notes explain historical names and artifacts; they do not change saved
experiments or make old checkpoints compatible with a new architecture.

## Five folds to one holdout

Replace `cross_validation` with `validation`, remove `n_splits`, set the desired
`validation_fraction` and `max_token_deviation`, and point `data.processed_path`
to `data/processed/train_validation.parquet`. Run `uv run preprocess` from raw
training data, or use the saved current manifest and volumes from Git.

Models share the saved grouped holdout. Old fold datasets/configurations are
rejected rather than interpreted as a holdout. Report synthesis rejects old
five-fold result directories; retain them for the corresponding historical code.

## Retired profiles

The only active model selector is `SR2-Qwen3-4B-lora`; `warmup` is a separate
build-only utility. On 2026-10-27, frozen `qwen3-4b` and
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

The current marker is `swap_equivariant_harmonic_discrepancy_v2`. Batched A/B
projections and the shared win readout are retained, while the relative tie
correction changes from `s - 2q + b` to `s - q + b`; win logits contain no `q`.
Parameter names/shapes and storage schemas are unchanged, but predictions for
the same weights change. V2 selects new warmup keys and rejects v1 production
checkpoints through the architecture marker. Rebuild for v2; historical v1
artifacts are preserved without automatic weight conversion.

`model warmup --build` now uses the SR2 profile and writes
`SR2_Qwen3_4B_lora_warmup/`. Current presets select
`Skywork/Skywork-Reward-V2-Qwen3-4B` in `warmup.model_name` and align standalone
model/head settings with their SR2 validation settings. Update custom standalone
settings when switching from the original Qwen backbone. Old
`qwen3_4b_lora_EXPT2_warmup/` entries stay in place and are not read or relabeled
as SR2 caches. Matching original training IDs and effective model/head settings
remain required; full-data builds use both partitions and therefore a different
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
matching configuration before submission. Historical artifacts are not converted.
Storage schemas and model/head architecture are unchanged.

Python callers must supply a tokenizer to `ConversationSerializer` and pass the
result as `serializer=` when constructing either model-data repository. Resolve
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
| `model MODEL --test` | `model MODEL --submit` |
| `DATA_PATH` | Set `data.raw_dir` and `data.processed_path` in YAML |
| `extraction_batch_size` | `extraction_budget`, measured in padded tokens; old branch counts are not equivalent values |
| LoRA `inference_batch_size` | `head_inference_batch_size`; keep the head batch count |
| LoRA `head_learning_rate` | Remove; `learning_rate` now controls both adapters and the classifier head during joint training |
| LoRA-local `warmup_epochs`, `warmup_training_batch_size`, `warmup_learning_rate` | Move to the required top-level `warmup` section |
| `warmup_ratio` | Choose explicit `warmup.warmup_epochs`, batch size, learning rate, and extraction budget |
| LoRA/warmup `A/B_swap` | Remove; active LoRA and standalone warmup always use original rows |
| `training_token_budget` or Boolean `gradient_checkpointing` | Use the two current checkpointing/packing controls below |

Joint training now applies one cosine curve across all joint epochs, with
`min_lr_ratio` defaulting to `0.125` even when omitted from an older configuration.
Set `min_lr_ratio: 1` to keep the shared learning rate fixed. Configurations
containing `head_learning_rate` fail strict unknown-key validation; remove it
before using current commands. Head warmup remains at its fixed rate, and
existing warmup cache identities and production checkpoint compatibility are
unchanged. Historical configuration snapshots are preserved. See
[cosine decay](lora.md#cosine-learning-rate-decay).

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
| Historical frozen head checkpoints, including schemas before v3 | Retained for historical code; no active frozen profile or conversion to SR2 |
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
