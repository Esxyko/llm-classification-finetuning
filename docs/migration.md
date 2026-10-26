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

The SR2 selector, config key, result prefix, checkpoint/cache directories,
head tensors, architecture marker `swap_equivariant_tie_from_wins_v1`,
adapter/head schema v2, batch-plan schema v4, and warmup schema v5 are unchanged.
The EXPT2 head architecture remains in SR2 even though the EXPT2 profile is retired.
Compatible existing SR2 checkpoints and caches require no conversion.

`model warmup --build` now uses the SR2 profile and writes
`SR2_Qwen3_4B_lora_warmup/`. Current presets select
`Skywork/Skywork-Reward-V2-Qwen3-4B` in `warmup.model_name` and align standalone
model/head settings with their SR2 validation settings. Update custom standalone
settings when switching from the original Qwen backbone. Old
`qwen3_4b_lora_EXPT2_warmup/` entries stay in place and are not read or relabeled
as SR2 caches. Matching original training IDs and effective model/head settings
remain required; full-data builds use both partitions and therefore a different
warmup entry. See [warmup configuration](configuration.md#standalone-warmup).

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
| LoRA-local `warmup_epochs`, `warmup_training_batch_size`, `warmup_learning_rate` | Move to the required top-level `warmup` section |
| `warmup_ratio` | Choose explicit `warmup.warmup_epochs`, batch size, learning rate, and extraction budget |
| LoRA/warmup `A/B_swap` | Remove; active LoRA and standalone warmup always use original rows |
| `training_token_budget` or Boolean `gradient_checkpointing` | Use the two current checkpointing/packing controls below |

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
