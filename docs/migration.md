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

The active selectors are frozen `qwen3-4b`, `qwen3-4b-lora-EXPT2`, and
`SR2-Qwen3-4B-lora`. Retired selectors include `qwen3-1.7b`,
`qwen3-1.7b-mono-input`, `qwen3-4b-qlora`, `qwen3-4b-lora`, and
`qwen3-4b-lora-EXPT`. They are not CLI choices or members of `ALL`.

The old `qwen3_4b_lora`, `qwen3_4b_lora_EXPT`, and `qwen3_4b_qlora` sections are
unknown keys. Start from a current preset when moving to EXPT2 or SR2; renaming
a section alone does not migrate its objective, warmup fields, or saved head.

Local archives, when present, live under the Git-ignored
`src/llm_classification_finetuning/models/archive/`. They are not included in a
fresh clone and are never imported by active profiles. Shared legacy head and
quantization branches remain inside the EXPT2 implementation, but there is no
active QLoRA registration.

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
| Warmup schemas before v5 or retired-profile heads | Not reused by active EXPT2/SR2 |
| Frozen head schemas before v3 (`2H` inputs) | Rebuild for the current `3H` absolute-difference head; branch embeddings can remain reusable |
| Adapter/head schema v1 or another head architecture | Rebuild for the current schema-v2 EXPT2/SR2 head |
| Historical quantized adapter/head checkpoints | Incompatible with active non-quantized LoRA profiles |

Model downloads use a native Hub cache under optional `MODEL_PATH`; they are
separate from `CACHE_PATH` and the training-cache multipart policy. See
[caching](caching.md) for current identities and recovery rules.

## Preserve historical results

Keep each historical `validation.csv`, `metrics.json`, and `config.yaml` together.
Do not rewrite saved section names, parameter values, resolved paths, or
aggregation markers to resemble current presets. Historical holdout runs remain
reportable where their existing aggregation policy is supported; retired-model
snapshots require matching historical code to execute.
