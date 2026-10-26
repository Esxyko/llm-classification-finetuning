# Troubleshooting

[Documentation index](../README.md)

Start with the first specific error. A notebook's final `CalledProcessError`
reports a failed child process, not its cause. Batch validation streams the
underlying output and saves it under `results/logs/validate-*.log`.

## Environment and downloads

| Symptom | Check / action |
| --- | --- |
| Project requires a newer Python than the notebook kernel | Install with `uv sync --locked` and launch project work through `uv run`; the project requires Python 3.14+ |
| Config file missing or paths unexpectedly point elsewhere | Run from the intended project root or supply `--config`; relative data/cache/output paths follow the working directory, not the YAML parent |
| `.env` edit has no effect | Existing environment variables take precedence; check the actual process, especially a notebook's `os.environ` and its subprocesses |
| Kaggle download fails | Accept competition rules, verify `KAGGLE_API_TOKEN`, and check connectivity; the saved processed split does not require a raw download |
| Config/model/tokenizer download fails | Check endpoint, network, and repository access; `MODEL_PATH` only selects cache storage |

For `CAS Client Error` / `401 Unauthorized`, the failing request may be Xet
rather than the repository itself. With the default mirror, package initialization
sets `HF_HUB_DISABLE_XET=1` unless explicitly configured. Set it before HF imports
if HTTP downloads are intended, and restart a notebook kernel that already
imported those libraries. A Hub authorization failure can still require valid
`HF_TOKEN` and repository access. See [download defaults](getting-started.md#hugging-face-downloads).

## Configuration and data

| Symptom | Check / action |
| --- | --- |
| Missing/unknown YAML field, even in an unused model section | Every command validates the entire file; no values are inherited from the root preset. Use the [field reference](configuration.md#required-fields-and-defaults) |
| `train.csv` missing during preprocessing | Run `uv run data` first, or `uv run init` for both steps |
| Processed `.parquet` file appears absent | Check for its `.manifest.json` and all referenced `.parts/` files; keep the configured path ending in `.parquet` |
| Processed multipart file is corrupt | Restore the complete matching artifact or rerun preprocessing; training does not silently reconstruct it from raw CSVs |
| Token balance cannot be reached | Verify tokenizer/revision and grouped data; group exchanges must preserve class/row counts. A failed preparation does not replace the previous split |
| LoRA input exceeds `max_length` | Raise the limit using the reported ID/branch/count; lowering batch size or token budgets does not shorten that input |
| Legacy fold or retired-model error | Follow [migration](migration.md); old formats are rejected rather than aliased |

## Cache reuse and checkpoints

| Symptom | Check / action |
| --- | --- |
| Warmup repeats after standalone warmup | Compare effective model/head settings and original training IDs, not just shared optimization fields; see [standalone warmup](configuration.md#standalone-warmup) |
| Warmup state survives edited training content | Warmup keys exclude text/labels when IDs are unchanged; use `--refresh-cache` to retrain |
| Cache lock timeout | Another process may hold the artifact session; finish that operation before retrying |
| Checkpoint not found after validation | Validation does not save a production checkpoint; run `--build`, using the intended tag |
| Checkpoint incompatible | Match architecture/model/input identity or rebuild. LoRA also checks head dropout, adapter settings, and quantization; see [checkpoint rules](lora.md#checkpoints) |

Changing GPU placement or compute settings can be allowed by checkpoint
compatibility without reproducing the same predictions. SR2 warmup cache and
production-checkpoint compatibility are independent; see [cache identities](caching.md).

## CUDA, attention, and memory

For unsupported CUDA/precision, select an available `gpu.device`, or two distinct
`gpu.submit_devices` for submission. Both submission devices must support the
selected precision. There is no automatic precision or single-GPU fallback.

For T4 submission `No available kernel`, check the
[submission pins](getting-started.md#submission-environment-pytorch-211--cuda-128),
`fp16`, and `sdpa`. The inference process applies its K/V-head expansion workaround;
a notebook-only patch does not affect an independent child process. Math fallback
remains disabled, so unsupported fused inputs still fail.

Match an OOM to its stage before adjusting settings:

| Failing stage | Setting to reduce |
| --- | --- |
| LoRA warmup extraction | `warmup.extraction_budget` |
| LoRA warmup head | `warmup.warmup_training_batch_size` |
| LoRA joint training, positive threshold | `gradient_checkpointing_threshold` and/or `gradient_checkpointing_budget` |
| LoRA fixed initial batches, threshold `0` | `training_batch_size`; adjust accumulation to retain target update size |
| LoRA joint-validation/submission extraction | Selected model section's `extraction_budget` |
| LoRA head inference | Selected model section's `head_inference_batch_size` |

With positive token packing, changing batch size and accumulation while keeping
their product fixed does not lower packing limits. A zero second-pass budget
disables singleton merging; a zero threshold enables checkpointing on every
joint batch. Warmup/inference do not use checkpointing. Oversized individual
inputs still run alone above a budget and may not fit. See [memory tuning](lora.md#memory-tuning).

LoRA never truncates. Lowering `max_length` rejects inputs above the new limit;
reduce the stage's batch or token budget when addressing memory use.

## Reports and Colab

| Symptom | Check / action |
| --- | --- |
| Latest report selects an unexpected directory | Select the run by name; discovery uses directory modification time, then validates that directory |
| CSV rows / aggregation do not match | Use the run's processed split and metrics; SR2 originals only, historical frozen both orientations, historical EXPT2 originals only, in the required order |
| Report config has retired sections or paths from another machine | Supply a separate current-format config pointing to the run's original processed split; retain the original snapshot |
| Drive results destination remains empty during training | The local notebook copies results only after its batch command returns or raises; caches configured on Drive write during their own stages |
| Drive path or shortcut is inaccessible | Verify the actual mounted path and destination permissions in the runtime |
| Runtime checkpoint disappears after cleanup | The local notebook copies `results/`, not `models/`; preserve production checkpoints separately before release |

See [report inputs](reports.md#report-inputs) and
[Colab persistence and cleanup](colab.md) for the exact behavior.
