# Validation reports and metrics

[Documentation index](../README.md)

A successful holdout validation publishes one timestamped directory containing
`validation.csv`, `metrics.json`, and `config.yaml`. Builds and submissions do
not produce validation reports. Files are staged and exposed together after
training succeeds; predictions come from the final epoch.

## Generate a report

```shell
uv run result
```

Without a run name, discovery selects the most recently modified eligible direct
child of `results/`, excluding `comprehensive`, `submit`, `logs`, historical
`test`, and `.comprehensive-*` staging/backup directories. It then validates that
folder; it does not search backward for the latest valid run.

For a specific run, replace `RUN_NAME` with its directory name:

```shell
uv run result RUN_NAME --config results/RUN_NAME/config.yaml
```

Use the corresponding processed Parquet, not a newly generated split. Saved
snapshots can contain absolute paths from another machine; adjust a copy's data
paths if necessary. Each invocation replaces `results/comprehensive/`, so copy
reports elsewhere if you want to retain several comparisons.

## Report inputs

The selected directory must contain exactly one top-level CSV, named
`validation.csv`, and a `metrics.json` identifying a holdout. Legacy five-fold
runs are rejected. Required CSV columns, in order, are:

`id,winner_model_a,winner_model_b,winner_tie`

| Profile / aggregation | Required validation rows |
| --- | --- |
| Frozen / `ab_swap_average` | All saved `split=validation` rows, original and swapped, in processed-data order |
| EXPT2 or SR2 / `original_only` | Only original validation rows, in processed-data order, one per ID |

The processed artifact is fully validated before selecting rows. Training rows
do not enter reports. IDs must be finite integers; probabilities must be finite,
within `[0, 1]`, and sum to one within absolute tolerance `1e-6`.

## Publication and outputs

| File under `results/comprehensive/` | Contents |
| --- | --- |
| `confusion_matrix.png` | 3-by-3 counts and expected-class row percentages, with swapped predictions restored to original A/B order |
| `records.xlsx` | One source-ID record with split, expected/actual labels, incorrect count, and log loss |

The workbook supports filtering, and its `AVG` row recalculates over visible
rows. Reports are staged before directory replacement. On a recoverable promotion
failure, publication restores the previous report; if recovery or backup cleanup
fails, it reports the retained backup path. The two directory renames can briefly
make the report path unavailable to concurrent readers.

### Aggregation policy

`metrics.json` selects aggregation; the model directory's name does not.

| `prediction_aggregation` | Votes and loss per source ID |
| --- | --- |
| `ab_swap_average` | One prediction, one confusion count, and loss of the averaged distribution; original and restored-swapped probabilities must already match within `1e-6` |
| `original_only` | One original prediction, one confusion count, `incorrects` of 0 or 1, and one log loss |
| Missing or `null` in a valid historical holdout | Two orientation votes; `actual_labels` includes both; `incorrects` is 0–2; loss is the mean of the two orientation losses |

Report synthesis validates already-averaged predictions; it does not average
inconsistent new votes. Unknown markers are rejected. Keep metrics and CSVs
together so historical runs retain their intended policy.

## Configuration snapshots

The adjacent `config.yaml` uses public input names, includes loaded defaults and
resolved data paths, and contains only shared sections plus the current model:

| Run | Included sections |
| --- | --- |
| Frozen | `data`, `validation`, `gpu`, `qwen3_4b` |
| EXPT2 | `data`, `validation`, `gpu`, `warmup`, `qwen3_4b_lora_EXPT2` |
| SR2 | `data`, `validation`, `gpu`, `warmup`, `SR2_Qwen3_4B_lora` |

LoRA snapshots keep shared warmup controls under `warmup`, omit internal resolved
fields, and omit `A/B_swap` from both model and warmup sections. Each section
retains its own extraction budget and smoothing value. Other model sections are
excluded; `metrics.json` has no embedded `config` field or source-config path.

A snapshot can be passed to `--config`, but reproducing a run also needs matching
data/code/dependencies and model identity. It does not capture `.env` or persist
`MODEL_PATH`/`CACHE_PATH`. Retain historical snapshots unchanged; old model
sections may require historical code. See [migration](migration.md).

## Model metrics

Current metrics use `validation_strategy: holdout`, `validation_loss`, and a flat
`epochs` list. `source_training_rows` and `validation_rows` count canonical IDs;
`training_rows` counts the rows selected by the profile's A/B policy. There are
no fold counters or cross-fold means. Training can use label smoothing, but
validation/reporting log loss is unsmoothed.

| Profile | `head_architecture` | `embedding_combination` |
| --- | --- | --- |
| Frozen | `pairwise_mlp` | `concat_a_b_abs_diff` |
| EXPT2 / SR2 | `swap_equivariant_tie_from_wins_v1` | `sum_signed_diff_abs_diff` |

These markers identify current model behavior; historical results preserve their
original values. Combination markers are metrics, not additional YAML options.

## Timing metrics

Times are elapsed wall-clock durations formatted as `HH:MM:SS`, rounded to whole
seconds, with unbounded hours. GPU phase timing waits for queued GPU work.

| Field / scope | Includes |
| --- | --- |
| Epoch `training_time` | That epoch's training phase |
| Epoch `validation_time` | Validation phase; joint LoRA validation includes fresh extraction plus head inference |
| Epoch `total_time` | Epoch setup and training/validation; excludes warmup feature extraction performed before epochs |
| Top-level `training_time`, `validation_time` | Sums of executed epoch phase durations |
| Final `total_time`, `config_total_time` | Complete successful model run: initialization, input/cache/embedding preparation, training, validation, cleanup, and publication |

Warmup validation reuses its extracted features. Cache work and warmup extraction
are included in complete-run time, not epoch phases. A warmup hit contributes no
historical epoch losses or times; joint epoch numbering retains the configured
warmup offset. `model ALL` times each model separately. Builds do not write
`metrics.json`; their summaries count only epochs executed in that invocation.

The [progress counter](models.md#training-progress) excludes extraction and
publication, so its time estimate has a different scope.

## GPU memory peaks

Holdout validation resets PyTorch allocator peaks once per model on `gpu.device`
before pipeline initialization. The interval covers loading, preparation,
warmup, training, validation, cleanup, and result publication. Epochs share that
interval. `ALL` and batch validation use this same path; builds/submissions do
not enable it.

The terminal prints allocated/reserved peaks in GiB. Successful runs atomically
append integer byte counts with complete-run timing, for example:

```json
{
  "gpu_memory": {
    "device": "cuda:0",
    "peak_allocated_bytes": 123456789,
    "peak_reserved_bytes": 234567890
  }
}
```

Allocated bytes measure PyTorch tensor memory; reserved bytes measure its caching
allocator, including unused blocks. They are separate maxima and need not occur
together. They exclude other processes and allocations outside that allocator.
Monitoring neither clears caches nor subtracts initial occupancy; resident memory
at the beginning contributes to the peak, including across sequential runs.

On failure, the command attempts an unsynchronized terminal peak summary without
masking the execution error. It does not append final GPU/runtime metrics for
the failed execution. Inspect batch [logs](commands.md#batch-validation) for the
underlying error.
