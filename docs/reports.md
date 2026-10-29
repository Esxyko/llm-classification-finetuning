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
child of `RESULT_PATH` (default `./results`), excluding `comprehensive`, `submit`,
`logs`, historical `test`, and `.comprehensive-*` staging/backup directories. It
then validates that folder; it does not search backward for the latest valid run.

For a specific run, replace `RUN_NAME` with its directory name:

```shell
uv run result RUN_NAME --config path/to/report.yaml
```

Use a current-format `report.yaml` with the run's original processed Parquet
and matching validation fraction. It may contain only the required `data`,
`validation`, and `gpu` sections. Historical snapshots containing retired model
sections are rejected by the current loader; do not pass them directly or rewrite
them. Current SR2 snapshots can still be used when their paths resolve correctly.
Each invocation replaces `<RESULT_PATH>/comprehensive/`, so copy
reports elsewhere if you want to retain several comparisons.

## Report inputs

The selected directory must contain exactly one top-level CSV, named
`validation.csv`, and a `metrics.json` identifying a holdout. Legacy five-fold
runs are rejected. Required CSV columns, in order, are:

`id,winner_model_a,winner_model_b,winner_tie`

| Profile / aggregation | Required validation rows |
| --- | --- |
| SR2 / `original_only` | Only original validation rows, in processed-data order, one per ID |
| Historical frozen / `ab_swap_average` | All saved `split=validation` rows, original and swapped, in processed-data order |
| Historical EXPT2 / `original_only` | Only original validation rows, in processed-data order, one per ID |

The processed artifact is fully validated before selecting rows. Training rows
do not enter reports. IDs must be finite integers; probabilities must be finite,
within `[0, 1]`, and sum to one within absolute tolerance `1e-6`.

## Publication and outputs

| File under `<RESULT_PATH>/comprehensive/` | Contents |
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
| SR2 | `data`, `validation`, `gpu`, `warmup`, `SR2_Qwen3_4B_lora` |

LoRA snapshots keep shared warmup controls under `warmup`, omit internal resolved
fields, and omit `A/B_swap` from both model and warmup sections. Each section
retains its own extraction budget and smoothing value. Other model sections are
excluded; `metrics.json` has no embedded `config` field or source-config path.

A current SR2 snapshot can be passed to `--config`, but reproducing a run also needs matching
data/code/dependencies and model identity. It does not capture `.env` or persist
`MODEL_PATH`/`CACHE_PATH`/`RESULT_PATH`. Retain historical snapshots unchanged;
old model sections require historical code to execute; reporting uses a separate
current configuration as described above. See [migration](migration.md).

## Model metrics

Current metrics use `validation_strategy: holdout`, `validation_loss`, and a flat
`epochs` list. `source_training_rows` and `validation_rows` count canonical IDs;
`training_rows` counts the rows selected by the profile's A/B policy. There are
no fold counters or cross-fold means. Training can use label smoothing, but
validation/reporting log loss is unsmoothed.

| Profile | `head_architecture` | `embedding_combination` |
| --- | --- | --- |
| SR2 | `swap_equivariant_harmonic_discrepancy_v3` | `cross_projected_linear_diff_harmonic_abs_diff` |
| Historical SR2 with harmonic/discrepancy features from wins | `swap_equivariant_harmonic_discrepancy_v2` | `cross_projected_harmonic_abs_diff` |
| Historical SR2 with doubled relative correction | `swap_equivariant_harmonic_discrepancy_v1` | `cross_projected_harmonic_abs_diff` |
| Historical SR2 with readout biases | `swap_equivariant_tie_from_wins_v2` | `sum_signed_diff_abs_diff` |
| Historical frozen | `pairwise_mlp` | `concat_a_b_abs_diff` |
| Historical EXPT2 | `swap_equivariant_tie_from_wins_v1` | `sum_signed_diff_abs_diff` |

These markers identify model behavior; historical results preserve their
original values. Combination markers are metrics, not additional YAML options.

## Timing metrics

Times are elapsed wall-clock durations formatted as `HH:MM:SS`, rounded to whole
seconds, with unbounded hours. GPU phase timing waits for queued GPU work.

| Field / scope | Includes |
| --- | --- |
| Epoch `training_time` | That epoch's training phase, including full-training-set inference at the end of warmup epochs |
| Epoch `validation_time` | Validation phase; joint LoRA validation includes fresh extraction plus head inference |
| Epoch `total_time` | Epoch setup and training/validation; excludes warmup feature extraction performed before epochs |
| Top-level `training_time`, `validation_time` | Sums of executed epoch phase durations |
| Final `total_time`, `config_total_time` | Complete successful model run: initialization, input/cache/embedding preparation, training, validation, cleanup, and publication |

Warmup train and validation inference reuse their shared extracted features.
Warmup epoch `training_loss` is the final-weight training-set loss with warmup
label smoothing and dropout disabled; joint epoch `training_loss` remains the
row-weighted average observed during optimization. All `validation_loss` values
are unsmoothed.

Each joint epoch also reports `last_step_training_loss` in its terminal log and,
for validation runs, its `metrics.json` epoch entry. This is the sum of training
losses across all micro-batches in the last consumed optimizer group divided by
that group's actual row count, including any merged tail. It reuses the losses
computed before the parameter update, with the same training objective, label
smoothing, and dropout as `training_loss`; no extra training-set evaluation is
performed. A consumed group still counts when the FP16 scaler skips its update.
Logs show six decimal places; JSON retains the full floating-point value. Warmup
epochs omit this field. Build runs print it in joint epoch logs, while their final
`final_training_loss` retains its existing final-epoch average meaning.

Cache work and warmup extraction are included in complete-run time, not epoch
phases. A warmup hit contributes no
historical epoch losses or times; joint epoch numbering retains the configured
warmup offset. `model ALL` times each model separately. Builds do not write
`metrics.json`; their summaries count only epochs executed in that invocation.
Standalone warmup reports current train and validation losses on cache hits as
well. Complete hits execute zero training epochs; partial hits count only the
remaining epochs trained in the current invocation.

The [progress counter](models.md#training-progress) excludes extraction and
publication, so its time estimate has a different scope.

## GPU memory peaks

Holdout validation, full-data builds, and standalone warmup monitor PyTorch
allocator peaks on `gpu.device`. Monitoring starts before GPU work and model
loading and continues through saving, export, cleanup, and result publication.
Full-data builds first complete CPU input/batch preparation and both output-path
write preflights, then start monitoring. Each model invocation has its own
statistics, including under `ALL` and batch scripts. Submission does not enable
monitoring.

Training and validation have independent, cross-epoch maxima for the work
executed in the current invocation:

| Scope | Includes |
| --- | --- |
| `training` | Warmup and joint optimization, plus warmup training-set evaluation |
| `validation` | Holdout feature extraction and head inference |
| `global` | All monitored work, including model loading, preliminary warmup feature extraction, cache saves, build export, cleanup, and publication |

Before each phase, the monitor saves the preceding interval's global maxima and
resets allocator peak counters. At phase completion it merges the observed
allocated/reserved maxima into that phase and the global statistics, then resets
the counters for work outside the phase. Global maxima therefore survive every
reset, even when loading or export exceeds all training and validation peaks.
These resets do not change allocator caches or live tensors.

At the end of the run, the terminal prints `training`, `validation`, and finally
`global` peaks once each in GiB, identifying the model and device. Unexecuted
phases are omitted; full-data builds have no validation entry. A complete
standalone warmup cache hit still records current training-set evaluation as
`training` and current holdout evaluation as `validation`, despite executing zero
optimization epochs. Partial hits count only work executed during this run.

Successful holdout runs atomically append integer byte counts with complete-run
timing. Existing top-level `gpu_memory` fields retain their global meaning;
`stages` adds the independent phase peaks, for example:

```json
{
  "gpu_memory": {
    "device": "cuda:0",
    "peak_allocated_bytes": 123456789,
    "peak_reserved_bytes": 234567890,
    "stages": {
      "training": {
        "device": "cuda:0",
        "peak_allocated_bytes": 120000000,
        "peak_reserved_bytes": 234567890
      },
      "validation": {
        "device": "cuda:0",
        "peak_allocated_bytes": 100000000,
        "peak_reserved_bytes": 234567890
      }
    }
  }
}
```

Unexecuted phases are also omitted from `stages`. Builds and standalone warmup
report only in the terminal and do not create a `metrics.json` file.

Allocated bytes measure PyTorch tensor memory; reserved bytes measure its caching
allocator, including unused blocks. They are separate maxima and need not occur
together. They exclude other processes and allocations outside that allocator.
Monitoring neither clears caches nor subtracts initial occupancy; resident memory
at the beginning contributes to each interval's peak, including across sequential
runs. In particular, validation's reserved peak can include blocks cached during
training even though the historical training peak counter has been reset.

On failure after monitoring starts, the command attempts an unsynchronized
terminal summary of the observed phase and global peaks, labeled `before failure`,
without masking the execution error. If allocator reads fail, previously recorded
peaks remain available. Failures before GPU monitoring starts do not trigger CUDA
initialization for reporting. The command does not append final GPU/runtime
metrics for the failed execution. Inspect batch [validation logs](commands.md#batch-validation)
or [build logs](commands.md#batch-builds) for the underlying error.
