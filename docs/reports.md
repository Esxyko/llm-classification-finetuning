# Generate validation reports

[Documentation index](../README.md)

Run [holdout validation](models.md) first; builds and submissions do not produce
validation reports. The selected configuration supplies the processed reference.
Each active model writes the final epoch's holdout probabilities to
`validation.csv` alongside `metrics.json` and `config.yaml` in one timestamped run
directory. The three files are staged and published together after training
succeeds. `config.yaml` records the effective `data`, `validation`, `gpu`, and
current model sections, including default values and resolved data paths. LoRA
snapshots also include the required `warmup` section, with the three warmup
optimization fields and internal `warmup_extraction_budget` omitted from
the current LoRA model section. Each section retains its own `extraction_budget`.
EXPT2 and SR2 snapshots omit `A/B_swap` from both the model and warmup sections
because both use original rows and a single prediction per ID.
The current LoRA model configuration section is `qwen3_4b_lora_EXPT2` or
`SR2_Qwen3_4B_lora`. SR2 follows the EXPT2 reporting rules in this guide,
including the same head and embedding markers, original-only row counts,
prediction aggregation, timing, and GPU memory metrics.
Other model sections are omitted; frozen-model snapshots also omit `warmup`.
The snapshot uses the same YAML field names as the input configuration and can
be passed to `--config`.
`metrics.json` does not contain a `config` field.
Metrics use `validation_strategy: "holdout"`, `validation_loss`, a flat `epochs`
list, canonical `source_training_rows` and `validation_rows`, and the actual
A/B-selected `training_rows` for the frozen profile, or original-only
`training_rows` for EXPT2.
No fold count or cross-fold average is recorded.

New runs record `embedding_combination` in `metrics.json`:
`"concat_a_b_abs_diff"` identifies `[A, B, |A - B|]` with an elementwise absolute
difference for the frozen profile.
EXPT2 records `"sum_signed_diff_abs_diff"` for its sum, signed
difference, and absolute difference features. New runs also record `head_architecture`:
`"pairwise_mlp"` for the frozen classifier,
and `"swap_equivariant_tie_from_wins_v1"` for EXPT2's head deriving ties from
the activated win branches.
This combination marker is kept in metrics only; the
prediction CSVs, configuration snapshot, terminal summary, and generated reports
retain their existing formats. Historical standard LoRA and EXPT results retain
their original markers and remain readable by report synthesis.

Create reports for the latest validation run:

```shell
uv run result
```

Or select a direct subfolder of `results/`:

```shell
uv run result qwen3-4b-YYYYMMDD-HHMMSS
```

Without an argument, the command selects the most recently modified result
subfolder, excluding `comprehensive`, `submit`, and historical `test` directories.
It also excludes `.comprehensive-*` staging/backup directories. The selected
folder is then validated; discovery does not search for the latest valid run.

Report generation requires exactly one top-level CSV named `validation.csv`
and a `metrics.json` identifying a holdout run. Legacy five-fold runs are
retained but rejected with a migration message. CSV columns must be:

```text
id,winner_model_a,winner_model_b,winner_tie
```

Rows follow the exact ID order of the processed Parquet's `split=validation`
rows, including original and swapped rows for the frozen model.
EXPT2 writes only the original validation rows, preserving their order,
with one row per ID.
The processed artifact is still fully validated before selecting those rows.
Training rows do not enter reports.
IDs must be finite integers; probabilities must be finite, lie in `[0, 1]`, and
sum to one within an absolute
tolerance of `1e-6`.

## Publication and outputs

Both report files are generated in a staging directory before replacing
`results/comprehensive/`. A staging or promotion failure leaves the previous
report in place or restores it. If restoration fails, the command reports the
path of the retained backup. If backup cleanup fails after promotion, the new
report remains available and the command reports the backup path. Directory
replacement uses two renames, so concurrent readers can briefly see the
directory as unavailable.

| File | Contents |
| --- | --- |
| `confusion_matrix.png` | 3 × 3 counts and expected-class row percentages, after restoring swapped predictions to the original A/B orientation |
| `records.xlsx` | One row per source ID with split, expected and actual labels, incorrect count, and log loss |

The workbook supports filtering; its `AVG` row recalculates over visible rows.
Aggregation is selected by `metrics.json`, rather than the model's name:

- `prediction_aggregation: "ab_swap_average"`: Each ID contributes one prediction,
  one confusion-matrix count, and the log loss of its averaged distribution.
  Original and A/B-restored swapped probabilities must already match within
  `1e-6`; the report command validates them and does not average distinct votes.
  The frozen model publishes this marker; historical standard LoRA also uses it.
- `prediction_aggregation: "original_only"`: EXPT2 supplies one original
  prediction per ID. Each ID contributes one actual label, an `incorrects`
  value of 0 or 1, one log loss, and one confusion-matrix count. Swapped
  predictions are neither required nor generated.
- Missing or null `prediction_aggregation` in a valid holdout `metrics.json`:
  Each ID contributes two orientation votes to the confusion matrix.
  `actual_labels` contains both labels, `incorrects` ranges from 0 to 2, and
  `contributed_loss` is the mean of the two orientation log losses.

An unknown aggregation marker is rejected. Keep the run's `metrics.json` with
its CSVs so report synthesis uses the intended policy.

## Timing metrics

For frozen and EXPT2 LoRA validation runs (including their head-warmup epochs),
`metrics.json` records `training_time`, `validation_time`, and `total_time`
as `HH:MM:SS` strings rounded
to whole seconds for every executed epoch and run. Hours can exceed 24.
Training and validation times measure each phase's elapsed wall time after
waiting for GPU work to finish. Run totals also include initialization,
prediction alignment, and LoRA cleanup; epoch totals include epoch setup.
LoRA's first warmup training time includes canonical feature extraction for
both training and validation IDs. Each LoRA joint-validation time includes fresh
backbone extraction followed by head inference; warmup validation reuses the
existing features. Reused warmup adds no old losses, epoch entries, or timings
to the new run; joint epoch numbers retain their configured warmup offset.
Cache loading and saving are excluded from epoch phase timings and included
in complete-run runtime. Full-data builds do not write `metrics.json`; their
epoch counts include only epochs executed in the current run.
Top-level `training_time` and `validation_time` sum the corresponding durations
of executed epochs. Final `total_time` measures the complete model run, including
initialization, input loading, embedding preparation, cleanup, and publication.
`config_total_time` records the same duration for this model and selected config.
With `model ALL`, each model is timed separately.
The command also prints each model's duration at completion.

## GPU memory peaks

Holdout validation automatically monitors the configured `gpu.device` for each
model's complete run. Monitoring begins before pipeline initialization and covers
model loading, embedding preparation, all executed epochs (including LoRA head
warmup), training, validation, cleanup, and result publication. All active models,
`model ALL`, and `scripts/validate.py` use the same monitoring path.
Builds and submissions do not enable this monitoring.

After GPU work finishes, the command prints the model selector, device, and
allocated/reserved peaks in GiB (`bytes / 1024**3`) with two decimal places. The
same peaks are atomically recorded with the complete runtime in `metrics.json`
as integer byte counts:

```json
"gpu_memory": {
  "device": "cuda:0",
  "peak_allocated_bytes": 123456789,
  "peak_reserved_bytes": 234567890
}
```

`peak_allocated_bytes` measures memory occupied by PyTorch tensors;
`peak_reserved_bytes` measures memory managed by PyTorch's caching allocator,
including unused cached blocks. These are separate maxima and need not occur
at the same time. They do not measure total GPU usage, other processes, or
allocations outside PyTorch's allocator. See the
[PyTorch memory management reference](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management).

Peaks are reset once per model run, never between epochs or phases.
Monitoring does not clear cached memory or subtract the starting occupancy:
memory still resident when a run begins contributes to its absolute peak. This
also applies to sequential models in `model ALL`; a complete embedding-cache
hit reports the actual work performed, including head training.

If execution fails after monitoring starts, the command attempts to print the
peaks observed before failure without waiting for GPU synchronization. Failure
to read or print those peaks does not replace the original exception. Failed
execution does not record final GPU peaks or complete runtime metrics.
