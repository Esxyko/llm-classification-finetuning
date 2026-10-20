# Generate validation reports

[Documentation index](../README.md)

Run [cross-validation](models.md) first; builds and submissions do not produce
validation-fold reports. The selected [configuration](configuration.md) supplies
the processed reference file and fold count. Result-folder selection does not
filter by model or the configuration recorded in `metrics.json`.

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

The selected folder must contain exactly one CSV per configured fold. Each CSV
must have these columns in order:

```text
id,winner_model_a,winner_model_b,winner_tie
```

Rows must follow the corresponding fold's exact ID order in the configured
processed Parquet, including both original and swapped rows. Filenames do not
need to include the fold number because the command infers it from that
sequence. Only top-level `.csv` files are counted. IDs must be finite integers;
probabilities must be finite, lie in `[0, 1]`, and sum to one within an absolute
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
| `records.xlsx` | One row per source ID with fold, expected and actual labels, incorrect count, and log loss |

The workbook supports filtering; its `AVG` row recalculates over visible rows.
Aggregation is selected by `metrics.json`, rather than the model's name:

- `prediction_aggregation: "ab_swap_average"`: Each ID contributes one prediction,
  one confusion-matrix count, and the log loss of its averaged distribution.
  Original and A/B-restored swapped probabilities must already match within
  `1e-6`; the report command validates them and does not average distinct votes.
  Both active models publish this marker.
- Missing or null `prediction_aggregation` (including a missing `metrics.json`):
  Each ID contributes two orientation votes to the confusion matrix.
  `actual_labels` contains both labels, `incorrects` ranges from 0 to 2, and
  `contributed_loss` is the mean of the two orientation log losses.

An unknown aggregation marker is rejected. Keep the run's `metrics.json` with
its CSVs so report synthesis uses the intended policy.

## Timing metrics

For both frozen and LoRA models (including head warmup), `metrics.json` records
`training_time`, `validation_time`, and `total_time` as `HH:MM:SS` strings rounded
to whole seconds for every epoch and fold. Hours can exceed 24.
Training and validation times measure each phase's elapsed wall time after
waiting for GPU work to finish. Fold totals also include initialization,
prediction alignment, and LoRA fold cleanup; epoch totals include epoch setup.
LoRA's first warmup training time includes canonical feature extraction for
both training and validation IDs. Full-data builds do not write `metrics.json`.
Top-level `training_time` and `validation_time` sum that model's corresponding
phases across all folds. Top-level `total_time` measures the model's complete
run, including pipeline initialization, input loading, embedding preparation,
all folds, cleanup, and result publication. The final `config_total_time` field
records the same duration for this model under the selected config. With
`model ALL`, each model is timed separately and its metrics contain only its
own elapsed time. The command also prints each model's duration at completion.
