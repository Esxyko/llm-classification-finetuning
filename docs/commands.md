# Command reference

[Documentation index](../README.md)

Run from the project root after [setup](getting-started.md). All entry points
accept `--config PATH`, defaulting to `config.yaml` in the working directory,
except the batch scripts, which select configs through their constants.
Use `uv run COMMAND --help` to inspect the installed parser. In the separate
[submission environment](getting-started.md#submission-environment-pytorch-211--cuda-128),
run `model` directly instead of `uv run model`.

## Entry points

| Command | Purpose | Output with default paths |
| --- | --- | --- |
| `uv run init` | Download missing raw files and rebuild processed data | `data/processed/train_validation.parquet` |
| `uv run data` | Download missing Kaggle files only | `data/raw/*.csv` |
| `uv run preprocess` | Rebuild the split from local `train.csv` | `data/processed/train_validation.parquet` |
| `uv run model MODEL` | Validate on the saved holdout | `results/MODEL-<timestamp>/` |
| `uv run model MODEL --build` | Train on IDs from both partitions; save adapter/head and merged backbone | `models/SR2_Qwen3_4B_lora/adapter.pt` and `merged/` |
| `uv run model MODEL --submit` | Load a checkpoint and predict test probabilities | `results/submit/MODEL-<timestamp>/submission.csv` |
| `uv run model warmup --build` | Train/cache an SR2 head on the train partition only | `<CACHE_PATH>/SR2_Qwen3_4B_lora_warmup/<key>.pt` |
| `uv run result [RUN]` | Report a selected or latest validation run | `results/comprehensive/` |
| `uv run python scripts/validate.py` | Validate explicit model/config pairs sequentially | Validation directories and `results/logs/validate-*.log` |
| `uv run python scripts/build.py` | Build explicit model/config pairs sequentially, including standalone warmup | Adapter/head and merged backbone, or warmup caches; `results/logs/build-*.log` |

The processed `.parquet` path is logical; large files use a manifest and volumes.
See [multipart storage](caching.md#shared-storage-and-multipart-files).

`RESULT_PATH` selects the root for validation runs, submissions, reports, and
batch logs; the table shows its default `./results`. See
[path rules](configuration.md#configuration-files-and-paths).

## Model selection and execution modes

`MODEL` is `SR2-Qwen3-4B-lora` or uppercase `ALL`; the separate `warmup`
selector is build-only. Selector spelling differs from the
[YAML section names](configuration.md#configuration-sections).

| Option | Behavior |
| --- | --- |
| No mode flag | Holdout validation; publishes the final epoch's predictions, without saving a production checkpoint |
| `--build` | Train on all original IDs from both saved partitions and save adapter/head plus the merged backbone |
| `--submit` | Load the selected checkpoint; requires test files and two configured GPUs |
| `--refresh-cache` | Rebuild requested token entries and the selected mode's model-specific caches |
| `--checkpoint-tag TAG` | Choose `adapter_TAG.pt` and, for build, `merged_TAG/`; build/submit only |

`--build` and `--submit` are mutually exclusive. A tag starts with an ASCII
letter or digit and contains only ASCII letters, digits, `_`, or `-`.
Without a tag, builds replace `adapter.pt` and `merged/`. Use the same tag and
compatible configuration when submitting. Submission continues to use the
adapter/head checkpoint with the original base weights. See
[build outputs](models.md#build-and-submit).

`ALL` runs SR2 when its section is present in the YAML. It accepts the same
mode, refresh, and tag options, continues
after model failures, and exits with status `1` if any fail. An empty selection
is an error. The entire YAML is validated before any model runs.

## Standalone warmup

`warmup` is a separate selector requiring `--build`. It accepts `--config` and
`--refresh-cache`; the default validation command mode, `--submit`, and
`--checkpoint-tag` are unsupported.
It is excluded from `ALL`.

It prepares train and validation tokens, then trains the SR2 comparison head
only on original train-partition rows using the non-quantized backbone configured
in `warmup` (Skywork in the shipped presets). After each epoch, it evaluates the
entire training set with the final epoch weights and warmup label smoothing,
then evaluates validation with unsmoothed log loss. Both evaluations disable
dropout and reuse the frozen features loaded or extracted before training.
The epoch logs and final summary report both losses to six decimal places.

A complete compatible head hit restores the head, loads the backbone, and loads
cached train/validation features, extracting only missing branches, to recompute
both losses without training or rewriting the head-warmup cache; it reports zero
executed epochs. There are no adapters, joint
training, production checkpoints, or result reports. It shares SR2's cache
namespace; a validation run can reuse it when effective settings and training
IDs match. If only a compatible lower-epoch cache exists, the command selects
the largest completed epoch, trains the remaining epochs, and writes a separate
target cache. The summary displays `Resumed` and counts only newly executed epochs.
Normal full-data model builds train on both partitions.
See [warmup configuration](configuration.md#standalone-warmup) and
[cache reuse](caching.md#standalone-warmup-build).

## Cache refresh

`--refresh-cache` affects only entries requested by the selected execution mode.
For LoRA validation/build with warmup enabled, it forces a new warmup and
refreshes requested tokens, warmup feature entries, and batch plans. Standalone
warmup also refreshes its requested tokens, features, and head state. Submission
uses neither warmup states nor joint-training plans. Refresh is not a checkpoint rebuild or a
request to delete all existing caches; see [cache identities](caching.md).

## Batch validation

[scripts/validate.py](../scripts/validate.py) runs holdout validation only. Each
task invokes `uv run model MODEL --config PATH` from the project root. It does
not support the standalone `warmup` selector; model-internal head warmup still
follows the selected YAML.

Validation logs use `<RESULT_PATH>/logs/validate-*.log`.

## Batch builds

[scripts/build.py](../scripts/build.py) runs full-data builds and standalone
warmup. Each task invokes `uv run model MODEL --config PATH --build` from the
project root. A `warmup` task creates or reuses the train-partition head-warmup
cache; a registered model builds its full-data adapter/head and merged backbone.
For example, this selection runs standalone warmup before a full-data build:

```python
MODEL_CONFIGS: dict[str, list[str]] = {
    "warmup": ["1.yaml"],
    "SR2-Qwen3-4B-lora": ["3.yaml"],
}
```

Standalone warmup and full-data builds use different training ID sets; see
[cache reuse](caching.md#standalone-warmup-build). Build logs use
`<RESULT_PATH>/logs/build-*.log`.

The batch script does not add `--checkpoint-tag`. Multiple configs for the same
model replace the default `adapter.pt` and `merged/` outputs in sequence. Use the
model CLI directly with tags when separate checkpoints are needed.

### Shared batch behavior

Each script selects work independently through its top-level `CONFIG_DIR` and
`MODEL_CONFIGS` constants. Both currently select `SR2-Qwen3-4B-lora` with
`3.yaml` from `configs/`. Edit these constants to choose other combinations;
there are no script CLI selection flags. Dictionary keys are registered model
selectors, with `warmup` additionally supported by the build script. Each list
contains filenames relative to `CONFIG_DIR`, resolved from the script's project
root.

Only listed files run; newly added YAML files are not selected automatically.
Models run in dictionary order and filenames in list order. Empty lists skip
a model; no selected filenames is an error. Both scripts continue after missing
configs or failed runs, print a task summary, and exit with status `1` if any
task fails.

Each invocation streams script messages and combined child stdout/stderr to
the terminal and one immediately flushed UTF-8 log under `<RESULT_PATH>/logs/`
(default `./results/logs/`). Each script loads `.env` from its project root and
resolves relative result paths there, matching its child commands.
Unexpected exceptions and keyboard interruptions also write their traceback
before propagating. A final `CalledProcessError` in a notebook is a wrapper;
read the earlier model error or this log for the cause.

## Report selection

`RUN` is a direct child directory name under `results/`, not a path to a CSV.
Without it, discovery selects the most recently modified eligible directory,
then validates it; it does not search backward for a valid run. Reports always
replace `results/comprehensive/`, so copy a report if you want to keep several.
See [report input requirements](reports.md#report-inputs).

For removed selectors and `--test`, see [migration](migration.md).
