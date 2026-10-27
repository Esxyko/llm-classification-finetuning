# Command reference

[Documentation index](../README.md)

Run from the project root after [setup](getting-started.md). All entry points
accept `--config PATH`, defaulting to `config.yaml` in the working directory.
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
| `uv run model MODEL --build` | Train on IDs from both partitions and save a production checkpoint | `models/SR2_Qwen3_4B_lora/adapter.pt` |
| `uv run model MODEL --submit` | Load a checkpoint and predict test probabilities | `results/submit/MODEL-<timestamp>/submission.csv` |
| `uv run model warmup --build` | Train/cache an SR2 head on the train partition only | `<CACHE_PATH>/SR2_Qwen3_4B_lora_warmup/<key>.pt` |
| `uv run result [RUN]` | Report a selected or latest validation run | `results/comprehensive/` |
| `uv run python scripts/validate.py` | Execute explicit model/config pairs sequentially, including standalone warmup | Validation directories, warmup caches, and `results/logs/validate-*.log` |

The processed `.parquet` path is logical; large files use a manifest and volumes.
See [multipart storage](caching.md#shared-storage-and-multipart-files).

## Model selection and execution modes

`MODEL` is `SR2-Qwen3-4B-lora` or uppercase `ALL`; the separate `warmup`
selector is build-only. Selector spelling differs from the
[YAML section names](configuration.md#configuration-sections).

| Option | Behavior |
| --- | --- |
| No mode flag | Holdout validation; publishes the final epoch's predictions, without saving a production checkpoint |
| `--build` | Train on all original IDs from both saved partitions |
| `--submit` | Load the selected checkpoint; requires test files and two configured GPUs |
| `--refresh-cache` | Rebuild requested token entries and the selected mode's model-specific caches |
| `--checkpoint-tag TAG` | Choose `adapter_TAG.pt`; build/submit only |

`--build` and `--submit` are mutually exclusive. A tag starts with an ASCII
letter or digit and contains only ASCII letters, digits, `_`, or `-`.
Without a tag, builds replace `adapter.pt`. Use the same tag and
compatible configuration when submitting.

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
dropout and reuse the temporary frozen features extracted before training.
The epoch logs and final summary report both losses to six decimal places.

A compatible hit restores the head, loads the backbone, and extracts current
train/validation features to recompute both losses without training or rewriting
the warmup cache; it reports zero executed epochs. There are no adapters, joint
training, production checkpoints, or result reports. It shares SR2's cache
namespace; a validation run can reuse it when effective settings and training
IDs match. Normal full-data model builds train on both partitions.
See [warmup configuration](configuration.md#standalone-warmup) and
[cache reuse](caching.md#standalone-warmup-build).

## Cache refresh

`--refresh-cache` affects only entries requested by the selected execution mode.
For LoRA validation/build with warmup enabled, it forces a new warmup and
refreshes requested tokens and batch plans. Submission uses neither warmup
states nor joint-training plans. Refresh is not a checkpoint rebuild or a
request to delete all existing caches; see [cache identities](caching.md).

## Batch validation

[scripts/validate.py](../scripts/validate.py) selects work through its top-level
`CONFIG_DIR` and `MODEL_CONFIGS` constants. The current selection is
`SR2-Qwen3-4B-lora` with `1.yaml` from `configs/`. Edit these constants to
choose other combinations; there are no script CLI selection flags.

Each dictionary key is a registered model selector or `warmup`. Its list
contains filenames relative to `CONFIG_DIR`, which is resolved from the script's
project root.
Only listed files run; newly added YAML files are not selected automatically.
Models run in dictionary order and filenames in list order. Empty lists skip
a model; no selected filenames is an error.

The script always invokes children from the project root, using
`uv run model MODEL --config PATH`, adding `--build` for `warmup` to create or
reuse the train-partition head-warmup cache. For example, this selection runs
warmup before holdout validation:

```python
MODEL_CONFIGS: dict[str, list[str]] = {
    "warmup": ["1.yaml"],
    "SR2-Qwen3-4B-lora": ["1.yaml"],
}
```

It continues after missing configs or failed runs, prints a task summary, and
exits with status `1` if any task fails.

Each invocation streams script messages and combined child stdout/stderr to
the terminal and one immediately flushed UTF-8 log under `results/logs/`.
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
