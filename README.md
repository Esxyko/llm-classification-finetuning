# LLM Classification Finetuning

Compare two responses to a shared prompt and predict **A wins, B wins, or tie**
for the [Kaggle competition](https://www.kaggle.com/competitions/llm-classification-finetuning).

The project provides a saved prompt-grouped holdout, frozen and LoRA Skywork
classifiers, validation reports, and full-data checkpoint builds. Model workflows
use **one CUDA GPU**.

## Start here

1. [Install the project](docs/getting-started.md) with Python 3.14+ and `uv`.
2. Use the saved processed split, or [prepare it from raw data](docs/data-preparation.md).
3. [Validate a model and inspect its report](docs/models.md).
4. [Build a full-data checkpoint and export the trained backbone](docs/models.md#full-data-builds).

For Colab and the local VS Code notebook workflow, see
[Colab setup and persistence](docs/colab.md).

## Model profiles

Selectors and YAML section names are case-sensitive.

| CLI selector | YAML section / checkpoint directory | Backbone and classifier |
| --- | --- | --- |
| `SR2-Qwen3-4B` | `SR2_Qwen3_4B` | Frozen Skywork reward-trained Qwen3-4B; the same SR2 head; validation/build only |
| `SR2-Qwen3-4B-lora` | `SR2_Qwen3_4B_lora` | Skywork reward-trained Qwen3-4B LoRA; swap-equivariant three-class head |

Both profiles consume the saved train/validation split and train and predict on
original rows only. `ALL` runs configured frozen SR2 before LoRA.
The frozen model has independent training settings and shares
compatible head/feature caches with LoRA warmup and the separate `warmup --build`
selector. Retired Qwen profiles and
their historical artifacts are described in [migration](docs/migration.md).

## Documentation

| Guide | Use it to |
| --- | --- |
| [Getting started](docs/getting-started.md) | Install dependencies, choose saved or raw data, and run a first validation |
| [Colab](docs/colab.md) | Set remote paths, persist caches, and understand result copying and shutdown |
| [Commands](docs/commands.md) | Select modes, checkpoints, cache refresh, and batch validation |
| [Configuration](docs/configuration.md) | Look up YAML fields, defaults, paths, GPU settings, and token budgets |
| [Data preparation](docs/data-preparation.md) | Understand grouping, balancing, augmentation, and the processed schema |
| [Model workflows](docs/models.md) | Validate/build frozen or LoRA SR2, and inspect build outputs |
| [LoRA training](docs/lora.md) | Understand the head, warmup, batching, checkpointing, and memory controls |
| [Caching and Python APIs](docs/caching.md) | Check reuse conditions, multipart storage, and CPU preparation APIs |
| [Reports](docs/reports.md) | Interpret predictions, configuration snapshots, timings, and GPU peaks |
| [Troubleshooting](docs/troubleshooting.md) | Diagnose setup, input, cache, GPU, and report failures |
| [Migration](docs/migration.md) | Update retired selectors, old configuration keys, and five-fold artifacts |
| [Codebase reference](structure.md) | Locate module ownership and implementation invariants |

The executable configuration is [config.yaml](config.yaml); files in
[configs/](configs/) are independent complete presets. See
[documentation responsibilities](structure.md#documentation-boundaries) before
adding implementation or usage details.
