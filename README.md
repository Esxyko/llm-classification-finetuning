# LLM Classification Finetuning

Compare two responses to a shared prompt and predict **A wins, B wins, or tie**
for the [Kaggle competition](https://www.kaggle.com/competitions/llm-classification-finetuning).

The project provides a saved prompt-grouped holdout, frozen-backbone and LoRA
classifiers, validation reports, full-data checkpoint builds, and submission
inference. Training uses **one CUDA GPU**; submission uses **two CUDA GPUs**.

## Start here

1. [Install the project](docs/getting-started.md) with Python 3.14+ and `uv`.
2. Use the saved processed split, or [prepare it from raw data](docs/data-preparation.md).
3. [Validate a model and inspect its report](docs/models.md).
4. Build a full-data checkpoint, then generate a submission in the
   [submission environment](docs/getting-started.md#submission-environment-pytorch-211--cuda-128).

For Colab and the local VS Code notebook workflow, see
[Colab setup and persistence](docs/colab.md).

## Model profiles

Selectors and YAML section names are case-sensitive.

| CLI selector | YAML section / checkpoint directory | Backbone and classifier |
| --- | --- | --- |
| `qwen3-4b` | `qwen3_4b` | Frozen Qwen3-4B; trainable MLP over `[A, B, abs(A - B)]` |
| `qwen3-4b-lora-EXPT2` | `qwen3_4b_lora_EXPT2` | Qwen3-4B LoRA; swap-equivariant three-class head |
| `SR2-Qwen3-4B-lora` | `SR2_Qwen3_4B_lora` | Skywork reward-trained Qwen3-4B LoRA; the same EXPT2 head |

All profiles consume the same saved train/validation split. The frozen profile
supports A/B augmentation or an averaged-probability objective; both LoRA
profiles train and predict on original rows only. The separate `warmup --build`
selector prepares an optional EXPT2 head-warmup cache, not a production model.

## Documentation

| Guide | Use it to |
| --- | --- |
| [Getting started](docs/getting-started.md) | Install dependencies, choose saved or raw data, and run a first validation |
| [Colab](docs/colab.md) | Set remote paths, persist caches, and understand result copying and shutdown |
| [Commands](docs/commands.md) | Select modes, checkpoints, cache refresh, and batch validation |
| [Configuration](docs/configuration.md) | Look up YAML fields, defaults, paths, GPU settings, and token budgets |
| [Data preparation](docs/data-preparation.md) | Understand grouping, balancing, augmentation, and the processed schema |
| [Model workflows](docs/models.md) | Validate, build, submit, and understand frozen-model behavior |
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
