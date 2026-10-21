# LLM Classification Finetuning

Prepare leakage-safe train/validation splits, run frozen, LoRA, or QLoRA Qwen pairwise models, and
generate reports or submissions for Kaggle's
[LLM Classification Finetuning competition](https://www.kaggle.com/competitions/llm-classification-finetuning).

## Project overview

The project compares two model responses to a shared prompt and predicts an A
win, a B win, or a tie. It supports:

- One stratified 80/20 split grouped by canonical prompt, with A/B-swapped training rows.
- A frozen Qwen3-4B backbone with a trainable pairwise classifier.
- Qwen3-4B LoRA adapters trained together with the classifier.
- Qwen3-4B QLoRA with a 4-bit NF4 training base and a non-quantized submission base.
- Holdout validation, full-data checkpoints, competition submissions, and
  confusion-matrix and spreadsheet reports.

The workflow is data preparation → holdout validation and reports → full-data
build → submission. Training uses one CUDA GPU; submission inference uses two.
Start with the [installation and quick-start guide](docs/getting-started.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [Getting started](docs/getting-started.md) | Requirements, dependencies, Kaggle credentials, and initial setup |
| [Command reference](docs/commands.md) | Entry points, model selection, execution modes, and checkpoint flags |
| [Data preparation](docs/data-preparation.md) | Downloads, leakage-safe train/validation splits, augmentation, and processed columns |
| [Model workflows](docs/models.md) | Frozen-model training, A/B objectives, builds, and submissions |
| [LoRA training](docs/lora.md) | Input limits, measured lengths, warmup, batching, memory, and checkpoints |
| [QLoRA training](docs/qlora.md) | NF4 quantization, shared LoRA behavior, checkpoints, and non-quantized submissions |
| [Caching and Python APIs](docs/caching.md) | Shared token encodings, frozen embeddings, batch plans, and API examples |
| [Validation reports](docs/reports.md) | Validation CSV requirements, report outputs, publication, and timing metrics |
| [Configuration](docs/configuration.md) | YAML paths and sections, GPU settings, and batch experiments |
| [Troubleshooting](docs/troubleshooting.md) | Setup, configuration, GPU, memory, and input-length errors |
| [Codebase reference](structure.md) | Runtime flow, module ownership, artifact layout, and development invariants |

## Documentation responsibilities

This README owns the project overview and navigation. `docs/` owns setup,
usage, configuration, and behavioral references.
[`structure.md`](structure.md) owns code organization, module responsibilities,
and invariants to preserve when changing the implementation.
