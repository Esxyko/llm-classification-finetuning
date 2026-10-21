# Getting started

[Project overview and documentation index](../README.md)

## Requirements

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A Kaggle account with the competition rules accepted
- One CUDA GPU for training and two CUDA GPUs for submission inference (data
  preparation and report generation do not require a GPU)
- QLoRA training requires the installed bitsandbytes CUDA backend to support
  the selected GPU and CUDA runtime; see [QLoRA training](qlora.md).

## Quick start

Run these commands from the project root.

1. Install the locked dependencies:

   ```shell
   uv sync
   ```

2. Create `.env` from the provided template:

   ```powershell
   Copy-Item .env.example .env
   ```

   On macOS or Linux, run `cp .env.example .env` instead. Generate a token from
   [Kaggle API settings](https://www.kaggle.com/settings/api), then set it in
   `.env`:

   ```dotenv
   KAGGLE_API_TOKEN=your_token_here
   ```

   `.env` is ignored by Git. An existing environment variable takes precedence.

3. Download the competition files and prepare the training data:

   ```shell
   uv run init
   ```

   The output is `data/processed/train_validation.parquet`.

## Submission environment: PyTorch 2.11 / CUDA 12.8

For submission inference with **PyTorch 2.11.0+cu128**, **CUDA 12.8**, and
**Transformers 5.16.1**, install the dedicated requirements from the project root
in the submission Python environment (Python 3.14 or newer):

```shell
python -m pip install -r requirements-submit.txt
model qwen3-4b-lora --submit --config path/to/submit.yaml
```

[`requirements-submit.txt`](../requirements-submit.txt) pins both libraries and
selects the CUDA 12.8 PyTorch build. The normal `uv sync` workflow retains the
training versions in `uv.lock`. Run the installed `model` command directly in the
submission environment; `uv run` would synchronize the training lock again.

For two T4 GPUs, use `gpu.precision: fp16`,
`gpu.attention_implementation: sdpa`, and
`gpu.submit_devices: [cuda:0, cuda:1]` in the selected YAML. Start with the selected
model section's `extraction_batch_size: 1` and preserve its `max_length`.
`head_inference_batch_size` for LoRA/QLoRA controls only the head over extracted
CPU features. All present model sections still need their usual configuration
fields; see [configuration](configuration.md).

All three model profiles apply submission SDPA compatibility in the process that
executes `--submit`, before loading backbone weights. Transformers expands K/V
heads with `repeat_kv`, memory-efficient attention is enabled, and math fallback
is disabled. Flash and cuDNN attention remain eligible where supported. The GQA
predicate and SDPA backend flags are restored after success or failure. Explicit
`eager` attention keeps its existing behavior. No notebook-only patch or separate
execution of `scripts/patch.py` is needed, including for submission subprocesses.

The dependency pins and SDPA workaround still require a submission run on the
target Kaggle GPUs to verify runtime behavior and memory use.

## Next steps

- [Prepare training data](data-preparation.md) for the split and augmentation format.
- [Run models](models.md) for holdout validation, builds, and submissions.
- [Command reference](commands.md) for all entry points and execution flags.
- [Configuration](configuration.md) for paths, GPU settings, and batch experiments.
- [Troubleshooting](troubleshooting.md) for setup and runtime failures.
