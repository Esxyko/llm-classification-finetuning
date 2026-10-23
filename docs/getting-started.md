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

   The logical output is `data/processed/train_validation.parquet`. Large outputs
   use a manifest and volumes; see [processed data](data-preparation.md#processed-data).

For `uv run model warmup --build` with the saved default split, use the versioned
Parquet manifest and all referenced volumes included in the checkout. The loader
reconstructs the split automatically, so raw competition data is not required:

```shell
uv run model warmup --build
```

Install the locked dependencies first. The command downloads the configured
Hugging Face model and tokenizer as needed; these are not stored in this
repository. Raw CSVs and token-statistics reports are not warmup inputs.

## Hugging Face downloads

When `.env` or the environment specifies a non-empty `MODEL_PATH`, each downloaded
model and its tokenizer use the native Hub cache directly in that directory,
for example `<MODEL_PATH>/models--Qwen--Qwen3-4B/`. Downloaded files persist across
processes without packing or unpacking. Otherwise the standard Hugging Face
cache settings apply. Existing native downloads can be copied or moved with the
[model migration script](caching.md#hugging-face-model-downloads).

Project commands and token-statistics downloads default to
[`https://huggingface.co`](https://huggingface.co) for Hugging Face model,
configuration, and tokenizer files. Package initialization sets `HF_ENDPOINT`
before importing Hugging Face libraries. An existing `HF_ENDPOINT` environment
variable takes precedence; set it in `.env` or before starting the command to
use another endpoint. This default does not require a `.env` entry.

When the endpoint is `https://hf-mirror.com`, initialization also defaults
`HF_HUB_DISABLE_XET` to `1` so model weights use HTTP downloads instead of the
Xet client, avoiding its CAS authorization failures. An existing
`HF_HUB_DISABLE_XET` value takes precedence; set it to `0` before starting the
command to opt back into Xet. Other endpoints keep their existing Xet behavior.
See the [Hugging Face environment variable reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisablexet).

For example, to select an endpoint in PowerShell:

```powershell
$env:HF_ENDPOINT = "https://huggingface.co"
uv run model warmup --build
```

When using the Python API, import `llm_classification_finetuning` before
`transformers` or `huggingface_hub` so the download defaults take effect. Restart
an existing notebook kernel or Python process if those libraries were already
imported.

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
