# Kaggle submission notebook

[Documentation index](../README.md)

[`submit.ipynb`](../submit.ipynb) scores the competition `test.csv` with the model written by
`model SR2-Qwen3-4B-lora --build` and writes `submission.csv`. It is a **code-competition
notebook**: it is self-contained, runs offline, and does not import this package. Its
runtime target is GPU T4 x2 with Python **3.13 or newer**;
installed inference dependencies must also support the kernel's Python version.
The full training project's Python 3.14 requirement does not apply here.
Ruff targets Python 3.13 for `submit.ipynb` through a per-file override in
`pyproject.toml`, preserving this compatibility when formatting the notebook.

Inference dependency references are embedded from the current `uv.lock`, including
PyTorch **2.14.0+cu130** and Transformers **5.17.0**. Compatible versions may differ from
the lock: the preflight enforces the project's inference minimums, installed packages'
base runtime constraints, and the SDPA/Qwen3 interfaces used by the notebook. Dependency
installation runs in the upload workflow before notebook execution; the inference
cells check and use that prepared environment offline.

## Prepare the model dataset

Upload a Kaggle Dataset containing the two build outputs (`models/SR2_Qwen3_4B_lora/`):

```text
SR2_Qwen3_4B_lora/
├── adapter.pt
└── merged/
    ├── config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja
    ├── export_metadata.json
    ├── model.safetensors.index.json
    └── model-00001-of-00002.safetensors, model-00002-of-00002.safetensors
```

Only `adapter.pt`'s `head_state_dict` is used; the LoRA weights are already merged into
`merged/`. Files downloaded from Drive as split archives (`*-001.zip`, shards named
`model-0000X-of-00002-00Y.safetensors`) must be unzipped first. The notebook tolerates the
`-00Y` shard suffix by symlinking to the names in the index, but it does not unzip parts.

Set `MODEL_DIR` in the configuration block **below the startup diagnostics in the first
code cell**. If that path does not exist, `/kaggle/input` is searched for an `adapter.pt`
beside a `merged/config.json`. The notebook rejects a merged
backbone whose `export_metadata.json` checksum does not match `adapter.pt`, a head that is
not `swap_equivariant_harmonic_discrepancy_v3`, and mismatched hidden sizes or base revisions.

## Run

The upload workflow runs these dependency setup commands before the notebook code:

```sh
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install transformers==5.17.0 pandas==3.0.5 safetensors==0.8.0 tqdm==4.70.1 packaging==26.3
```

These select the CUDA 13.0 PyTorch/TorchVision builds and the specified inference
package versions. The setup commands run outside the inference cells. For offline
execution, the upload workflow must make their wheels and transitive dependencies
available; the notebook itself uses the resulting installed packages.

1. Use a Python 3.13+ kernel, configure the upload-time dependency setup above, add
   the competition data and model dataset, select GPU T4 x2, and turn Internet off
   for inference.
2. Run all cells after dependency setup (first on the 3-row public `test.csv`). Review
   the startup report and preflight results; resolve any dependency or CUDA errors
   before submitting.
3. Check the final log line: `prior-filled rows` should be `0`, then submit.

## Startup diagnostics and dependency checks

After upload-time dependency setup, the first code cell sets import-time environment
defaults and prints system information using the standard library before importing
inference packages: OS, kernel, architecture,
Python version/interpreter, CPU model and logical core count, RAM, working directory, and
disk space. It then runs `nvidia-smi` and, when available, `nvcc --version`, each with a
10-second timeout. Missing tools or failed optional diagnostics are reported without
preventing the remaining information from being printed.

Before the first PyTorch/Transformers import, the first code cell removes unused
`torchaudio` with the current kernel's `sys.executable -m pip uninstall -y torchaudio`.
This prevents a preinstalled CUDA 12.8 TorchAudio extension from being loaded alongside
the CUDA 13.0 PyTorch build: the upload commands replace PyTorch and TorchVision but
do not replace TorchAudio. Uninstallation runs offline, skips an absent package, and
stops with an explicit error if removal fails. If PyTorch, Transformers, or TorchAudio
was already imported in this kernel, restart the kernel and run all cells from the
beginning so the cleanup precedes those imports.

The installed and locked versions of `torch`, `transformers`, `numpy`, `pandas`,
`safetensors`, `tokenizers`, `huggingface-hub`, `tqdm`, and `packaging` are shown together.
The next code cell checks dependencies and imports before any model is loaded:

- Python 3.13+ is required, matching the standalone notebook's Ruff target.
- Project minimums apply to the inference dependencies declared in `pyproject.toml`:
  `torch>=2.11.0`, `transformers>=5.16.1`, and `pandas>=3.0.5`.
- Installed package metadata supplies Python constraints and transitive base runtime
  requirements. Environment markers are evaluated for the current kernel; optional
  extras such as training, vision, and testing dependencies are excluded.
- Lock-version differences, including a different PyTorch CUDA build tag, are
  informational. Missing packages, unsatisfied constraints, or unavailable SDPA/Qwen3
  APIs stop execution with a collected error list. Prepare a compatible environment;
  restart the kernel after replacing packages that have already been imported.
- After imports, the report includes PyTorch's CUDA build, cuDNN version, compiled GPU
  architectures, CUDA availability, visible GPU count, and each GPU's compute capability
  and total/free memory. CUDA must be available before model loading.

The CUDA version in `nvidia-smi` is the version supported by the driver; `nvcc` describes
an installed CUDA Toolkit; `torch.version.cuda` describes the PyTorch build. They can
differ. A standalone Toolkit is optional when using prebuilt PyTorch wheels, while the
driver must support the chosen build.

Allocator defaults follow the project: `PYTORCH_ALLOC_CONF=expandable_segments:True` is
set only when no explicit allocator setting or legacy `PYTORCH_CUDA_ALLOC_CONF` setting
is present. Diagnostics print selected fields only, without dumping environment variables.

When project dependencies change, refresh `RuntimeDiagnostics.LOCKED_VERSIONS` from
`uv.lock` and `PROJECT_REQUIREMENTS` from the matching inference entries in
`pyproject.toml`; the submitted notebook keeps these references embedded for offline use.

## What it does

| Stage | Behavior |
| --- | --- |
| Environment | Upload-time dependency setup, then system/driver/dependency diagnostics and Python 3.13+, offline dependency, SDPA/Qwen3 API, and CUDA checks before model loading |
| Serialization | One chat-template conversation per branch, same as training; **no truncation and no `max_length`** |
| Batching | Identical branches are encoded once. Branches are sorted longest first and packed with the better of descending best-fit and contiguous DP under `token_budget` padded tokens and `max_rows_per_batch` |
| Backbone | Merged Qwen3 safetensors shards streamed directly to `cuda:0` (embeddings + first half of layers) and `cuda:1` (second half + norm) in FP16 without materializing the 8 GB model in host RAM or installing accelerate hooks |
| Pipelining | Up to `pipeline_depth` batches are enqueued before the oldest is synchronised, so GPU0 works on batch k+1 while GPU1 finishes batch k. Inputs and pooled outputs use pinned memory |
| Attention | `use_gqa_in_sdpa` is patched (default `use_gqa="off"`: K/V heads are repeated), only Flash / memory-efficient / cuDNN kernels are allowed and math fallback is disabled. `"auto"` probes native `enable_gqa` and adopts it only if it works and is faster |
| Pooling and head | Last real token pooled **before** `Qwen3RMSNorm` (avoiding a full-sequence FP32 norm buffer), FP32 features, SR2 v3 head in FP32, temperature scaling followed by softmax in the competition column order |
| Output | `submission.csv` validated against `sample_submission.csv` (columns, ids, order, finite, rows sum to 1) |

All patches are restored when the run ends.

## Tunables

Set `TEMPERATURE = 1.0` at the very top of the first code cell. It populates
`SubmitConfig.temperature` and applies `softmax(logits / temperature)` when scoring.
The temperature must be finite and positive: `1.0` preserves the original probabilities,
values above `1` soften them, and values below `1` sharpen them. The notebook uses
the configured value directly; it does not fit a temperature automatically.

The other tunables are fields of `SubmitConfig` in the configuration block below the
startup diagnostics in the first code cell.

| Field | Default | Notes |
| --- | --- | --- |
| `temperature` | `1.0` | Set via `TEMPERATURE` at the top; scales head logits before softmax and leaves fallback priors unchanged |
| `token_budget` | `32768` | Padded tokens per batch. A single longer branch runs alone. Lower it after an OOM; the notebook also halves it automatically |
| `max_rows_per_batch` | `128` | Cap for many short branches |
| `pipeline_depth` | `2` | `1` disables overlap between the stages |
| `split_layer` | `None` | First layer of stage 1; `None` splits evenly (18/18) |
| `use_gqa` | `"off"` | `"off"`, `"on"` or `"auto"` as described above |
| `devices` | `("cuda:0", "cuda:1")` | With one visible GPU the model is not split |
| `class_prior` | `(0.349, 0.342, 0.309)` | Used only for rows with non-finite features or a failed single-branch OOM |

## Failure handling and caveats

- **OOM:** pending work is drained, caches are emptied, the failing batch is halved and the
  token limit is lowered. A single branch that still fails is counted and scored with the prior.
- **FP16 overflow:** features are checked per row. Non-finite rows use `class_prior` and are
  reported in the final warning. The BF16 weights are narrowed to FP16, so overflow is the
  main numerical risk of T4 inference.
- **Very long inputs:** nothing is truncated; Qwen3's trained position limit is 40,960
  tokens, and the longest training branch is 13,866.
- **Duplicated code:** the notebook copies the SR2 v3 head and the conversation serializer.
  When either changes in the package, change the notebook too.

