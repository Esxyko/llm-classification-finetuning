# Colab setup and persistence

[Documentation index](../README.md)

Run project commands inside the Colab runtime, including when VS Code is the
notebook frontend. Paths such as `/content/...` belong to that runtime; a local
Windows terminal or local `.env` does not configure its project process.

## Set up a runtime checkout

1. Select a CUDA runtime. Validation/build uses one GPU; submission requires
   two. Choose precision supported by the selected devices in the YAML you will
   actually run. For a T4, use `fp16` instead of the root preset's `bf16`.
2. Mount Drive in a notebook cell if you want persistent caches and results:

   ```python
   from google.colab import drive

   drive.mount("/content/drive")
   ```

3. Clone the project and install its locked environment in the runtime terminal:

   ```shell
   git clone https://github.com/Esxyko/llm-classification-finetuning.git /content/llm-classification-finetuning
   cd /content/llm-classification-finetuning
   python -m pip install uv
   uv sync --locked
   ```

   For an existing checkout, keep or update it as intended instead of cloning
   over it. `uv` uses the project's Python 3.14 environment; the notebook
   kernel's interpreter can differ. Launch project work through `uv run`.
4. Create a `.env` in that runtime checkout, for example:

   ```dotenv
   MODEL_PATH=/content/drive/MyDrive/huggingface
   CACHE_PATH=/content/drive/MyDrive/llm-classification-finetuning/cache
   RESULT_PATH=/content/drive/MyDrive/llm-classification-finetuning/results
   HF_ENDPOINT=https://huggingface.co
   ```

   Set Kaggle credentials only if downloading raw files. The saved processed
   manifest and all volumes from Git are sufficient for model validation.
5. Run the selected configuration:

   ```shell
   uv run model SR2-Qwen3-4B-lora --config configs/1.yaml
   ```

   Or use `uv run python scripts/validate.py` for its explicit selection. Editing
   `config.yaml` does not change a run that selects `configs/1.yaml`.

A notebook's `os.environ` is inherited by subprocesses it launches and overrides
`.env` values. A separately opened terminal does not receive later notebook
environment changes; configure that terminal's process or the runtime `.env`.

## Where files are written

| Path or setting | Writer and timing | Persistence in this workflow |
| --- | --- | --- |
| `PROJECT_DIR` under `/content/` | Git checkout, installed environment, and local data | Runtime-local |
| `MODEL_PATH` on mounted Drive | HF config/tokenizer/weight loaders write missing downloads when requested | Directly to Drive; reused by later runs |
| `CACHE_PATH` on mounted Drive | Token, embedding, batch-plan, and warmup caches publish when those stages complete | Directly to Drive; unchanged hits do not republish |
| `RESULT_PATH` on mounted Drive | Batch logs stream during execution; completed model runs publish predictions, metrics, and YAML; submissions and comprehensive reports use their subdirectories | Directly to Drive |
| `PROJECT_DIR/models/` | `--build` publishes a production checkpoint | Runtime-local until copied separately |

Setting a path does not copy existing files. Point `CACHE_PATH` and `RESULT_PATH`
to Drive to publish their artifacts there during the run. When `RESULT_PATH` is
missing or blank, commands use `./results` in their working directory instead.
Large cache sessions reconstruct working files beside their logical cache;
allow space there for [multipart storage](caching.md#shared-storage-and-multipart-files).

Use paths that the mounted runtime can actually resolve and write. For a folder
exposed through a MyDrive shortcut, check the mounted path and destination
permissions in the runtime; a shortcut's location in the Drive web interface
alone does not establish write access. This guide does not verify any particular
account's shortcut target.

If manual runs use a runtime-local `RESULT_PATH`, copy the completed result
directory to Drive before releasing the runtime. With `RESULT_PATH` on Drive,
results are already written there. Also preserve the matching processed split
and any production checkpoint you need later. `MODEL_PATH` contains downloaded
base weights, not the project's trained `adapter.pt` or `head.pt`.

## Local notebook behavior

The working copy may include `!colab.ipynb`. It is ignored by Git and is not part
of a fresh clone; the manual workflow above does not depend on it. Its current
four-cell workflow is:

| Cell | Action |
| --- | --- |
| Settings | Define repository/branch, runtime project path, model/cache/result paths, and endpoint; export model/cache/result/endpoint settings |
| Drive | Mount `/content/drive` and create the model-cache directory |
| Setup | Clone, or fetch and hard-reset the runtime checkout to the selected remote branch; write model/cache/result paths and endpoint to runtime `.env`; install `uv`; synchronize dependencies |
| Validation and cleanup | Run `scripts/validate.py` with results and logs written directly to `RESULT_PATH`; flush/unmount Drive in `finally`; request runtime release |

The setup cell's `git reset --hard` discards tracked edits in the **runtime
checkout**. Preserve those edits before rerunning that cell. It does not upload
local Windows project changes; the runtime obtains code from the configured
remote branch.

### When results are written

The settings cell points `RESULT_PATH` to the mounted Drive results directory.
Batch logs stream to `<RESULT_PATH>/logs/` throughout execution; each successful
validation publishes its completed run directory there. Submission commands and
report generation also use this root. There is no end-of-run results copy step.

Previously published results remain at the configured destination when a later
task fails. An interrupted training run may have no completed result directory;
direct output is not a live checkpoint or a guarantee against interrupted Drive
writes. Production checkpoints under `models/` still need separate preservation.

The `finally` cleanup calls `drive.flush_and_unmount()`, then
`runtime.unassign()`, even after validation fails. An error inside cleanup can
prevent its remaining calls. To continue interactive work,
change the cleanup behavior before running that cell; after an unmount, later
Drive operations require mounting again.

## Diagnose a failed run

- Read the **first model error** in the streamed output or
  `<RESULT_PATH>/logs/validate-*.log` (default `results/logs/`). The trailing
  `CalledProcessError` only reports that the child command failed.
- Check the selected YAML, runtime working directory, and exported environment
  rather than the local Windows settings.
- If writing failed, inspect the configured `RESULT_PATH`, Drive mount, and
  destination permissions. The notebook's cleanup can still release the runtime
  afterward.
- Use the dedicated [submission environment](getting-started.md#submission-environment-pytorch-211--cuda-128)
  for submission pins; normal `uv run` uses the training lock.

See [troubleshooting](troubleshooting.md) for CUDA, memory, and input failures.
