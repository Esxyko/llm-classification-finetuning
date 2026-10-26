# Colab setup and persistence

[Documentation index](../README.md)

Run project commands inside the Colab runtime, including when VS Code is the
notebook frontend. Paths such as `/content/...` belong to that runtime; a local
Windows terminal or local `.env` does not configure its project process.

## Set up a runtime checkout

1. Select a CUDA runtime. Validation/build uses one GPU; submission requires
   two. Choose precision supported by the selected devices in the YAML you will
   actually run. For a T4, use `fp16` instead of the root preset's `bf16`.
2. Mount Drive in a notebook cell if you want persistent caches:

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
| `PROJECT_DIR` under `/content/` | Git checkout, installed environment, local data and outputs | Runtime-local |
| `MODEL_PATH` on mounted Drive | HF config/tokenizer/weight loaders write missing downloads when requested | Directly to Drive; reused by later runs |
| `CACHE_PATH` on mounted Drive | Token, embedding, batch-plan, and warmup caches publish when those stages complete | Directly to Drive; unchanged hits do not republish |
| `PROJECT_DIR/results/` | Logs stream during batch validation; completed model runs publish predictions, metrics, and YAML | Runtime-local until copied |
| `PROJECT_DIR/models/` | `--build` publishes a production checkpoint | Runtime-local until copied separately |
| `DRIVE_RESULTS_DIR` | Local notebook's post-run copy step | Destination only; application commands do not read this variable |

Setting a path does not copy existing files. If `CACHE_PATH` points to Drive,
cache publication happens during the run, independently of the results copy.
Large cache sessions reconstruct working files beside their logical cache;
allow space there for [multipart storage](caching.md#shared-storage-and-multipart-files).

Use paths that the mounted runtime can actually resolve and write. For a folder
exposed through a MyDrive shortcut, check the mounted path and destination
permissions in the runtime; a shortcut's location in the Drive web interface
alone does not establish write access. This guide does not verify any particular
account's shortcut target.

For manual runs, copy the completed result directory to Drive before releasing
the runtime. Also preserve the matching processed split and any production
checkpoint you need later. `MODEL_PATH` contains downloaded base weights, not
the project's trained `adapter.pt` or `head.pt`.

## Local notebook behavior

The working copy may include `!colab.ipynb`. It is ignored by Git and is not part
of a fresh clone; the manual workflow above does not depend on it. Its current
four-cell workflow is:

| Cell | Action |
| --- | --- |
| Settings | Define repository/branch, runtime project path, model/cache paths, endpoint, and results-copy destination; export model/cache/endpoint settings |
| Drive | Mount `/content/drive` and create the model-cache directory |
| Setup | Clone, or fetch and hard-reset the runtime checkout to the selected remote branch; write runtime `.env`; install `uv`; synchronize dependencies |
| Validation and cleanup | Run `scripts/validate.py`; copy selected results in `finally`; flush/unmount Drive; request runtime release |

The setup cell's `git reset --hard` discards tracked edits in the **runtime
checkout**. Preserve those edits before rerunning that cell. It does not upload
local Windows project changes; the runtime obtains code from the configured
remote branch.

### When results are copied

Before validation, the notebook records existing names under `results/`. Once
the whole batch script returns, or raises an ordinary exception, its `finally`
block copies entries whose names were not present before the run. The `logs`
directory is always copied, even if it already existed. Existing non-log
entries are skipped; the notebook does not continuously mirror results.

Thus completed runs and log output can be copied after a later task fails.
The copy covers `results/`, not production checkpoints under `models/`. A killed
kernel or terminated runtime may never execute `finally`, so this is not an
interrupt-proof backup.

A nested cleanup block calls `drive.flush_and_unmount()`, then
`runtime.unassign()`, even after a validation or copy error reaches it. An error
inside cleanup can prevent its remaining calls. To continue interactive work,
change the cleanup behavior before running that cell; after an unmount, later
Drive operations require mounting again.

## Diagnose a failed run

- Read the **first model error** in the streamed output or
  `results/logs/validate-*.log`. The trailing `CalledProcessError` only reports
  that the child command failed.
- Check the selected YAML, runtime working directory, and exported environment
  rather than the local Windows settings.
- If copying failed, inspect the runtime's local `results/` and the mounted
  destination. The notebook's cleanup can still release the runtime afterward.
- Use the dedicated [submission environment](getting-started.md#submission-environment-pytorch-211--cuda-128)
  for submission pins; normal `uv run` uses the training lock.

See [troubleshooting](troubleshooting.md) for CUDA, memory, and input failures.
