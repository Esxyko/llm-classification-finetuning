# Colab setup and persistence

[Documentation index](../README.md)

Run project commands inside the Colab runtime, including when VS Code is the
notebook frontend. Paths such as `/content/...` belong to that runtime; a local
Windows terminal or local `.env` does not configure its project process.

## Set up a runtime checkout

1. Select a CUDA runtime. Validation/build uses one GPU. Choose precision
   supported by the selected device in the YAML you will
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
   HF_HUB_OFFLINE=1
   ```

   Set Kaggle credentials only if downloading raw files. The saved processed
   manifest and all volumes from Git are sufficient for model validation.
   Populate `MODEL_PATH` with the complete native Hugging Face cache before
   running: config, tokenizer, refs/snapshots/blobs, and all weight shards. The
   separate local `download.ipynb` sets `HF_HUB_OFFLINE=0` for this preparation.
   Execution stays offline and reports any missing required local files instead
   of downloading them. Set offline mode before importing HF libraries; restart
   the kernel if they were already imported with different settings.
5. Run the selected configuration:

   ```shell
   uv run model SR2-Qwen3-4B-lora --config configs/1.yaml
   ```

   Or use `uv run python scripts/validate.py` for batch validation, or
   `uv run python scripts/build.py` for batch builds and standalone warmup.
   Both scripts currently select `configs/3.yaml` through their own constants.
   Editing `config.yaml` does not change a run that selects another YAML.

A notebook's `os.environ` is inherited by subprocesses it launches and overrides
`.env` values. A separately opened terminal does not receive later notebook
environment changes; configure that terminal's process or the runtime `.env`.

## Where files are written

| Path or setting | Writer and timing | Persistence in this workflow |
| --- | --- | --- |
| `PROJECT_DIR` under `/content/` | Git checkout, installed environment, and local data | Runtime-local |
| `MODEL_PATH` on mounted Drive | Separate online download preparation populates the native cache; offline model execution reads it | Directly on Drive; reused by later runs |
| `CACHE_PATH` on mounted Drive | Token, embedding, batch-plan, and warmup caches publish when those stages complete | Directly to Drive; unchanged hits do not republish |
| `RESULT_PATH` on mounted Drive | Batch logs stream during execution; completed model runs publish predictions, metrics, and YAML; comprehensive reports use their subdirectory | Directly to Drive |
| `PROJECT_DIR/models/` | `--build` publishes adapter/head and the merged backbone directory | Runtime-local until copied separately |
| `BUILD_OUTPUT_PATH` in `!build.ipynb` | Copy the current SR2 build directory after the batch finishes, including after a failed batch | Timestamped backup on Drive before runtime release |

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
and any production checkpoint you need later. Preserve the `merged[_TAG]/`
directory too when retaining the full trained backbone. `MODEL_PATH` contains
downloaded original base weights; build outputs are under `PROJECT_DIR/models/`.

## Local notebook behavior

The working copy may include `!validate.ipynb` and `!build.ipynb`. Both are
ignored by Git and are not part of a fresh clone; the manual workflow above
does not depend on them. Both use four cells:

| Cell | Action |
| --- | --- |
| Settings | Define repository/branch, runtime project path, model/cache/result paths, endpoint, and `HF_HUB_OFFLINE=1`; export these settings before HF imports; the build notebook also defines `BUILD_OUTPUT_PATH` |
| Drive | Mount `/content/drive` and create the model-cache directory |
| Setup | Clone, or fetch and hard-reset the runtime checkout to the selected remote branch; write model/cache/result paths, endpoint, and `HF_HUB_OFFLINE=1` to runtime `.env`; install `uv`; synchronize dependencies |
| Run and cleanup | The validation notebook runs `scripts/validate.py`, then flushes/unmounts Drive and requests runtime release in `finally`; the build notebook runs `scripts/build.py`, saves SR2 build outputs, then performs cleanup |

The setup cell's `git reset --hard` discards tracked edits in the **runtime
checkout**. Preserve those edits before rerunning that cell. It does not upload
local Windows project changes; the runtime obtains code from the configured
remote branch.

### When results are written

The settings cell points `RESULT_PATH` to the mounted Drive results directory.
Batch logs stream to `<RESULT_PATH>/logs/` throughout execution; each successful
validation publishes its completed run directory there. Report generation also
uses this root. There is no end-of-run results copy step.

Previously published results remain at the configured destination when a later
task fails. An interrupted training run may have no completed result directory;
direct output is not a live checkpoint or a guarantee against interrupted Drive
writes. Manual builds and the validation notebook require separate preservation
of production checkpoints under `models/`; the build notebook copies them as
described below.

The validation notebook's `finally` cleanup calls `drive.flush_and_unmount()`, then
`runtime.unassign()`, even after validation fails. An error inside cleanup can
prevent its remaining calls. To continue interactive work,
change the cleanup behavior before running that cell; after an unmount, later
Drive operations require mounting again.

### When build outputs are saved

The build notebook sets `BUILD_OUTPUT_PATH` to
`/content/drive/MyDrive/llm-classification-finetuning/models` by default. This
notebook-only setting is separate from `MODEL_PATH`, which holds the downloaded
base weights; it does not change the model CLI's output paths.

After the build batch finishes, its final cell copies the runtime's current
`models/SR2_Qwen3_4B_lora/` directory to
`<BUILD_OUTPUT_PATH>/build-<timestamp>/SR2_Qwen3_4B_lora/`. This includes the
adapter/head checkpoint, merged backbone, tokenizer, and export metadata that
exist there. Historical model directories are not copied. Backups use separate
timestamped directories; within a batch, multiple untagged builds still replace
the same runtime outputs, so the backup retains their final on-disk state.

The copy is also attempted after a failed batch to preserve any outputs already
produced, while retaining the original build error. If no SR2 directory exists,
as with a warmup-only run, copying is skipped; warmup caches are already written
to `CACHE_PATH`. A reused runtime can contain outputs from an earlier run, and
the backup includes the directory's current contents.

After copying succeeds or is skipped, the cell flushes/unmounts Drive and requests
runtime release. If copying fails, it reports the error and keeps Drive mounted
and the runtime available for recovery. A failed copy can leave an incomplete
backup directory. An error during Drive flushing prevents runtime release.

## Diagnose a failed run

- Read the **first model error** in the streamed output or
  `<RESULT_PATH>/logs/validate-*.log` or `build-*.log` (default `results/logs/`). The trailing
  `CalledProcessError` only reports that the child command failed.
- Check the selected YAML, runtime working directory, and exported environment
  rather than the local Windows settings.
- If writing failed, inspect the configured `RESULT_PATH`, Drive mount, and
  destination permissions. Cleanup can still release the runtime after a model
  failure; a build-output copy failure leaves it available for recovery.

See [troubleshooting](troubleshooting.md) for CUDA, memory, and input failures.
