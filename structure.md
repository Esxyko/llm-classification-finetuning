# Codebase Reference

Use this file to locate ownership before changing code. Keep orchestration in
pipelines, user interaction in CLI modules, and domain logic in focused classes.

## Documentation boundaries

| Document | Owns |
| --- | --- |
| [README.md](README.md) | Project overview, capabilities, and documentation index |
| [docs/](docs/) | Setup, commands, configuration, model behavior, cache/API usage, reports, and troubleshooting |
| [structure.md](structure.md) | Runtime flow, source-module responsibilities, artifact layout, and implementation invariants |

Keep step-by-step instructions and detailed usage references in `docs/`, and link
to them from the README. Update this file when code ownership, runtime flow,
artifact layout, or implementation invariants change; use links to the relevant
guide when a reader needs operational instructions.

## Runtime flow

```text
selected YAML config (defaults to config.yaml)
  -> init -> Kaggle download -> fold assignment + A/B augmentation
     -> data/processed/train_folds.parquet

train_folds.parquet
  -> model MODEL (default) -> fold CSVs + metrics -> result -> PNG + XLSX report
  -> model qwen3-4b --build -> models/qwen3_4b/head.pt
  -> model qwen3-4b --build --checkpoint-tag TAG -> models/qwen3_4b/head_TAG.pt
  -> model qwen3-4b-lora --build -> models/qwen3_4b_lora/adapter.pt
  -> model qwen3-4b-lora --build --checkpoint-tag TAG
     -> models/qwen3_4b_lora/adapter_TAG.pt

head.pt or adapter.pt + data/raw/test.csv + data/raw/sample_submission.csv
  -> model MODEL --submit -> results/submit/MODEL-.../submission.csv
```

All commands accept `--config PATH` and default to `config.yaml` in the working
directory. Relative config-file and data paths resolve from the working
directory. `init` and `data` read `.env` there; model artifacts and reports are
also rooted there, regardless of the config file's location.
The paths shown in the flow are those in the root preset; raw/processed paths
follow the selected YAML. Batch-config scripts instead run each model from the
project root derived from their own file location.

## Entry points

Console scripts are declared in `pyproject.toml`.

| Command | Module | Responsibility |
| --- | --- | --- |
| `init` | `cli.py` | Download missing raw files, then preprocess training data |
| `data` | `data/cli.py` | Download missing Kaggle files only |
| `preprocess` | `preprocess/cli.py` | Rebuild folds and augmentation from local raw data |
| `model` | `models/cli.py` | Select configured models, cross-validate, build checkpoints, or generate submissions |
| `result` | `result/cli.py` | Reduce fold predictions into comprehensive reports |

CLI modules parse arguments, translate expected domain errors into exit code `1`,
and print summaries. Business logic belongs below the CLI layer.

## Package map

`src/llm_classification_finetuning/`

- `config.py`: Immutable config value objects, strict YAML validation, and path
  resolution.
- `cli_config.py`: Defines the shared `--config` argument for all CLI parsers.
- `errors.py`: Shared expected-error hierarchy rooted at `DataPreparationError`.
- `timing.py`: Formats elapsed durations as unbounded `HH:MM:SS` values.
- `pipeline.py`: Orchestrates the complete download-and-preprocess workflow and
  returns both download and fold results; an optional download callback allows
  the CLI to print download progress before preprocessing finishes.
- `cli_output.py`: Formats shared download and fold summaries for CLI commands.
- `data/`
  - `downloader.py`: Downloads the three required Kaggle CSVs, normalizes
    single-file ZIP responses (including existing wrappers), and skips downloads
    for existing files.
- `preprocess/`
  - `folds.py`: Derives labels and canonical prompt hashes, assigns stratified
    group folds, invokes augmentation, and atomically writes Parquet.
  - `augmentation.py`: Appends one A/B-swapped counterpart per source row.
- `models/`
  - `__init__.py`: Registers each selector with its profile and pipeline.
  - `cli.py`: Selects a registered Qwen model and an execution mode; wraps each
    cross-validation model run in complete-run GPU peak monitoring.
  - `profile.py`: Maps selectors to configuration and artifact namespaces.
  - `qwen3_4b/`: Active lightweight model profile definition.
  - `qwen3_4b_lora/`: Adapter model with single-GPU training and dual-GPU submission.
    - `data.py`: Orient shared canonical token/feature views and independently
      pad branches with CPU pooling positions.
    - `batching.py`: Prepare training loaders from config without model weights
      or CUDA setup, consume the common token cache, persist per-fold/build
      global batch plans, and shuffle fixed update groups with explicit row counts.
    - `model.py`: Encode both branches through one differentiable LoRA backbone,
      pool their last real tokens, and classify with the shared pairwise MLP.
    - `extractor.py`: Extract one no-gradient canonical feature pair per needed
      ID for warmup, each joint-validation pass, or submission, retaining shared
      temporary FP32 CPU features with stage-specific logs and no persistent caches.
    - `trainer.py`: Run fixed-rate head-only warmup, then jointly train fresh
      adapters/heads per fold or all-data build with linear decay, mixed
      precision, row-normalized update groups, frozen-feature warmup validation,
      fresh-feature extraction followed by averaged head inference for joint
      validation/test, and one weighted progress counter
      shared across training/validation stages and folds.
    - `checkpoint.py`: Atomically store adapters/head and validate portable
      architecture/input compatibility; retain the exact Hub base revision.
    - `pipeline.py`: Orchestrate cross-validation, builds, and dual-GPU submissions
      without persistent embedding caches.
  - `archive/`: Git-ignored local copies of the retired Qwen3-1.7B model
    implementations and their configuration sections. Nothing imports these
    models from the active package.
  - `common/pipeline.py`: Selects cross-validation, build, or submission behavior for
    frozen pairwise models.
  - `common/`
    - `contracts.py`: Defines model modes (including `ModelMode.SUBMIT`),
      optional-cache result summaries (including `ModelSubmitResult`), and the
      shared registry pipeline interface.
    - `inputs.py`: Loads training/test tables, checks required columns, folds,
      orientation counts and label mappings, and test/template ID order; computes
      source fingerprints for both active profiles.
    - `data.py`: Serializes aligned conversation branches for pairwise models.
    - `compatibility.py`: Builds shared cache and checkpoint compatibility keys.
    - `publishers.py`: Stages and publishes validation runs with epoch/fold
      durations and test submissions; atomically records each model's complete
      runtime and optional GPU memory peaks under its selected config.
    - `hardware.py`: Validates the training GPU or both submission GPUs, activates
      the first selected device, and maps complete Qwen3 decoder layers across
      the two submission devices. Owns the immutable GPU peak summary and
      per-run allocator monitoring without clearing caches.
    - `extractor.py`: Resolve pinned backbone/token identities on CPU, hash local
      weights when needed, and lazily encode only missing frozen branches on GPU.
    - `cache.py`: Persist per-branch embeddings by effective tokens and backbone
      computation identity in SQLite shards shared across datasets/configs/modes;
      rebind vectors to current positions and recompute only missing/corrupt rows.
    - `token_cache.py`: Independently cache individual text encodings by text,
      tokenizer content, and encoding options in shared SQLite shards; provide
      compact CPU text views, canonical pair views, and LoRA context validation.
    - `model.py`: Defines backbone pooling and the pairwise MLP head.
    - `orientation.py`: Averages original and swapped class probabilities and
      aligns validation predictions with processed fold rows.
    - `progress.py`: Counts weighted training update groups and validation batches,
      flushing progress logs at ten-percent milestones using CPU-only counters.
    - `trainer.py`: Trains fold-specific heads or one full-data head, sharing a
      training/validation progress counter across all folds and epochs.
    - `checkpoint.py`: Saves and validates the production head and compatibility
      metadata.
    - `predictor.py`: Runs no-gradient inference from a saved head.
- `result/`
  - `synthesizer.py`: Discovers and validates fold CSVs, reverses swapped-row
    orientation, and reduces each pair to one source record.
    The `metrics.json` aggregation marker selects one already-averaged
    prediction per pair or two legacy orientation votes, regardless of model name.
  - `publisher.py`: Stages both report files, replaces the report directory,
    and restores the previous report on a publication failure.
  - `writers.py`: Renders the confusion-matrix PNG and filterable Excel workbook.

Each package `__init__.py` exposes its intended public surface. Prefer those
exports over importing private helpers across package boundaries.

## Files and generated artifacts

| Path | Role |
| --- | --- |
| `config.yaml` | Data paths, folds, training `gpu.device`, two `gpu.submit_devices`, and both model sections |
| `.env.example` | Template for the local Kaggle token in `.env` |
| `pyproject.toml` / `uv.lock` | Package metadata, commands, and locked dependencies |
| `README.md` | Project overview and documentation index |
| `docs/` | Detailed setup, usage, configuration, and behavioral references |
| `structure.md` | Code ownership, runtime flow, artifact layout, and implementation invariants |
| `scripts/validate_all_configs.py` | Cross-validate the top-level `MODELS` selection (LoRA by default) for each YAML file in `configs/` |
| `scripts/build_all_configs.py` | Train the top-level `MODELS` selection (LoRA by default) on all folds for each YAML file in `configs/`, with separate tagged checkpoints |
| `configs/` | Complete YAML configurations with both models and shared training/submission GPU settings |
| `data/raw/` | Downloaded competition CSVs |
| `data/processed/` | Default location for folded Parquet data, shared tokens, frozen embeddings, and LoRA batch plans |
| `data/processed/token_cache/` | Shared per-text token encodings for both models and training/test inputs |
| `data/processed/embedding_cache/` | Per-branch frozen embeddings shared across training/test, with separate computation identities |
| `data/processed/qwen3_4b_lora_batches/` | LoRA fold/build batch-plan caches; legacy token files are unused |
| `models/qwen3_4b/` | Saved classifier head and metadata for the active model |
| `models/qwen3_4b_lora/` | Saved adapters, classifier head, and metadata in one `adapter[_TAG].pt` file |
| `results/` | Validation runs, test submissions, and comprehensive reports |
| `results/submit/` | Timestamped submission CSVs; report discovery excludes this and historical `results/test/` |

`data/`, `models/`, and `results/` are runtime artifacts rather than source
modules. Do not make application logic depend on a particular timestamped run.

## Invariants to preserve

- Rows with the same canonical prompt must stay in one fold.
- Every source row has exactly one swapped row with the same `id`, `group_id`, and
  `fold`; A/B fields and labels `0`/`1` are reversed, while label `2` is unchanged.
- Prediction CSVs use `id`, `winner_model_a`, `winner_model_b`, and `winner_tie`
  in that order, with finite probabilities that sum to one.
- Frozen embeddings are shared across training/test and addressed by actual
  token IDs after truncation, independent of labels, folds, IDs, reference order,
  source files, serializers, and configured length caps when effective IDs match.
  Namespaces retain the exact Hub model revision or local weight/configuration
  content, backbone hidden size, precision, attention implementation, TF32,
  pooling/storage format, and extraction version. Different identities coexist.
  Returned feature rows follow current canonical IDs and order. Duplicate inputs
  share entries; only missing/corrupt branches are encoded. Complete hits do not
  load backbone weights. Missing branches are independently right-padded, so
  floating-point rounding may differ with batch composition. Old dataset-wide
  embedding files remain untouched and are not reused. Head checkpoint
  compatibility is validated independently of embedding reuse.
- Token encodings are shared across models, training/test inputs, datasets, and
  configurations when their actual tokenizer/encoding content matches. Keys use
  individual text content, encoding options, cache schema, and tokenizer
  implementation/library versions, excluding source paths/IDs, labels, folds,
  row order, and model-weight identities. Tokenizer fingerprints omit padding,
  truncation, and decoding state; full encodings remain reusable across context
  limits. Frozen extraction separately caches exact tokenizer truncations only
  for overflowing texts, then right-pads missing branches for feature extraction.
  Missing/corrupt entries are recomputed transactionally; corrupt shards are
  archived before rebuilding. Refresh replaces only requested encodings.
- Both active profiles serialize aligned prompt/response turns as
  `prompt\nresponse`, joined by blank lines. There is no active mono-input
  serialization or training path; archived implementations are outside the
  current module contracts.
- Qwen3-4B `A/B_swap: aug` trains on both orientations with per-row loss;
  `A/B_swap: inf` optimizes the averaged probability of both orientations for
  each original ID. Both modes average probabilities for validation and test.
- LoRA supports `A/B_swap: aug` with per-row cross-entropy on both orientations,
  and `A/B_swap: inf` with negative log-likelihood of averaged, A/B-restored
  probabilities once per original ID. Both warmup and joint training use the
  selected objective. Joint `aug` training encodes each oriented row separately;
  joint `inf` training reuses each branch encoding for both head orientations.
  Validation/test also reuse branch encodings for both head orientations.
  Training batches, accumulation, scheduler steps, and build row counts use
  the selected training rows (original IDs only for `inf`).
  Optional head-only warmup uses a no-gradient,
  evaluation-mode backbone at a fixed head learning rate, followed by joint
  training with gradients through both branches and linear learning-rate decay.
  Canonical A/B token IDs are stored once in compact int32 CPU memory, cached on
  disk through the common text cache, and shared across epochs/folds/runs.
  For Hub backbones, a pinned base/tokenizer revision is resolved before training
  model initialization, while cache identities use the actual tokenizer content
  rather than model revision. Each token-store preparation revalidates the
  configured context limit.
  No LoRA input is truncated: empty or over-limit branches fail before training/encoding with
  the offending ID and branch. Warmup extracts each training/validation ID once
  in canonical orientation using `warmup_extraction_batch_size`, then selects
  oriented feature views with `warmup_training_batch_size`. Warmup validation
  reuses the frozen CPU features in `head_inference_batch_size` batches. Both feature
  views are released at the start of joint training; `warmup_epochs: 0` skips
  warmup feature extraction. Each joint-validation pass and test submission
  extracts fresh canonical features from the current model in eval/no-gradient
  mode using the same `warmup_extraction_batch_size`, then runs the head in
  `head_inference_batch_size` batches. Temporary FP32 CPU inference features are
  released on return, never persisted or reused across joint epochs/folds.
  Joint-validation timing includes extraction, which is excluded from progress;
  progress counts head batches with the original stage weights independently
  of whether a feature loader was supplied. Zero warmup epochs still permit
  validation/submission extraction.
  The default `training_token_budget: 0` retains fixed micro-batches and the
  original seeded shuffle, while reusing cached tokens. A positive budget
  compares three stable global A/B length sorts for each selected training set,
  forms `training_batch_size * gradient_accumulation_steps` rows per update, and
  best-fit packs each group by independently padded A/B token cost. Plans minimize
  candidate micro-batch counts, then padded tokens, and are cached separately
  from tokens. Plan keys use sorted selected identities/orientations and A/B
  lengths, budget, group size, and algorithm version. Labels, source row order,
  fold identifiers, token content beyond lengths, and unselected inputs are
  excluded. Stored sample identities are rebound to current indices on every
  read, and current labels are used. Group membership stays fixed;
  each zero-based joint epoch shuffles groups and their micro-batches with
  `random_state + fold + joint_epoch` (builds omit fold). The short group can
  appear anywhere and supplies its actual row count to the trainer. Over-budget
  singletons retain all tokens. Update-group row counts, clipping/update
  boundaries, and scheduler step counts are preserved; grouping, floating-point
  order, and dropout draws may differ from fixed-batch training.
  `--refresh-cache` refreshes requested shared token encodings and LoRA plans;
  test inference uses the same text cache with its checkpoint's tokenizer revision.
  Total epochs equal `warmup_epochs + epochs`; both phases retain the head's
  optimizer state and normalize accumulation by actual row counts, including
  incomplete groups. Validation and test average both orientations' probabilities.
- Cross-validation and builds use one shared `gpu.device`. `--submit` requires
  exactly two distinct `gpu.submit_devices`, defaulting to `[cuda:0, cuda:1]`,
  independently of the training GPU. Both devices must support `gpu.precision`;
  GPU names and VRAM sizes are unrestricted. Embedding/rotary modules and the
  first `N // 2` decoder layers reside on the first GPU, remaining decoder layers
  and final normalization on the second. The classifier head resides on the
  first GPU; adapters follow their base layers. Submission uses no gradient
  checkpointing, replicas, CPU/disk offload, or single-GPU fallback. Inputs follow
  the embedding device, pooling indices follow hidden states, and pooled features
  return to the head device. Hardware placement is excluded from checkpoint and
  cache compatibility, and complete frozen cache hits skip backbone loading.
- LoRA checkpoint compatibility excludes hardware, compute precision, and batch
  settings; model name, classifier size/dropout, LoRA settings, input format,
  context, and A/B training mode are checked. Hub base revisions are retained;
  local base weights are neither stored nor content-fingerprinted.
- Frozen head compatibility includes model name, context, serializer, compute
  precision, attention implementation, TF32, classifier hidden size, and A/B
  training mode, as well as schema/class count. It does not pin the backbone
  revision or local weight contents; embedding cache identity is stricter in
  those respects. Inference additionally checks backbone hidden size.
- Model sections are optional. `model ALL` runs only configured profiles in
  registry order and rejects configurations with no model sections.
- Frozen full-data builds save only the classifier head; LoRA builds save
  adapters and head without base weights or optimizer state. Cross-validation
  does not save checkpoints. Full-data builds have no validation set;
  cross-validation publishes the final epoch's predictions, with validation
  after every epoch (including LoRA head warmup).
- Training progress uses one counter across all stages, folds, and epochs of a
  model run, including validation batches after every epoch. Totals use
  A/B-selected training rows and original validation IDs, rounding batches/groups
  separately per fold. Training groups (including FP16-skipped updates) and
  validation batches count as one unit, except LoRA warmup training/validation
  count as half a unit each. Extraction, preparation, caches, and publication are
  excluded. Dynamic micro-batches preserve update-group boundaries. See
  [training progress](docs/models.md#training-progress).
- Cross-validation monitors PyTorch allocated and reserved memory peaks on
  `gpu.device` from before pipeline initialization through cleanup and result
  publication. Each model/config resets peaks once; folds and epochs share that
  interval. Monitoring does not clear caches or subtract initial occupancy.
  Successful runs print GiB values and atomically append integer byte counts
  alongside complete runtime in `metrics.json`; failed runs only attempt a
  terminal peak summary without masking the execution error. Builds and
  submissions do not enable monitoring. See [GPU memory peaks](docs/reports.md#gpu-memory-peaks).
- Generated files and directories are staged before publication. Report
  publication restores the previous directory after a recoverable promotion
  failure and reports the backup path if restoration fails.
