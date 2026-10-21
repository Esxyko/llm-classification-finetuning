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
  -> init -> Kaggle download -> stratified 80/20 holdout -> token balancing + A/B augmentation
     -> data/processed/train_validation.parquet

train_validation.parquet
  -> model MODEL (default) -> validation.csv + metrics -> result -> PNG + XLSX report
  -> model qwen3-4b --build -> models/qwen3_4b/head.pt
  -> model qwen3-4b --build --checkpoint-tag TAG -> models/qwen3_4b/head_TAG.pt
  -> model qwen3-4b-lora --build -> models/qwen3_4b_lora/adapter.pt
  -> model qwen3-4b-lora --build --checkpoint-tag TAG
     -> models/qwen3_4b_lora/adapter_TAG.pt
  -> model qwen3-4b-qlora --build -> models/qwen3_4b_qlora/adapter.pt
  -> model qwen3-4b-qlora --build --checkpoint-tag TAG
     -> models/qwen3_4b_qlora/adapter_TAG.pt

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
| `preprocess` | `preprocess/cli.py` | Rebuild the holdout split and augmentation from local raw data |
| `model` | `models/cli.py` | Select configured models, validate on the holdout, build checkpoints, or generate submissions |
| `result` | `result/cli.py` | Reduce holdout predictions into comprehensive reports |

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
  returns both download and split results; an optional download callback allows
  the CLI to print download progress before preprocessing finishes.
- `cli_output.py`: Formats shared download and split summaries for CLI commands.
- `data/`
  - `downloader.py`: Downloads the three required Kaggle CSVs, normalizes
    single-file ZIP responses (including existing wrappers), and skips downloads
    for existing files.
- `preprocess/`
  - `splits.py`: Derives labels and canonical prompt hashes, assigns stratified
    group partitions, invokes token balancing and augmentation, summarizes canonical
    token/class counts, and atomically writes Parquet only after balancing succeeds.
  - `token_balance.py`: Counts unpadded/untruncated canonical A+B tokens using
    the training serializer and shared token cache, then deterministically swaps
    equal-class whole prompt groups to meet the configured token tolerance.
  - `augmentation.py`: Appends one A/B-swapped counterpart per source row.
- `models/`
  - `__init__.py`: Registers each selector with its profile and pipeline.
  - `cli.py`: Selects a registered Qwen model and an execution mode; wraps each
    holdout validation model run in complete-run GPU peak monitoring.
  - `profile.py`: Maps selectors to configuration and artifact namespaces.
  - `qwen3_4b/`: Active lightweight model profile definition.
  - `qwen3_4b_lora/`: Shared LoRA/QLoRA adapter implementation with single-GPU
    training and dual-GPU submission. The selected profile controls training
    quantization, configuration keys, logs, and artifact namespaces.
    - `data.py`: Orient shared canonical token/feature views and independently
      right-pad branches without padding masks, with CPU pooling positions.
    - `batching.py`: Prepare training loaders from config without model weights
      or CUDA setup, consume the common token cache, persist holdout/build
      global batch plans, supply optional warmup identity contexts, and shuffle
      fixed update groups with explicit row counts.
    - `model.py`: Encode both branches through one differentiable LoRA backbone,
      optionally load a single-GPU NF4/double-quantized training base and prepare
      it before adapter injection, pool last real tokens, and classify with the
      shared pairwise MLP architecture through a LoRA-specific head subclass.
      Own BF16 frozen embedding/norm restoration, feature dtype, and head
      autocast with FP32 logits. Quantized weights are placed during loading.
    - `extractor.py`: Extract one no-gradient canonical feature pair per needed
      ID for warmup, each joint-validation pass, or submission, retaining shared
      temporary CPU features (BF16 in BF16 mode, otherwise FP32) with
      stage-specific logs and no persistent caches.
    - `trainer.py`: Run fixed-rate head-only warmup, then jointly train fresh
      adapters/head for one holdout run or all-data build with linear decay, mixed
      precision, row-normalized update groups, frozen-feature warmup validation,
      fresh-feature extraction followed by averaged head inference for joint
      validation/test, and one weighted progress counter
      shared across all executed training/validation stages.
    - `warmup_cache.py`: Identify completed head warmup by selected original
      training/validation ID sets and warmup optimization settings, atomically store FP32 head
      and named AdamW/scaler/random states, and restore compatible states across
      precision changes without retaining features or adapters.
    - `checkpoint.py`: Atomically store adapters/head and validate portable
      architecture/input compatibility; retain the exact Hub base revision.
    - `pipeline.py`: Orchestrate holdout validation, builds, and dual-GPU submissions
      without persistent embedding caches; explicitly disable quantization for
      submissions, including QLoRA.
  - `qwen3_4b_qlora/`: Register the independent QLoRA profile, selecting the
    shared adapter pipeline with fixed 4-bit NF4/double-quantized training and
    validation, and non-quantized submission at the checkpoint's base revision.
  - `archive/`: Git-ignored local copies of the retired Qwen3-1.7B model
    implementations and their configuration sections. Nothing imports these
    models from the active package.
  - `common/pipeline.py`: Selects holdout validation, build, or submission behavior for
    frozen pairwise models.
  - `common/`
    - `backbone.py`: Fingerprint local backbone weights/configuration by content
      for frozen embedding and adapter head-warmup caches.
    - `contracts.py`: Defines `ModelMode.VALIDATION`, `BUILD`, and `SUBMIT`,
      optional-cache result summaries (including `ModelSubmitResult`), and the
      shared registry pipeline interface.
    - `inputs.py`: Loads training/test tables, checks required columns, split membership and prompt isolation,
      orientation counts and label mappings, and test/template ID order; computes
      source fingerprints for all active profiles.
    - `data.py`: Serializes aligned conversation branches for pairwise models.
    - `compatibility.py`: Builds shared cache and checkpoint compatibility keys.
    - `publishers.py`: Stages and atomically publishes validation.csv and single-run
      metrics in one run directory, and publishes test submissions;
      atomically records each model's complete
      runtime and optional GPU memory peaks under its selected config.
    - `hardware.py`: Validates the training GPU or both submission GPUs, activates
      the first selected device, and maps complete Qwen3 decoder layers across
      the two submission devices. Owns the immutable GPU peak summary and
      per-run allocator monitoring without clearing caches.
    - `extractor.py`: Resolve pinned backbone/token identities on CPU, hash local
      weights when needed, and lazily encode only missing frozen branches on GPU.
    - `cache_storage.py`: Own the shared disk-backed cache file sessions, Windows/Linux
      process locks, atomic publication, checksummed multipart storage at 100 MiB,
      and corruption archives. Volumes are at most 99 MiB; private `.cache-work/`
      files are Git-ignored and removed after sessions.
    - `cache.py`: Persist per-branch embeddings by effective tokens and backbone
      computation identity in SQLite shards shared across datasets/configs/modes;
      rebind vectors to current positions and recompute only missing/corrupt rows.
    - `token_cache.py`: Independently cache individual text encodings by text,
      tokenizer content, and encoding options in shared SQLite shards; provide
      compact CPU text views, canonical pair views, and LoRA context validation.
    - `model.py`: Defines backbone pooling and the pairwise MLP head.
    - `orientation.py`: Averages original and swapped class probabilities and
      aligns validation predictions with processed split rows.
    - `progress.py`: Counts weighted training update groups and validation batches,
      flushing progress logs at five-percent milestones using CPU-only counters.
    - `trainer.py`: Returns one completed holdout head result or trains one
      full-data head, sharing a training/validation progress counter across all
      executed epochs.
    - `checkpoint.py`: Saves and validates the production head and compatibility
      metadata.
    - `predictor.py`: Runs no-gradient inference from a saved head.
- `result/`
  - `synthesizer.py`: Discovers and validates validation.csv, reverses swapped-row
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
| `config.yaml` | Data paths, validation settings, training `gpu.device`, two `gpu.submit_devices`, and three optional model sections |
| `.env.example` | Template for the local Kaggle token in `.env` |
| `pyproject.toml` / `uv.lock` | Package metadata, commands, and locked dependencies |
| `README.md` | Project overview and documentation index |
| `docs/` | Detailed setup, usage, configuration, and behavioral references |
| `structure.md` | Code ownership, runtime flow, artifact layout, and implementation invariants |
| `scripts/validate_all_configs.py` | Validate the top-level `MODELS` selection (LoRA by default) for each YAML file in `configs/` |
| `scripts/build_all_configs.py` | Train the top-level `MODELS` selection (LoRA by default) on all original IDs from both partitions for each YAML file in `configs/`, with separate tagged checkpoints |
| `scripts/count_split_tokens.py` | Count actual unpadded, untruncated A/B tokens in the saved split using training serialization; publish per-ID CSV, split CSV, and tokenizer/source metadata |
| `configs/` | Complete YAML configurations with all three models and shared training/submission GPU settings |
| `data/raw/` | Downloaded competition CSVs |
| `data/processed/` | Default location for train/validation Parquet data, shared tokens, frozen embeddings, and LoRA batch plans |
| `data/processed/token_cache/` | Shared per-text token encodings for all models and training/test inputs |
| `data/processed/embedding_cache/` | Per-branch frozen embeddings shared across training/test, with separate computation identities |
| `data/processed/adapter_batches/` | Shared LoRA/QLoRA holdout/build batch-plan caches, keyed by selected rows, lengths, budget, and update-group size |
| `data/processed/qwen3_4b_lora_warmup/` | Completed LoRA head warmup and training-state caches by actual training IDs |
| `data/processed/qwen3_4b_qlora_warmup/` | Independent completed QLoRA head-warmup caches |
| `models/qwen3_4b/` | Saved classifier head and metadata for the active model |
| `models/qwen3_4b_lora/` | Saved adapters, classifier head, and metadata in one `adapter[_TAG].pt` file |
| `models/qwen3_4b_qlora/` | QLoRA adapters, classifier head, and quantization metadata in one `adapter[_TAG].pt` file |
| `results/` | Validation runs, test submissions, and comprehensive reports |
| `results/submit/` | Timestamped submission CSVs; report discovery excludes this and historical `results/test/` |

`data/`, `models/`, and `results/` are runtime artifacts rather than source
modules. Git tracks the shared model-cache directories under `data/processed/`,
including multipart manifests and volumes, while `.cache-work/` directories
remain ignored. Do not make application logic depend on a particular timestamped run.

## Invariants to preserve

- All three models use the same saved 80/20 holdout with seed 42. Prompt groups
  remain whole. Original-class quotas use largest remainders and the nearest
  total validation row count; multirow groups are selected first, then singleton
  groups fill quotas. Infeasible quotas fail without publishing a dataset.
- Row and token proportions use canonical inputs before A/B augmentation.
  Validation row deviation `abs(validation_share / validation_fraction - 1)`
  must be strictly below 0.005. Token deviation uses the same formula and must
  be strictly below `validation.max_token_deviation` (default 0.005). Token
  counts include special tokens and exclude padding/truncation. Equal-class
  whole-group exchanges preserve the initial row/class counts, with at most
  1,000 improving swaps. Failure leaves the previous dataset in place.
- Training input validation requires both partitions, one split per prompt group,
  complete orientation pairs and matching labels, and, when supplied, the
  configured validation fraction. Legacy fold artifacts/configurations fail with
  migration instructions. Old files are retained.
- Every source row has exactly one swapped row with the same `id`, `group_id`, and
  `split`; A/B fields and labels `0`/`1` are reversed, while label `2` is unchanged.
- Prediction CSVs use `id`, `winner_model_a`, `winner_model_b`, and `winner_tie`
  in that order, with finite probabilities that sum to one.
- Frozen embeddings are shared across training/test and addressed by actual
  token IDs after truncation, independent of labels, splits, IDs, reference order,
  source files, serializers, and configured length caps when effective IDs match.
  Namespaces retain the exact Hub model revision or local weight/configuration
  content, backbone hidden size, precision, pooling/storage format, and extraction
  version. Attention implementation and TF32 are excluded. Different identities
  coexist. Reads check blob length and checksum without scanning for NaN/Inf;
  newly extracted vectors are checked for finite values before caching.
  Returned feature rows follow current canonical IDs and order. Duplicate inputs
  share entries; only missing/corrupt branches are encoded. Complete hits do not
  load backbone weights. Missing branches are independently right-padded, so
  floating-point rounding may differ with batch composition. Old dataset-wide
  embedding files remain untouched and are not reused. Head checkpoint
  compatibility is validated independently of embedding reuse.
- Token encodings are shared across models, training/test inputs, datasets, and
  configurations when their actual tokenizer/encoding content matches. Keys use
  individual text content, encoding options, cache schema, and tokenizer
  implementation/library versions, excluding source paths/IDs, labels, splits,
  row order, and model-weight identities. Tokenizer fingerprints omit padding,
  truncation, and decoding state; full encodings remain reusable across context
  limits. Frozen extraction separately caches exact tokenizer truncations only
  for overflowing texts, then right-pads missing branches for feature extraction.
  Missing/corrupt entries are recomputed transactionally; corrupt shards are
  archived before rebuilding. Reads check token count, byte length, and checksum
  without scanning IDs for negative values; newly encoded IDs are checked before
  caching. Refresh replaces only requested encodings.
- All active profiles serialize aligned prompt/response turns as
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
  disk through the common text cache, and shared across epochs/stages/runs.
  For Hub backbones, a pinned base/tokenizer revision is resolved before training
  model initialization, while cache identities use the actual tokenizer content
  rather than model revision. Each token-store preparation revalidates the
  configured context limit.
  No LoRA input is truncated: empty or over-limit branches fail before training/encoding with
  the offending ID and branch. Warmup extracts each training/validation ID once
  in canonical orientation using `warmup_extraction_batch_size`, then selects
  oriented feature views with `warmup_training_batch_size`. Each warmup batch
  performs one optimizer update without `gradient_accumulation_steps`, which
  applies only to joint training. Warmup validation reuses the frozen CPU features
  in `head_inference_batch_size` batches. Both feature
  views are released at the start of joint training; `warmup_epochs: 0` skips
  warmup feature extraction. Each joint-validation pass and test submission
  extracts fresh canonical features from the current model in eval/no-gradient
  mode using the same `warmup_extraction_batch_size`, then runs the head in
  `head_inference_batch_size` batches. Temporary CPU inference features use BF16
  in BF16 mode and FP32 otherwise; they are released on return, never persisted
  or reused across joint epochs/runs.
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
  lengths, budget, group size, and algorithm version. LoRA and QLoRA share the
  same plan namespace; quantization does not affect planning. Labels, source row order,
  split names, token content beyond lengths, and unselected inputs are
  excluded. Stored sample identities are rebound to current indices on every
  read, and current labels are used. Group membership stays fixed;
  each zero-based joint epoch shuffles groups and their micro-batches with
  `random_state + joint_epoch` for both holdout runs and builds. The short group can
  appear anywhere and supplies its actual row count to the trainer. Over-budget
  singletons retain all tokens. Update-group row counts, clipping/update
  boundaries, and scheduler step counts are preserved; grouping, floating-point
  order, and dropout draws may differ from fixed-batch training.
  `--refresh-cache` refreshes requested shared token encodings, LoRA plans,
  and completed head warmup;
  test inference uses the same text cache with its checkpoint's tokenizer revision.
  Uncached total epochs equal `warmup_epochs + epochs`; both phases retain the head's
  optimizer state and normalize loss by actual row counts per update, including
  short warmup batches and incomplete joint accumulation groups. Validation and
  test average both orientations' probabilities.
- LoRA/QLoRA head-warmup caches identify only the sorted original training ID
  set; builds use the full original ID set. Validation IDs and holdout/build mode
  do not affect identity, because validation does not update the head or select
  a stopping checkpoint. Split names, paths, row order, source-file bytes, text,
  and labels do not affect this data identity.
  Keys also include
  exact Hub revision or local weight/configuration contents, tokenizer and
  serialization identities, backbone/head dimensions, head dropout,
  quantization, A/B objective, warmup epochs/training batch size/learning rate,
  effective seed, weight decay, gradient clipping, and cache schema. Compute
  precision, attention/TF32, extraction batch size, all LoRA initialization
  settings, local model paths, and joint learning rates/epochs/batching/
  accumulation/token budget are excluded. Schema v3 uses training-only ID
  identities; legacy v1/v2 warmup files are no longer read.
  Current input/length validation still runs. Completed
  caches atomically persist FP32 heads and named head AdamW moments/steps,
  scaler/source precision, and Python/NumPy/Torch CPU/selected-CUDA random states
  after the final warmup validation (or training for builds). Reads validate all
  restorable components; missing/corrupt/incompatible files rebuild. Reuse skips
  warmup extraction/training/validation and restores no adapter weights. Joint
  training retains current learning rates and schedule; only FP16-to-FP16 loads
  the saved scaler. Quantization profiles remain isolated; identical training
  sets/settings
  can reuse warmup across holdout/build calls. Zero warmup epochs
  disable cache I/O, and manual splits without context stay uncached. Histories,
  build epoch counts, timings, and progress include only actual executed work;
  joint epoch numbers retain the configured warmup offset. See
  [warmup caches](docs/caching.md#loraqlora-head-warmup).
- Holdout validation and builds use one shared `gpu.device`. `--submit` requires
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
- LoRA backbone inputs are independently right-padded without constructing or
  passing padding masks during warmup, joint training, validation, or submission.
  Qwen retains causal attention, so real tokens cannot attend to right padding;
  pooling uses CPU-computed last-real-token positions. With SDPA and standard
  Qwen3-4B, this enables native GQA without explicit K/V head repetition.
- LoRA SDPA encoding permits only Flash, memory-efficient, and cuDNN backends;
  math fallback is disabled for warmup, joint training, validation, submission,
  and gradient-checkpoint recomputation. Unsupported inputs fail, and scoped
  backend flags are restored without changing the frozen profile's policy.
- LoRA/QLoRA BF16 mode keeps residual states, activation-checkpoint layer inputs,
  pooled features, and CPU feature storage in BF16. The LoRA-specific head
  inherits the shared MLP without changing state-dict keys, uses BF16 autocast
  for all training/inference calls, and returns FP32 logits for losses and
  orientation averaging. Trainable adapter/head parameters, gradients, and AdamW
  moments remain FP32. FP16/FP32 modes retain FP32 features and head computation;
  the frozen model's precision policy is unchanged.
- LoRA checkpoint compatibility excludes hardware, compute precision, and batch
  settings; model name, classifier size/dropout, LoRA settings, input format,
  schema/class count, pooling, truncation side, and bias are checked. Context
  and A/B training mode are recorded but not checked. Hub base revisions are
  retained; local base weights are neither stored nor content-fingerprinted.
- QLoRA shares the LoRA training, batching, pooling, objective, attention,
  progress, and publication invariants above. Its base uses bitsandbytes NF4
  with double quantization for warmup, joint training, and validation; compute
  dtype follows `gpu.precision`, while adapter/head parameters remain FP32 and
  the base stays frozen. In BF16 mode, frozen embeddings and all Qwen3 RMSNorm
  weights are restored to BF16 after k-bit preparation, before adapter injection;
  RMSNorm retains FP32 variance math and returns BF16. K-bit preparation does
  not enable checkpointing or clear the allocator cache; the classifier
  configures checkpointing once with its SDPA
  recomputation context. Submission explicitly reloads the non-quantized base
  at the saved revision using the existing two-GPU map, then restores adapters
  and head without merging or dequantizing a saved base. Checkpoints store a
  checked `training_quantization` marker (`bnb-nf4-double` versus LoRA's `none`);
  a missing marker means legacy LoRA. Model sections and batch-plan namespaces
  stay independent, while token encodings are shared. Neither profile persists
  live embeddings. The batch scripts retain their LoRA-only defaults.
- Frozen head compatibility includes model name, serializer, classifier hidden
  size, and schema/class count. Context, compute precision, attention
  implementation, TF32, and A/B training mode are recorded but not checked.
  It does not pin the backbone revision or local weight contents; embedding
  cache identity is stricter in those respects. Inference additionally checks
  backbone hidden size.
- Model sections are optional. `model ALL` runs only configured profiles in
  registry order and rejects configurations with no model sections.
- Frozen full-data builds save only the classifier head; LoRA builds save
  adapters and head without base weights or optimizer state. Holdout validation
  does not save production checkpoints, but adapter profiles save completed
  head-warmup caches. Full-data builds have no validation set;
  holdout validation publishes the final epoch's predictions, with validation
  after every executed epoch (including uncached LoRA head warmup).
- Each holdout trainer returns one completed result after its final epoch.
  The publisher atomically exposes validation.csv and metrics.json together.
  Metrics use validation_strategy=holdout, validation_loss, a flat epochs list,
  and canonical/selected row counts, without fold counters or cross-fold means.
  Result synthesis consumes only split=validation, keeps both orientations for
  A/B alignment, and rejects legacy five-fold results. Excel uses a split column.
  Complete runtime and GPU peaks are recorded after the model run succeeds.
- Training progress uses one counter across all executed stages and epochs of a
  model run, including validation batches after every epoch. Totals use
  A/B-selected training rows and original validation IDs, rounding training and
  validation batches/groups separately. Training groups (including FP16-skipped updates) and
  validation batches count as one unit, except LoRA warmup training/validation
  batches count as half a unit each. Extraction, preparation, caches, and publication are
  excluded, and reused warmup contributes no work. Dynamic micro-batches
  preserve update-group boundaries. See
  [training progress](docs/models.md#training-progress).
- Holdout validation monitors PyTorch allocated and reserved memory peaks on
  `gpu.device` from before pipeline initialization through cleanup and result
  publication. Each model/config resets peaks once; epochs and stages share that
  interval. Monitoring does not clear caches or subtract initial occupancy.
  Successful runs print GiB values and atomically append integer byte counts
  alongside complete runtime in `metrics.json`; failed runs only attempt a
  terminal peak summary without masking the execution error. Builds and
  submissions do not enable monitoring. See [GPU memory peaks](docs/reports.md#gpu-memory-peaks).
- All active persistent caches use the common `CacheStorage` interface without
  changing their logical keys or serialization schemas. Serialized files at or
  above 100 MiB are published as a manifest and SHA-256-checked volumes of at
  most 99 MiB. Single-file and multipart inputs use the same read/update API;
  process locks cover reconstruction through publication or archival. Only
  changed cache sessions publish, and failed publication retains the previous
  valid artifact. Full working files and SQLite journals stay in Git-ignored
  `.cache-work/` directories. Production checkpoints are outside this size rule.
  See [shared cache storage](docs/caching.md#shared-storage-and-multipart-files).
- Generated files and directories are staged before publication. Report
  publication restores the previous directory after a recoverable promotion
  failure and reports the backup path if restoration fails.
