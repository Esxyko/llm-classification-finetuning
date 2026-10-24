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
     -> data/processed/train_validation.parquet (single file or manifest + volumes)

train_validation.parquet
  -> model MODEL (default) -> validation.csv + metrics -> result -> PNG + XLSX report
  -> model warmup --build (train partition only)
     -> data/processed/qwen3_4b_lora_warmup/<key>.pt
  -> model qwen3-4b --build -> models/qwen3_4b/head.pt
  -> model qwen3-4b --build --checkpoint-tag TAG -> models/qwen3_4b/head_TAG.pt
  -> model qwen3-4b-lora --build -> models/qwen3_4b_lora/adapter.pt
  -> model qwen3-4b-lora --build --checkpoint-tag TAG
     -> models/qwen3_4b_lora/adapter_TAG.pt

head.pt or adapter.pt + data/raw/test.csv + data/raw/sample_submission.csv
  -> model MODEL --submit -> results/submit/MODEL-.../submission.csv
```

All commands accept `--config PATH` and default to `config.yaml` in the working
directory. Relative config-file paths resolve from the working directory.
`AppConfig.load` reads `.env` there without overriding existing environment
variables. Package initialization also loads that `.env` before HF imports.
A non-empty `MODEL_PATH` stores downloaded Hub repositories directly in a
persistent native cache, such as `<MODEL_PATH>/models--Qwen--Qwen3-4B/`.
`model_storage.load_pretrained` passes that root as `cache_dir` for shared
config/tokenizer/backbone loads. Hugging Face manages revisions, download locks,
and resumable downloads; files persist across processes without packing or
unpacking. `HF_HOME` and authentication settings are preserved; missing or blank
`MODEL_PATH` retains native HF caching. Explicit local model directories load
directly. A relative `MODEL_PATH`
resolves from the working directory. `DATA_PATH` is no longer used.
Relative YAML raw/processed paths, model artifacts, and reports resolve from the
working directory regardless of the config file's location. Absolute YAML data
paths stay absolute.
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

- `__init__.py`: Loads working-directory `.env` without overriding existing
  environment variables before HF imports, and sets the default HF download endpoint to
  `https://hf-mirror.com` before library imports, preserving an existing
  `HF_ENDPOINT` environment variable. For this mirror, also defaults
  `HF_HUB_DISABLE_XET` to `1` to use HTTP downloads, preserving an explicit
  Xet setting and leaving other endpoints' Xet behavior unchanged.
- `config.py`: Immutable config value objects, strict YAML validation, `.env`
  loading, and raw/processed-data path resolution from the working directory.
  `WarmupModelConfig` owns shared head-warmup fields;
  `LoRAModelConfig` extends it with joint-training and adapter settings. The YAML
  loader resolves its three warmup optimization fields from the required top-level
  `warmup` section and rejects those keys in `qwen3_4b_lora`; zero warmup epochs
  disable LoRA warmup, while the standalone command requires positive epochs.
- `model_storage.py`: Routes config/tokenizer/backbone loads through the shared
  `load_pretrained` wrapper to a persistent native Hub cache at optional
  `MODEL_PATH`, preserving default caching and explicit local model paths.
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
    token/class counts, and atomically publishes Parquet through `CacheStorage`
    only after balancing succeeds, applying the shared multipart size policy.
  - `token_balance.py`: Counts unpadded/untruncated canonical A+B tokens using
    the training serializer and shared token cache, then deterministically swaps
    equal-class whole prompt groups to meet the configured token tolerance.
  - `augmentation.py`: Appends one A/B-swapped counterpart per source row.
- `models/`
  - `__init__.py`: Registers each selector with its profile and pipeline.
  - `cli.py`: Selects a registered Qwen model and an execution mode; wraps each
    holdout validation model run in complete-run GPU peak monitoring. Also
    accepts the explicit build-only `warmup` selector outside the registry,
    rejects unsupported flags before data/GPU setup, and keeps it out of `ALL`.
  - `profile.py`: Maps selectors to configuration and artifact namespaces.
  - `warmup.py`: Orchestrates train-partition-only non-quantized head-warmup
    cache builds from the `warmup` section, reusing adapter batching
    and training without adapters, joint plans, validation, or checkpoints.
  - `qwen3_4b/`: Active lightweight model profile definition.
  - `qwen3_4b_lora/`: LoRA adapter implementation with single-GPU training and
    dual-GPU submission. Shared quantization branches for the archived QLoRA
    profile remain here, along with their dependencies. Only the non-quantized
    LoRA profile is actively registered; profiles still control configuration
    keys, logs, and artifact namespaces.
    - `data.py`: Orient shared canonical token/feature views, flatten canonical
      A/B branches for extraction, and independently right-pad branch batches
      without padding masks, with CPU pooling positions.
    - `batching.py`: Prepare training loaders from config without model weights
      or CUDA setup, consume the common token cache, persist holdout/build
      global batch plans, supply optional warmup identity contexts, and own
      the merged optimizer-group layout shared by warmup, joint training, and
      step counts.
      Shuffle full packing groups before merging any short tail into the last
      epoch update, then optionally best-fit pack checkpointed singletons within
      each final update before collation, supplying explicit row counts.
      Its `warmup_only` path
      prepares train-only tokens and references without joint batch plans.
    - `model.py`: Encode both branches through one differentiable LoRA backbone,
      optionally load a single-GPU NF4/double-quantized training base and prepare
      it before adapter injection, pool last real tokens, and classify with the
      shared pairwise MLP architecture through a LoRA-specific head subclass.
      The `warmup_only` path loads a non-quantized frozen base without adapter
      injection. Own BF16 frozen embedding/norm restoration, feature dtype, and head
      autocast with FP32 logits. Select gradient checkpointing once per joint
      A/B micro-batch from the padded token threshold, retaining the state through
      backward and cleaning input-gradient hooks when disabled. Evaluation and
      no-gradient extraction disable checkpointing. Backbone weights are placed
      directly on their target GPUs during loading, including non-quantized
      training and standalone warmup.
    - `extractor.py`: Batch individual A/B branches and reconstruct one
      no-gradient canonical feature pair per needed ID for warmup, each
      joint-validation pass, or submission, retaining shared
      temporary CPU features (BF16 in BF16 mode, otherwise FP32) with
      stage-specific logs and no persistent caches.
    - `trainer.py`: Run fixed-rate head-only warmup, then jointly train fresh
      adapters/head for one holdout run or all-data build with linear decay, mixed
      precision, row-normalized merged update groups in both stages,
      frozen-feature warmup validation,
      fresh-feature extraction followed by averaged head inference for joint
      validation/test, and one weighted progress counter
      shared across all executed training/validation stages. Standalone warmup
      returns valid cache hits before loading weights and stops after head
      warmup on a miss. The retained QLoRA branch checks LoRA warmup before
      its own cache.
    - `warmup_cache.py`: Identify completed head warmup by selected original
      training ID sets and warmup optimization settings, retain actual source
      paths across profiles, atomically store FP32 head
      and named AdamW/scaler/random states, and restore compatible states across
      precision changes without retaining features or adapters.
    - `checkpoint.py`: Atomically store adapters/head and validate portable
      architecture/input compatibility; retain the exact Hub base revision.
    - `pipeline.py`: Orchestrate holdout validation, builds, and dual-GPU submissions
      without persistent embedding caches; explicitly disable quantization for
      submissions, including the retained QLoRA path.
  - `archive/`: Git-ignored local copies of the retired Qwen3-1.7B and QLoRA
    profiles and their configuration sections. `qwen3_4b_qlora/` retains its
    entry module and historical guide in `README.md`; archive-level
    `config.yaml` retains its settings. Nothing imports archived profiles from
    the active package. They are excluded from CLI choices and `ALL`; the
    active config loader rejects `qwen3_4b_qlora` as an unknown root key. Local
    archives are not included in a fresh clone.
  - `common/pipeline.py`: Selects holdout validation, build, or submission behavior for
    frozen pairwise models.
  - `common/`
    - `attention.py`: Scope SDPA submission compatibility to the inference process;
      force Transformers K/V-head expansion, enable fused backends without math,
      and restore the GQA predicate and backend flags after success or failure.
    - `backbone.py`: Fingerprint local backbone weights/configuration by content
      for frozen embedding and adapter head-warmup caches.
    - `contracts.py`: Defines `ModelMode.VALIDATION`, `BUILD`, and `SUBMIT`,
      optional-cache result summaries (including `ModelSubmitResult`), and the
      shared registry pipeline interface. `WarmupBuildResult` reports the
      logical cache path, reuse, oriented training rows, and executed epochs.
    - `inputs.py`: Loads training/test tables, checks required columns, split membership and prompt isolation,
      orientation counts and label mappings, and test/template ID order; reads
      processed Parquet through `CacheStorage` and computes source fingerprints
      from reconstructed bytes for all active profiles.
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
    - `cache_storage.py`: Own the shared disk-backed cache and processed-Parquet
      file sessions, Windows/Linux process locks, atomic publication, and
      checksummed multipart storage at 100 MiB,
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
      flushing two-line progress logs with elapsed/estimated total durations at
      five-percent milestones using CPU-only counters and a monotonic clock.
    - `trainer.py`: Returns one completed holdout head result or trains one
      full-data head, sharing a training/validation progress counter across all
      executed epochs.
    - `checkpoint.py`: Saves and validates the production head and compatibility
      metadata.
    - `predictor.py`: Runs no-gradient inference from a saved head.
    - `publishers.py`: Atomically publishes validation predictions, metrics, and
      a YAML snapshot of the effective shared/current-model configuration,
      including the separate `warmup` section for LoRA and excluding its three
      warmup optimization fields from the LoRA section;
      records complete-run timing/GPU peaks and publishes submission CSVs.
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
| `config.yaml` | Data paths, validation settings, training `gpu.device`, two `gpu.submit_devices`, two optional model sections, and a `warmup` section required by LoRA and standalone warmup |
| `.env.example` | Template for the local Kaggle token and optional `MODEL_PATH` in `.env` |
| `pyproject.toml` / `uv.lock` | Package metadata, commands, and locked dependencies |
| `requirements-submit.txt` | Submission-only installation pins for PyTorch 2.11.0+cu128 and Transformers 5.16.1; retains the normal training lock |
| `README.md` | Project overview and documentation index |
| `docs/` | Detailed setup, usage, configuration, and behavioral references |
| `structure.md` | Code ownership, runtime flow, artifact layout, and implementation invariants |
| `scripts/validate_all_configs.py` | Validate the top-level `MODELS` selection (LoRA by default) for each YAML file in `configs/` |
| `scripts/build_all_configs.py` | Train the top-level `MODELS` selection (LoRA by default) on all original IDs from both partitions for each YAML file in `configs/`, with separate tagged checkpoints |
| `scripts/count_split_tokens.py` | Count actual unpadded, untruncated A/B tokens in the saved split using training serialization; publish per-ID CSV, split CSV, and tokenizer/source metadata |
| `scripts/migrate_hf_models.py` | Copy or move existing native HF repositories directly into `MODEL_PATH`, preserving internal links, verifying file contents, and staging publication; supports source selection and dry runs |
| `<MODEL_PATH>/models--*/` | Persistent native Hub repositories, including weights, configurations, tokenizers, refs, and snapshots |
| `configs/` | Complete YAML configurations with both active models and shared training/submission GPU settings |
| `data/raw/` | Default downloaded competition CSV location, relative to the working directory |
| `data/processed/` | Default location for train/validation Parquet data, shared tokens, frozen embeddings, and LoRA batch plans |
| `data/processed/train_validation.parquet` | Logical path of the saved default train/validation split, required by standalone warmup; large files use `.manifest.json` and `.parts/` with regular Git storage |
| `data/processed/token_cache/` | Shared per-text token encodings for all models and training/test inputs |
| `data/processed/embedding_cache/` | Per-branch frozen embeddings shared across training/test, with separate computation identities |
| `data/processed/adapter_batches/` | Shared LoRA holdout/build base batch-plan caches, keyed by selected rows, lengths, checkpointing threshold, and update-group size; second-pass checkpointing budgets are excluded |
| `data/processed/qwen3_4b_lora_warmup/` | Completed non-quantized LoRA head warmup by actual training IDs, also written by standalone warmup |
| `data/processed/qwen3_4b_qlora_warmup/` | Historical quantized QLoRA head-warmup caches; retained without active CLI access |
| `models/qwen3_4b/` | Saved classifier head and metadata for the active model |
| `models/qwen3_4b_lora/` | Saved adapters, classifier head, and metadata in one `adapter[_TAG].pt` file |
| `models/qwen3_4b_qlora/` | Historical QLoRA adapters, classifier head, and quantization metadata in one `adapter[_TAG].pt` file; retained |
| `results/` | Validation runs, test submissions, and comprehensive reports |
| `results/MODEL-YYYYMMDD-HHMMSS/config.yaml` | Effective `data`, `validation`, `gpu`, and current model sections, using input YAML field names, resolved data paths, and loaded defaults |
| `results/submit/` | Timestamped submission CSVs; report discovery excludes this and historical `results/test/` |

`data/`, `models/`, and `results/` are runtime artifacts rather than source
modules. Git tracks the default `data/processed/train_validation.parquet` or its
multipart manifest and volumes, and the shared model-cache directories under
`data/processed/`, using regular Git storage. Raw competition data, other processed
outputs, and `.cache-work/` directories remain ignored. Do not make application
logic depend on a particular timestamped run.

## Invariants to preserve

- Both active models use the same saved 80/20 holdout with seed 42. Prompt groups
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
  Optional head-only warmup trains the head at a fixed learning rate over features
  from a no-gradient, evaluation-mode backbone with adapter forward computation
  disabled. Joint training then uses gradients through both branches and linear
  learning-rate decay.
  Canonical A/B token IDs are stored once in compact int32 CPU memory, cached on
  disk through the common text cache, and shared across epochs/stages/runs.
  For Hub backbones, a pinned base/tokenizer revision is resolved before training
  model initialization, while cache identities use the actual tokenizer content
  rather than model revision. Each token-store preparation revalidates the
  configured context limit.
  No LoRA input is truncated: empty or over-limit branches fail before training/encoding with
  the offending ID and branch. Warmup extracts each training/validation ID once
  in canonical orientation using `extraction_batch_size`, then selects
  oriented feature views with `warmup_training_batch_size`. All no-gradient
  extraction counts individual A/B branches per forward, matching frozen-model
  extraction, and reconstructs pairs even across odd-sized batches. Warmup
  extraction completes before epoch timing starts; complete-run timing includes
  it. Adapter state is restored after warmup extraction. Warmup forward batches
  retain their configured size; a short final batch accumulates gradients with
  the preceding full batch for one update, normalized by their combined row count.
  Standalone warmup uses the same rule. Nonempty warmup epochs perform
  `max(1, training_rows // warmup_training_batch_size)` updates; a lone short
  batch updates once and exact multiples retain one update per full batch.
  Warmup uses its fixed learning rate independently of
  `gradient_accumulation_steps`, which applies only to joint training.
  Warmup validation reuses the frozen CPU features
  in `head_inference_batch_size` batches. Both feature
  views are released at the start of joint training; `warmup_epochs: 0` skips
  warmup feature extraction. Each joint-validation pass and test submission
  extracts fresh canonical features from the current model in eval/no-gradient
  mode using the same `extraction_batch_size`, then runs the head in
  `head_inference_batch_size` batches. Temporary CPU inference features use BF16
  in BF16 mode and FP32 otherwise; they are released on return, never persisted
  or reused across joint epochs/runs.
  Joint-validation timing includes extraction, which is excluded from progress;
  progress counts head batches with the original stage weights independently
  of whether a feature loader was supplied. Zero warmup epochs still permit
  validation/submission extraction.
  Joint updates target `G = training_batch_size * gradient_accumulation_steps`
  rows. A short final group merges into the preceding full update, with loss
  normalized by their combined row count. An epoch smaller than `G` retains one
  update; exact multiples retain all full groups. Scheduling and progress use
  `max(1, training_rows // G)` updates per nonempty joint epoch across
  holdout/build modes and fixed/token-budget batching.
  The default `gradient_checkpointing_threshold: 0` retains initial fixed
  micro-batches with gradient checkpointing on every joint-training batch, while
  reusing cached tokens. With the separate `gradient_checkpointing_budget: 0`,
  it also retains the original seeded shuffle. A positive threshold
  compares three stable global A/B length sorts for each selected training set,
  forms cached packing groups of up to `G` rows, and
  best-fit packs each group by independently padded A/B token cost. Plans minimize
  candidate micro-batch counts, then padded tokens, and are cached separately
  from tokens. Plan keys use sorted selected identities/orientations and A/B
  lengths, threshold, group size, and algorithm version. LoRA and retained QLoRA
  plans share the same namespace; quantization does not affect planning. Labels, source row order,
  split names, token content beyond lengths, and unselected inputs are
  excluded. Stored sample identities are rebound to current indices on every
  read, and current labels are used. Cached packing-group membership stays fixed;
  each zero-based joint epoch shuffles full groups with
  `random_state + joint_epoch` for both holdout runs and builds, then appends any
  short group's micro-batches to the last shuffled full group for one final
  update. A lone short group runs by itself. A positive
  `gradient_checkpointing_budget` then best-fit packs only checkpointed singleton
  micro-batches inside each final update, using the same descending-length and
  identity ordering as initial packing. Combined padded A/B cost cannot exceed
  that budget; an oversized singleton remains alone without truncation. Existing
  multi-row batches and non-checkpointed singletons remain intact. Planned
  micro-batches are shuffled after this optional pass. A zero threshold with a
  positive budget shuffles rows by `random_state + joint_epoch` into initial
  fixed-size batches and applies the same final-group packing in memory.
  Final epoch batches are memoized so length queries agree with iteration without
  advancing RNG state. The budget defaults to `0`, preserving initial batches
  and their execution order, and need not exceed the threshold. With a positive
  threshold, micro-batches whose padded A/B cost is at or below it disable
  checkpointing; batches above it enable checkpointing for both branches through
  backward, including merged multi-row batches. Warmup and inference always
  disable it. The old token-budget name and boolean checkpointing configuration
  are rejected. Plan schema `4` uses `threshold` metadata instead of `budget`;
  obsolete checked-in plans are removed and older versions rebuild. Only initial
  packing groups are persisted, so the second-pass budget is excluded from cache
  keys and final epoch batches are never written to disk.
  Fixed and token-budget training use the same update-group row counts,
  clipping/update boundaries, and scheduler step counts; grouping, floating-point
  order, and dropout draws may differ.
  `--refresh-cache` refreshes requested shared token encodings, LoRA plans,
  and completed head warmup;
  test inference uses the same text cache with its checkpoint's tokenizer revision.
  Uncached total epochs equal `warmup_epochs + epochs`; both phases retain the head's
  optimizer state and normalize loss by actual row counts per update, including
  merged final groups in warmup and joint training. Clipping and optimizer/scaler
  handling occur once per resulting group. Validation and
  test average both orientations' probabilities.
- LoRA head-warmup caches and the retained QLoRA cache implementation identify
  only the sorted original training ID set; builds use the full original ID set. Validation IDs and holdout/build mode
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
  accumulation/token budget are excluded. Schema v4 retains training-only ID
  identities and identifies merged-tail warmup; legacy v1/v2/v3 warmup files are
  no longer read. Token, embedding, and batch-plan cache identities are preserved.
  Current input/length validation still runs. Completed
  caches atomically persist FP32 heads and named head AdamW moments/steps,
  scaler/source precision, and Python/NumPy/Torch CPU/selected-CUDA random states
  after the final warmup validation (or training for builds). Reads validate all
  restorable components; missing/corrupt/incompatible files rebuild. Reuse skips
  warmup extraction/training/validation and restores no adapter weights. Joint
  training retains current learning rates and schedule; only FP16-to-FP16 loads
  the saved scaler. Source quantization identities and directories remain
  distinct: the retained QLoRA branch first validates a non-quantized LoRA cache
  against all other current compatibility fields, then checks its own quantized cache, then runs
  quantized warmup on a miss. LoRA does not read QLoRA caches. Refresh bypasses
  both QLoRA candidates and rebuilds its native cache. Identical training
  sets/settings can reuse warmup across holdout/build calls. Standalone warmup
  uses only train IDs, requires positive warmup epochs, and creates no adapters
  or persistent features; normal full-data builds use a different ID set.
  Zero warmup epochs
  disable cache I/O, and manual splits without context stay uncached. Histories,
  build epoch counts, timings, and progress include only actual executed work;
  joint epoch numbers retain the configured warmup offset. See
  [warmup caches](docs/caching.md#lora-head-warmup).
- Holdout validation, builds, and standalone warmup use one shared `gpu.device`.
  Single-GPU frozen and adapter backbones load with `device_map={"": str(device)}`
  rather than moving the complete backbone after loading. Explicit submission
  maps are preserved; the default loading map does not disable LoRA joint-training
  gradient checkpointing. `--submit` requires
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
  Qwen3-4B, this enables native GQA without explicit K/V head repetition during
  training and validation.
- Both active profiles' SDPA submissions temporarily disable Transformers' native
  GQA predicate before model loading so K/V heads are explicitly repeated. This
  permits memory-efficient SDPA on T4 with PyTorch 2.11.0+cu128 and Transformers
  5.16.1. Flash, memory-efficient, and cuDNN backends are enabled, with math
  disabled. Both the original predicate and backend flags are restored on exit,
  including exceptions and sequential `ALL` runs. Explicit eager attention and
  training/validation retain their existing policies. The compatibility path
  runs inside the submission process, including subprocesses, without an
  external notebook patch. See [submission setup](docs/getting-started.md#submission-environment-pytorch-211--cuda-128).
- LoRA SDPA encoding permits only Flash, memory-efficient, and cuDNN backends;
  math fallback is disabled for warmup, joint training, validation, submission,
  and gradient-checkpoint recomputation. Unsupported inputs fail, and scoped
  backend flags are restored without changing the frozen profile's policy.
- LoRA BF16 mode (also retained in the QLoRA path) keeps residual states,
  activation-checkpoint layer inputs,
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
- Archived QLoRA support remains in shared LoRA modules with its quantization
  dependencies, but has no active model registration or configuration section.
  The retained implementation shares the training, batching, pooling, objective,
  attention, progress, and publication invariants above. Its base uses
  bitsandbytes NF4 with double quantization for uncached warmup, joint training,
  and validation; compute
  dtype follows `gpu.precision`, while adapter/head parameters remain FP32 and
  the base stays frozen. In BF16 mode, frozen embeddings and all Qwen3 RMSNorm
  weights are restored to BF16 after k-bit preparation, before adapter injection;
  RMSNorm retains FP32 variance math and returns BF16. K-bit preparation does
  not enable checkpointing or clear the allocator cache; the classifier
  selects checkpointing per joint A/B micro-batch with its SDPA recomputation
  context. Submission explicitly reloads the non-quantized base
  at the saved revision using the existing two-GPU map, then restores adapters
  and head without merging or dequantizing a saved base. Checkpoints store a
  checked `training_quantization` marker (`bnb-nf4-double` versus LoRA's `none`);
  a missing marker means legacy LoRA. The archived configuration was independent
  of LoRA; batch plans and token encodings remain shared, and live embeddings
  are never persisted. The batch scripts retain their LoRA-only defaults.
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
  The publisher atomically exposes validation.csv, metrics.json, and config.yaml
  together. Metrics do not contain a `config` field; the adjacent YAML includes
  `data`, `validation`, `gpu`, and the current model section. LoRA snapshots also
  include the required `warmup` section and omit the three warmup optimization
  fields from `qwen3_4b_lora`; frozen-model snapshots omit `warmup`. Other model
  sections are excluded.
  Metrics use validation_strategy=holdout, validation_loss, a flat epochs list,
  and canonical/selected row counts, without fold counters or cross-fold means.
  Result synthesis consumes only split=validation, keeps both orientations for
  A/B alignment, and rejects legacy five-fold results. Excel uses a split column.
  Complete runtime and GPU peaks are recorded after the model run succeeds.
- Training progress uses one counter across all executed stages and epochs of a
  model run, including validation batches after every epoch. Totals use
  A/B-selected training rows and original validation IDs, counting merged warmup
  and joint updates and rounding batch counts separately. Training groups
  (including FP16-skipped updates) and validation batches count as one unit,
  except warmup training groups and warmup validation batches count as half a
  unit each, including standalone warmup. Extraction, preparation, caches, and
  publication are excluded, and reused warmup contributes no work. Dynamic
  micro-batches
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
- All active persistent caches and the processed training Parquet use the common
  `CacheStorage` interface without changing logical keys or serialization schemas.
  Serialized files at or above 100 MiB are published as a manifest and
  SHA-256-checked volumes of at
  most 99 MiB. Single-file and multipart inputs use the same read/update API;
  process locks cover reconstruction through publication or archival. Only
  changed cache sessions publish, and failed publication retains the previous
  valid artifact. Full working files and SQLite journals stay in Git-ignored
  `.cache-work/` directories. Processed-data readers recognize logical presence
  through a single file or manifest and fingerprint reconstructed Parquet bytes;
  corrupt volumes are archived and require preprocessing again. Production
  checkpoints are outside this size rule.
  See [shared cache storage](docs/caching.md#shared-storage-and-multipart-files).
- Generated files and directories are staged before publication. Report
  publication restores the previous directory after a recoverable promotion
  failure and reports the backup path if restoration fails.
