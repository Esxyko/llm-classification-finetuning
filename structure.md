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
     -> cache/qwen3_4b_lora_EXPT2_warmup/<key>.pt
  -> model qwen3-4b --build -> models/qwen3_4b/head.pt
  -> model qwen3-4b --build --checkpoint-tag TAG -> models/qwen3_4b/head_TAG.pt
  -> model qwen3-4b-lora-EXPT2 --build -> models/qwen3_4b_lora_EXPT2/adapter.pt
  -> model qwen3-4b-lora-EXPT2 --build --checkpoint-tag TAG
     -> models/qwen3_4b_lora_EXPT2/adapter_TAG.pt
  -> model SR2-Qwen3-4B-lora --build [--checkpoint-tag TAG]
     -> models/SR2_Qwen3_4B_lora/adapter[_TAG].pt
     -> cache/SR2_Qwen3_4B_lora_warmup/<key>.pt (when warmup is enabled)

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
`AppConfig.cache_dir` resolves `CACHE_PATH` from `.env` or the environment,
defaulting to `./cache` when missing or blank. Relative paths resolve from the
working directory and `~` expands to the user's home directory. All shared
token, embedding, adapter batch-plan, and head-warmup caches use this root,
independently of YAML data paths and `MODEL_PATH`.
Relative YAML raw/processed paths, model artifacts, and reports resolve from the
working directory regardless of the config file's location. Absolute YAML data
paths stay absolute.
The paths shown in the flow use the root preset and default `CACHE_PATH`;
raw/processed paths follow the selected YAML. The batch validation script instead
runs each model from the project root derived from its own file location.

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
  loading, raw/processed-data path resolution, and `CACHE_PATH` resolution into
  `AppConfig.cache_dir` from the working directory.
  `WarmupModelConfig` owns shared head-warmup fields;
  `LoRAModelConfig` extends it with joint-training and adapter settings. All
  classification heads always include the absolute-difference feature.
  The YAML loader resolves its three warmup optimization fields from the required
  top-level `warmup` section and rejects those keys in either LoRA section. It also resolves
  internal `warmup_label_smoothing` from `warmup.label_smoothing` for all
  head warmup, while each LoRA section's `label_smoothing` controls only joint
  training. It resolves `warmup_extraction_budget` from `warmup.extraction_budget`,
  retaining the LoRA section's own extraction budget for joint validation and
  submission. All extraction budgets are required positive padded-token limits;
  the previous batch-size keys and both internal fields are rejected in YAML.
  Zero warmup epochs disable LoRA warmup, while the standalone command requires
  positive epochs. EXPT2 reads `qwen3_4b_lora_EXPT2`; SR2 reads
  `SR2_Qwen3_4B_lora` through the same loader. Both are optional. The retired shared
  `qwen3_4b_lora` section is rejected. Both LoRA profiles and standalone warmup
  reject `A/B_swap` in YAML and initialize their internal `ab_swap` policy to
  `none`; effective profile copies enforce the same original-only policy.
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
    Receives the shared cache root from orchestration for token balancing.
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
  - `profile.py`: Maps selectors to configuration/artifact namespaces and head architectures;
    resolves immutable effective settings, including both LoRA profiles' A/B policy override.
  - `warmup.py`: Orchestrates train-partition-only non-quantized head-warmup
    cache builds from the `warmup` section, reusing adapter batching
    and training with the EXPT2 head and original-only policy, without
    adapters, joint plans, validation, or checkpoints. Its cache namespace is
    shared with EXPT2; effective warmup settings and train IDs must match.
  - `qwen3_4b/`: Active lightweight model profile definition.
  - `SR2_Qwen3_4B_lora/`: Lightweight Skywork reward-backbone profile using
    `Skywork/Skywork-Reward-V2-Qwen3-4B` in the shipped presets, the EXPT2 head,
    and the shared LoRA pipeline below. Owns separate configuration, checkpoint,
    warmup-cache, and result namespaces. All EXPT2 training/inference invariants
    below apply to SR2. Standalone warmup continues to use the EXPT2 namespace.
  - `qwen3_4b_lora_EXPT2/`: LoRA adapter implementation with single-GPU training and
    dual-GPU submission. Shared quantization branches for the archived QLoRA
    profile remain here, along with their dependencies. EXPT2 and SR2 register
    this pipeline with their own identities. EXPT2 preserves its checkpoint,
    warmup, and result namespaces. Default Python training APIs resolve EXPT2.
    - `profile.py`: Own the EXPT2 identity shared by training, batching, and
      standalone warmup without importing the package registration surface.
    - `data.py`: Orient shared canonical token/feature views, flatten canonical
      A/B branches for extraction, and independently right-pad branch batches
      without padding masks, with CPU pooling positions.
    - `batching.py`: Prepare training loaders from config without model weights
      or CUDA setup, consume the common token cache, persist holdout/build
      global batch plans, supply optional warmup identity contexts, and own
      the merged optimizer-group layout shared by warmup, joint training, and
      step counts.
      Shuffle full packing groups before merging any short tail into the last
      full group, then optionally best-fit pack checkpointed singletons within
      each final update before collation. Keep the merged joint update last,
      supplying matching explicit row counts, as in warmup.
      Its `warmup_only` path
      prepares train-only tokens and references without joint batch plans.
    - `model.py`: Encode both branches through one differentiable LoRA backbone
      loaded through `AutoModel`; SR2 retains Skywork Transformer weights and
      omits its scalar reward head, using the same last-real-token pooling.
      Optionally load a single-GPU NF4/double-quantized training base and prepare
      it before adapter injection, pool last real tokens, and construct the
      selected comparison head through the LoRA head factory.
      The `warmup_only` path loads a non-quantized frozen base without adapter
      injection. Own BF16 frozen embedding/norm restoration, feature dtype, and head
      autocast with FP32 logits. Select gradient checkpointing once per joint
      A/B micro-batch from the padded token threshold, retaining the state through
      backward and cleaning input-gradient hooks when disabled. Evaluation and
      no-gradient extraction disable checkpointing. Backbone weights are placed
      directly on their target GPUs during loading, including non-quantized
      training and standalone warmup.
    - `heads.py`: Own the legacy precision-wrapped MLP and the experimental
      swap-equivariant heads, selected by one factory shared with cache validation.
      EXPT stores a fused symmetric projection with `b_u,b_w`, a directional
      projection, and one shared readout; one exact GELU precedes dropout with a
      shared win mask. EXPT2 projects symmetric/directional features into the
      two activated win branches, then projects their undropped mean once to
      produce the activated tie representation. Its win branches share a dropout
      mask, ties use an independent mask, and one readout scores all three classes.
      All heads retain BF16 autocast and FP32 logits.
    - `extractor.py`: Batch individual A/B branches and reconstruct one
      no-gradient canonical feature pair per needed ID for warmup, each
      joint-validation pass, or submission, retaining shared
      temporary CPU features (BF16 in BF16 mode, otherwise FP32) with
      stage-specific logs and no persistent caches.
    - `trainer.py`: Run fixed-rate head-only warmup, then jointly train fresh
      adapters/head for one holdout run or all-data build with fixed learning
      rates, mixed precision, row-normalized merged update groups in both stages,
      frozen-feature warmup validation,
      fresh-feature extraction followed by original-only head inference for joint
      validation/test, and one weighted progress counter
      shared across all executed training/validation stages. Standalone warmup
      returns valid cache hits before loading weights and stops after head
      warmup on a miss. EXPT2 and standalone warmup consume original rows only in both training
      stages, run one head call for validation/submission, and return one row per ID.
      The retained QLoRA branch checks LoRA warmup before
      its own cache.
    - `warmup_cache.py`: Identify completed head warmup by selected original
      training ID sets and warmup optimization settings, retain actual source
      paths across profiles, atomically store FP32 head
      and named AdamW/scaler/random states, and restore compatible states across
      precision changes without retaining features or adapters.
    - `checkpoint.py`: Atomically store adapters/head and validate portable
      architecture/input compatibility, including the selected head; retain
      the exact Hub base revision. Missing head markers identify the legacy MLP.
    - `pipeline.py`: Orchestrate holdout validation, builds, and dual-GPU submissions
      without persistent embedding caches; explicitly disable quantization for
      submissions, including the retained QLoRA path.
  - `archive/`: Git-ignored local copies of retired Qwen3-1.7B, QLoRA,
    standard LoRA, and EXPT profiles. Standard LoRA and EXPT were retired on
    2026-10-25; their directories retain original source, configuration/preset
    snapshots, notices, and historical workflow/configuration documentation.
    QLoRA retains its README and archive-level settings. Active code never
    imports this archive. Retired selectors are excluded from CLI choices and
    `ALL`, and their configuration keys are rejected. Historical results,
    checkpoints, and archived configurations remain in place. Retired-profile
    warmup caches are unused by active profiles and may be removed. Local
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
    - `extraction_batching.py`: Own the shared deterministic, in-memory
      `ExtractionBatchSampler` for frozen and adapter extraction, sorting branches
      by descending length and original index and best-fit packing by padded
      token cost. Length queries and repeated iteration use the same fixed plan.
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
    - `model.py`: Defines backbone pooling and the shared pairwise MLP head over
      `[A, B, |A - B|]`, including the embedding-combination marker
      for metrics.
    - `orientation.py`: Averages original and swapped class probabilities and
      aligns validation predictions with processed split rows.
    - `loss.py`: Shares uniform label smoothing for cross entropy on logits and
      A/B-averaged log probabilities, with mean or summed training reductions.
      Zero smoothing preserves the existing objectives; validation remains
      unsmoothed log loss.
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
    The `metrics.json` aggregation marker selects one original prediction,
    one already-averaged prediction per pair, or two legacy orientation votes,
    regardless of model name.
  - `publisher.py`: Stages both report files, replaces the report directory,
    and restores the previous report on a publication failure.
  - `writers.py`: Renders the confusion-matrix PNG and filterable Excel workbook.

Each package `__init__.py` exposes its intended public surface. Prefer those
exports over importing private helpers across package boundaries.

## Files and generated artifacts

| Path | Role |
| --- | --- |
| `config.yaml` | Data paths, validation settings, training `gpu.device`, two `gpu.submit_devices`, three optional model sections, and a `warmup` section required by LoRA and standalone warmup |
| `.env.example` | Template for the local Kaggle token and optional `MODEL_PATH` in `.env` |
| `pyproject.toml` / `uv.lock` | Package metadata, commands, and locked dependencies |
| `requirements-submit.txt` | Submission-only installation pins for PyTorch 2.11.0+cu128 and Transformers 5.16.1; retains the normal training lock |
| `README.md` | Project overview and documentation index |
| `docs/` | Detailed setup, usage, configuration, and behavioral references |
| `structure.md` | Code ownership, runtime flow, artifact layout, and implementation invariants |
| `scripts/validate.py` | Validate the explicit model-to-config filename lists in top-level `MODEL_CONFIGS`, resolving the string `CONFIG_DIR` from the project root and running models and configs in their declared order; stream script messages and combined child stdout/stderr to the terminal and one immediately flushed UTF-8 run log, retaining output and unexpected tracebacks on failure |
| `<MODEL_PATH>/models--*/` | Persistent native Hub repositories, including weights, configurations, tokenizers, refs, and snapshots |
| `configs/` | Complete YAML configurations for frozen and LoRA profiles with shared training/submission GPU settings |
| `data/raw/` | Default downloaded competition CSV location, relative to the working directory |
| `data/processed/` | Default location for train/validation Parquet data |
| `data/processed/train_validation.parquet` | Logical path of the saved default train/validation split, required by standalone warmup; large files use `.manifest.json` and `.parts/` with regular Git storage |
| `<CACHE_PATH>/` | Shared training-cache root, defaulting to `cache/` |
| `<CACHE_PATH>/token_cache/` | Shared per-text token encodings for all models and training/test inputs |
| `<CACHE_PATH>/embedding_cache/` | Per-branch frozen embeddings shared across training/test, with separate computation identities |
| `<CACHE_PATH>/adapter_batches/` | Shared LoRA holdout/build base batch-plan caches, keyed by selected rows, lengths, checkpointing threshold, and update-group size; second-pass checkpointing budgets are excluded |
| `<CACHE_PATH>/qwen3_4b_lora_warmup/` | Historical standard MLP head-warmup cache namespace; unused by active profiles |
| `<CACHE_PATH>/qwen3_4b_lora_EXPT_warmup/` | Historical EXPT head-warmup cache namespace; unused by active profiles |
| `<CACHE_PATH>/qwen3_4b_lora_EXPT2_warmup/` | Completed original-only EXPT2 head warmup, also written by standalone warmup; tracked using the shared cache-storage policy |
| `<CACHE_PATH>/SR2_Qwen3_4B_lora_warmup/` | Completed SR2 head warmup on the Skywork backbone, isolated from EXPT2 and tracked using the shared cache-storage policy |
| `<CACHE_PATH>/qwen3_4b_qlora_warmup/` | Historical quantized QLoRA head-warmup cache namespace; unused by active profiles |
| `models/qwen3_4b/` | Saved classifier head and metadata for the active model |
| `models/qwen3_4b_lora/` | Historical standard LoRA adapter/head checkpoints; retained |
| `models/qwen3_4b_lora_EXPT/` | Historical EXPT adapter/head checkpoints; retained |
| `models/qwen3_4b_lora_EXPT2/` | Saved EXPT2 adapters, symmetric/directional/tie projections, shared readout, and metadata |
| `models/SR2_Qwen3_4B_lora/` | Saved Skywork LoRA adapters, EXPT2 head, and metadata in `adapter[_TAG].pt` |
| `models/qwen3_4b_qlora/` | Historical QLoRA adapters, classifier head, and quantization metadata in one `adapter[_TAG].pt` file; retained |
| `results/` | Validation runs, test submissions, and comprehensive reports |
| `results/logs/` | One timestamped `validate-YYYYMMDD-HHMMSS-microseconds.log` per batch validation invocation, including failed runs; excluded from report source selection |
| `results/qwen3-4b-lora-EXPT2-*/` | EXPT2 original-only validation predictions, metrics, and effective configuration |
| `results/submit/qwen3-4b-lora-EXPT2-*/` | EXPT2 submissions generated from its own checkpoint |
| `results/SR2-Qwen3-4B-lora-*/` | SR2 original-only validation predictions, metrics, and effective configuration |
| `results/submit/SR2-Qwen3-4B-lora-*/` | SR2 submissions generated from its own checkpoint |
| `results/MODEL-YYYYMMDD-HHMMSS/config.yaml` | Effective `data`, `validation`, `gpu`, and current model sections, using input YAML field names, resolved data paths, and loaded defaults |
| `results/submit/` | Timestamped submission CSVs; report discovery excludes this and historical `results/test/` |

`data/`, `cache/`, `models/`, and `results/` are runtime artifacts rather than
source modules. Git tracks the default `data/processed/train_validation.parquet`
or its multipart manifest and volumes, and the shared model-cache directories
under the default `cache/` root, using regular Git storage. Raw competition data,
other processed outputs, and `.cache-work/` directories remain ignored. Do not
make application logic depend on a particular timestamped run.

## Invariants to preserve

- All active models consume the same saved holdout. The root preset uses an
  80/20 split with seed 42; the validation fraction and seed are configurable.
  Prompt groups remain whole. Original-class quotas use largest remainders and
  the nearest total validation row count; multirow groups are selected first,
  then singleton groups fill quotas. Infeasible quotas fail without publishing
  a dataset.
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
- The frozen profile uses the shared pairwise head with concatenated pooled
  features `[A, B, |A - B|]` and `3H` inputs for backbone hidden size `H`.
  EXPT2 and standalone warmup use the tie-from-wins head below. Legacy MLP
  and EXPT head definitions remain internal to the adapter implementation.
- EXPT2 uses `u=W_s[A+B;abs(A-B)]+b_1`, `v=W_d(A-B)` with
  `W_s:[h,2D]`, `W_d:[h,D]`, and no directional bias. One exact GELU acts on
  stacked `[u+v,u-v]` to produce `p_A,p_B`. The tie representation is
  `GELU(W_t((p_A+p_B)/2)+b_2)` with `W_t:[h,h]`, computed by one tie projection
  before any dropout. The symmetric, directional, and tie weights are trained
  directly without L/M matrices. Dropout shares the win mask and uses an
  independent tie mask; one shared bias-free `Linear(h,1)` scores the stacked
  `[p_A,p_B,t]`. Feature swaps exchange win scores and preserve ties under the
  same masks. Training and prediction use original rows only, with one
  head call per ID and no orientation averaging. Standalone warmup uses this
  same architecture and original-only policy.
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
- EXPT2 uses original rows and per-row cross entropy for both warmup and joint
  training. Training batches, accumulation, optimizer steps, progress, and build
  row counts use original IDs. Validation and test make one prediction per ID.
  Optional head-only warmup trains the head at a fixed learning rate over features
  from a no-gradient, evaluation-mode backbone with adapter forward computation
  disabled. Joint training then uses gradients through both branches and fixed
  adapter `learning_rate` and classifier `head_learning_rate` values.
  Canonical A/B token IDs are stored once in compact int32 CPU memory, cached on
  disk through the common text cache, and shared across epochs/stages/runs.
  For Hub backbones, a pinned base/tokenizer revision is resolved before training
  model initialization, while cache identities use the actual tokenizer content
  rather than model revision. Each token-store preparation revalidates the
  configured context limit.
  No LoRA input is truncated: empty or over-limit branches fail before training/encoding with
  the offending ID and branch. Warmup extracts each training/validation ID once
  in canonical orientation using `warmup.extraction_budget`, then selects
  oriented feature views with `warmup_training_batch_size`. All no-gradient
  extraction uses the shared sampler over individual A/B branches, matching
  frozen-model extraction. Branches are sorted by length descending and original
  branch index, then best-fit packed with cost `branches * longest_length` at or
  below the positive `extraction_budget`. Oversized branches run alone; existing
  truncation/context validation still applies. Each extractor writes features
  back by original branch index, preserving canonical ID and pair alignment even
  across reordered or odd-sized batches. Plans are fixed in memory and excluded
  from persistent cache/checkpoint identities. Warmup
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
  mode using the selected LoRA section's `extraction_budget`, then runs the head in
  `head_inference_batch_size` batches. Temporary CPU inference features use BF16
  in BF16 mode and FP32 otherwise; they are released on return, never persisted
  or reused across joint epochs/runs.
  Joint-validation timing includes extraction, which is excluded from progress;
  progress counts head batches with the original stage weights independently
  of whether a feature loader was supplied. Zero warmup epochs still permit
  validation/submission extraction.
  Joint updates target `G = training_batch_size * gradient_accumulation_steps`
  rows. A short final group merges into the preceding full update, with loss
  normalized by their combined row count, and the combined joint update runs
  last, as in warmup. An epoch smaller than `G` retains one
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
  short group's micro-batches to the last shuffled full group for one merged
  update. A lone short group runs by itself. A positive
  `gradient_checkpointing_budget` then best-fit packs only checkpointed singleton
  micro-batches inside each final update, using the same descending-length and
  identity ordering as initial packing. Combined padded A/B cost cannot exceed
  that budget; an oversized singleton remains alone without truncation. Existing
  multi-row batches and non-checkpointed singletons remain intact. Planned
  micro-batches are shuffled after this optional pass. The merged joint update
  remains last with its finalized membership and internal batch order.
  A zero threshold with a positive budget shuffles rows by
  `random_state + joint_epoch` into initial fixed-size batches and applies the
  same final-group packing in memory.
  Final epoch batches are memoized so length queries agree with iteration without
  advancing RNG state. The budget defaults to `0`, preserving initial batches
  and their order within each update, and need not exceed the threshold.
  Fixed batching with both settings at zero retains the original seeded batches
  and keeps the merged joint update last. With a positive
  threshold, micro-batches whose padded A/B cost is at or below it disable
  checkpointing; batches above it enable checkpointing for both branches through
  backward, including merged multi-row batches. Warmup and inference always
  disable it. The old token-budget name and boolean checkpointing configuration
  are rejected. Plan schema `4` uses `threshold` metadata instead of `budget`;
  older versions are not reused and rebuild when needed. Only initial
  packing groups are persisted, so the second-pass budget is excluded from cache
  keys and final epoch batches are never written to disk.
  Fixed and token-budget training use the same update-group row counts,
  clipping/update boundaries, and optimizer step counts; grouping, floating-point
  order, and dropout draws may differ.
  `--refresh-cache` refreshes requested shared token encodings, LoRA plans,
  and completed head warmup;
  test inference uses the same text cache with its checkpoint's tokenizer revision.
  Uncached total epochs equal `warmup_epochs + epochs`; both phases retain the head's
  optimizer state and normalize loss by actual row counts per update, including
  merged groups in warmup and joint training. Clipping and optimizer/scaler
  handling occur once per resulting group. Validation and test use one
  original prediction per ID.
- LoRA head-warmup caches and the retained QLoRA cache implementation identify
  only the sorted original training ID set; builds use the full original ID set. Validation IDs and holdout/build mode
  do not affect identity, because validation does not update the head or select
  a stopping checkpoint. Split names, paths, row order, source-file bytes, text,
  and labels do not affect this data identity.
  Keys also include
  exact Hub revision or local weight/configuration contents, tokenizer and
  serialization identities, backbone/head dimensions, head dropout,
  quantization, A/B objective, warmup epochs/training batch size/learning rate,
  effective seed, weight decay, gradient clipping, nonzero `warmup.label_smoothing`,
  and cache schema. Zero smoothing retains existing cache keys. Compute
  precision, attention/TF32, extraction token budget, all LoRA initialization
  settings, local model paths, and joint label smoothing/learning rates/epochs/batching/
  accumulation/token budget are excluded. Schema v5 preserves EXPT2 cache
  identities and architecture markers, retaining training-only ID identities
  and merged-tail warmup. Retired-profile caches remain separate; legacy
  v1/v2/v3/v4 warmup files are retained and no longer read. Token,
  embedding, and batch-plan cache identities are preserved.
  Current input/length validation still runs. Completed
  caches atomically persist FP32 heads and named head AdamW moments/steps,
  scaler/source precision, and Python/NumPy/Torch CPU/selected-CUDA random states
  after the final warmup validation (or training for builds). Reads validate all
  restorable components; missing/corrupt/incompatible files rebuild. Reuse skips
  warmup extraction/training/validation and restores no adapter weights. Joint
  training retains the configured fixed learning rates; only FP16-to-FP16 loads
  the saved scaler. Source quantization identities and directories remain
  distinct: the retained QLoRA branch first validates a non-quantized LoRA cache
  against all other current compatibility fields, then checks its own quantized cache, then runs
  quantized warmup on a miss. LoRA does not read QLoRA caches. Refresh bypasses
  both QLoRA candidates and rebuilds its native cache. Identical training
  sets/settings can reuse warmup across holdout/build calls. Standalone warmup
  uses only original train IDs, writes the EXPT2 cache namespace with the
  same architecture marker, requires positive warmup epochs, and creates no adapters
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
- All active profiles' SDPA submissions temporarily disable Transformers' native
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
  pooled features, and CPU feature storage in BF16. The LoRA head factory
  preserves legacy MLP state-dict keys and all heads use BF16 autocast
  for all training/inference calls, and returns FP32 logits for losses and
  orientation averaging. Trainable adapter/head parameters, gradients, and AdamW
  moments remain FP32. PEFT's `disable_input_dtype_casting` scopes BF16 adapter
  input handling during backbone encoding and checkpoint recomputation, letting
  autocast choose matmul precision without BF16-to-FP32-to-BF16 input round trips.
  Adapter parameter promotion stays enabled. FP16/FP32 modes retain FP32 features
  and head computation; the frozen model's precision policy is unchanged.
- LoRA checkpoint compatibility excludes hardware, compute precision, and batch
  settings; model name, classifier size/dropout, LoRA settings, input format,
  schema/class count, pooling, truncation side, and bias are checked. Context
  and A/B training mode are recorded but not checked. Hub base revisions are
  retained; local base weights are neither stored nor content-fingerprinted.
  Schema v2 checks EXPT2 head tensor names and shapes before loading the
  backbone. `head_architecture: swap_equivariant_tie_from_wins_v1` identifies
  its separate tie projection. Checkpoint schema v2 and warmup schema v5 remain
  unchanged. Architecture identities and factory-derived tensor checks reject
  retired-profile head states without changing existing EXPT2 cache keys.
  A missing marker identifies the legacy standard MLP; schema v1 checkpoints
  require rebuilding.
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
  are never persisted.
- Frozen head compatibility includes model name, serializer, classifier hidden
  size, and schema/class count. Context, compute precision, attention
  implementation, TF32, and A/B training mode are recorded but not checked.
  It does not pin the backbone revision or local weight contents; embedding
  cache identity is stricter in those respects. Inference additionally checks
  backbone hidden size. Schema v3 identifies the `3H` absolute-difference head;
  earlier checkpoints require rebuilding without invalidating branch embeddings.
- Model sections are optional. `model ALL` runs only configured profiles in
  registry order and rejects configurations with no model sections. Registry
  order is frozen Qwen3-4B, EXPT2, then SR2. The `qwen3_4b_lora_EXPT2` and
  `SR2_Qwen3_4B_lora` sections enable their respective profiles. Standard LoRA
  and EXPT selectors/exports are retired, and old LoRA configuration sections
  have no aliases. The batch validation script uses explicit `MODEL_CONFIGS`
  dictionary order and independent filename lists; its current selection is
  SR2 with `1.yaml` through `4.yaml`.
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
  fields and internal `warmup_extraction_budget` and `warmup_label_smoothing`
  from the selected LoRA section; both sections retain their own
  `extraction_budget` and `label_smoothing`.
  Frozen-model snapshots omit `warmup`. Other model sections are excluded.
  Metrics use validation_strategy=holdout, validation_loss, a flat epochs list,
  and canonical/selected row counts, without fold counters or cross-fold means.
  New runs record `head_architecture`; `embedding_combination` is
  `concat_a_b_abs_diff` or EXPT2's `sum_signed_diff_abs_diff`
  in metrics only, matching the selected head. EXPT2 and SR2 snapshots omit `A/B_swap`
  from the model and warmup sections; other snapshots retain the selected flags;
  historical results retain their original contents. Preset updates and model
  retirement must not rewrite saved snapshots, including their section names,
  parameter values, or resolved paths.
  Result synthesis consumes only split=validation. Both LoRA profiles mark
  `original_only` and retain original rows with one vote per ID; the frozen
  profile retains both orientations for A/B alignment. Historical LoRA results
  remain readable through their existing aggregation markers. Legacy
  five-fold results are rejected. Excel uses a split column.
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
