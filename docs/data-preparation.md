# Prepare training data

[Documentation index](../README.md)

After [setup](getting-started.md), run `uv run init` to download missing raw files
and prepare data. Use `uv run data` for downloads only, or `uv run preprocess`
to rebuild from local raw files. All commands accept `--config PATH`.

## One shared train/validation split

All three models use the same saved split. Preparation starts from the complete
raw training table, derives labels and canonical prompt hashes, and assigns
whole prompt groups to training or validation. The default validation fraction
is 20%, using random seed 42. It does not select a fold from the previous data.

Original-class quotas use the nearest total validation row count and largest
fractional remainders. Multirow prompt groups are selected deterministically,
then singleton groups fill the remaining class quotas. Preparation fails if
the quotas cannot be filled without splitting prompts. Classes retain their
original proportions; they are not resampled to equal sizes.

Token counting uses the training conversation serializer and configured pinned
tokenizer, with special tokens and the shared text cache. Counts exclude padding
and truncation. Complete prompt groups with identical three-class count vectors
are exchanged until the validation token volume meets its target, preserving
the initial row count and class counts exactly. At most 1,000 improving exchanges
are allowed. No model weights or GPU are needed.

Both row and token proportions are measured **before A/B augmentation**. For
validation fraction `f`, the relative deviation is `abs(validation_share / f - 1)`.
Row deviation must be strictly below 0.005; token deviation must be strictly
below `validation.max_token_deviation` (default 0.005). At `f = 0.2`, this means
less than 0.5% relative error around the 20% target, not 0.5 percentage points.
The summary also shows each partition's deviation from its own target.

One swapped counterpart is appended to every original row after splitting.
Only after balance and integrity checks succeed is the Zstandard-compressed
Parquet atomically published. Failure leaves any previous dataset in place.

## Processed data

The default logical output is `data/processed/train_validation.parquet`.
The pipeline uses the same storage interface as caches: compressed Parquet files
smaller than 100 MiB remain single files; files at or above that size are published
as `train_validation.parquet.manifest.json` and a
`train_validation.parquet.parts/` directory, with each volume at most 99 MiB.
Commit the manifest and all referenced volumes together; the saved default split
uses regular Git storage. Keep `data.processed_path` pointing to the logical
`.parquet` path for either layout.

Model training, standalone warmup, and token statistics automatically reconstruct
and checksum multipart inputs in an ignored `.cache-work/` directory. The source
fingerprint hashes the reconstructed Parquet bytes, preserving its identity
across storage layouts. Existing single-file inputs remain readable and adopt
the size rule when preprocessing next writes them. Corrupt multipart data is
archived and loading fails with instructions to rerun preprocessing. See
[shared storage](caching.md#shared-storage-and-multipart-files) for locks, atomic
publication, and temporary disk-space requirements.

Source columns are retained, with these additions:

| Column | Meaning |
| --- | --- |
| `label` | `0` = model A wins, `1` = model B wins, `2` = tie |
| `group_id` | SHA-256 hash of canonical prompt JSON |
| `split` | `train` or `validation` |
| `is_swapped` | Whether this is the A/B-swapped counterpart |

Each ID has one original and one swapped row in the same prompt group and split.
Original rows precede swapped rows in source order. Input loading rejects
missing values, invalid labels, incomplete orientation pairs, swapped-label
mismatches, or prompt groups crossing the split. No original ID is dropped.

Every model serializes aligned conversation branches as chronological
`prompt\nresponse` turns joined by blank lines, without a chat template.
Prompt and response values must be JSON arrays with the same nonzero turn count.
Prompts must be strings; a response may be a string or `null`, which becomes an
empty response. LoRA/QLoRA reject over-limit inputs rather than truncating them.

## Token statistics

After regeneration, run:

```shell
uv run python scripts/count_split_tokens.py --local-files-only
```

The script uses training serialization and shared token encodings while retaining
the saved split. Its default model is `qwen3-4b-lora`; `--model`, `--config`,
`--revision`, `--batch-size`, and `--output-dir` select alternatives. When the
model name matches `validation.tokenizer_name`, the pinned validation tokenizer
revision is used unless overridden. Omit `--local-files-only` to permit downloads.

Reports default to `data/processed/split_token_stats/`:

| File | Contents |
| --- | --- |
| `per_id.csv` | Original ID, split, label, A/B lengths, and total tokens |
| `splits.csv` | Both partitions' original class counts, rows, tokens, shares, deviations, and branch-length statistics |
| `summary.json` | Source fingerprint, tokenizer identity, balance settings, and full-precision partition statistics |

`tokens_total` counts canonical A+B inputs once; `tokens_augmented` counts both
orientations. `effective_rows` and `effective_tokens` apply the model's training
orientation multiplier to the training partition (`aug` = 2, `inf` = 1).
Validation uses original IDs with both probability orientations averaged.
These volumes exclude epoch repetition, padding, warmup, and feature-cache reuse.
Over-limit branches are reported without truncating or rejecting the statistics.

## Migration from five folds

Replace `cross_validation` with `validation`, remove `n_splits`, and use
`validation_fraction: 0.2` and `max_token_deviation: 0.005`. Set the new processed
path and rerun preprocessing. The root preset and active experiment presets
already use this format. Legacy configurations and artifacts report a migration
error rather than being interpreted as a holdout. Existing five-fold data,
token statistics, results, and archived configurations are retained.

Continue with [model training](models.md). See [structure.md](../structure.md)
for module responsibilities and invariants.
