# Prepare training data

[Documentation index](../README.md)

After [setup](getting-started.md), choose the operation you need:

| Command / input | Behavior |
| --- | --- |
| Saved processed manifest and all volumes from this checkout | Start model validation directly; no raw download or preprocessing required |
| `uv run init` | Download missing competition files, then rebuild processed training data |
| `uv run data` | Download missing `train.csv`, `test.csv`, and `sample_submission.csv` only |
| `uv run preprocess` | Rebuild processed data from local `train.csv` |

All commands accept `--config PATH`. Downloads skip existing files and normalize
single-file ZIP wrappers. Preprocessing needs a tokenizer and may download its
files, but loads no model weights and needs no GPU. Use the same saved split
when comparing model experiments.

## One shared train/validation split

Preparation starts from the complete raw training table, derives labels and
canonical prompt hashes, and assigns whole prompt groups to train or validation.
The root preset uses validation fraction `0.2` and seed `42`. Classes retain their
original proportions; they are not resampled to equal sizes.

Original-class quotas use the nearest total validation row count and largest
fractional remainders. Multirow prompt groups are selected deterministically,
then singleton groups fill the remaining class quotas. Infeasible quotas fail
rather than splitting a prompt group.

Token counting uses the training conversation serializer, configured tokenizer,
special tokens, and shared [text cache](caching.md#shared-token-encodings), without
padding or truncation. Whole groups with identical three-class count vectors
are exchanged to meet the validation token target. This preserves initial row
and class counts exactly; at most 1,000 improving exchanges are allowed.

Both row and token shares are measured **before augmentation**. For validation
fraction `f`, relative deviation is `abs(validation_share / f - 1)`:

| Quantity | Required strict upper bound |
| --- | --- |
| Validation canonical row deviation | `0.005` |
| Validation canonical token deviation | `validation.max_token_deviation`, default `0.005` |

At `f = 0.2`, this means less than 0.5% relative error around a 20% target, not
0.5 percentage points. The summary also reports each partition's deviation from
its own target. The tokenizer/revision used for balancing is independently
configured under [validation](configuration.md#data-and-validation).

One A/B-swapped counterpart is appended to each original after splitting.
Only after balance and integrity checks succeed is the compressed Parquet
published. Failure leaves the previous dataset in place.

## Processed data

The default logical path is `data/processed/train_validation.parquet`. Keep
`data.processed_path` pointing there whether it is a single file or multipart:

- Below 100 MiB: the Zstandard-compressed `.parquet` file.
- At least 100 MiB: `.parquet.manifest.json` plus `.parquet.parts/`, each volume
  at most 99 MiB.

Loaders reconstruct and checksum the artifact automatically in an ignored
`.cache-work/` directory. The data fingerprint hashes reconstructed Parquet
bytes, independent of storage layout. Keep the manifest and all referenced
volumes together. Corrupt multipart data is archived and requires preprocessing
again. See [shared storage](caching.md#shared-storage-and-multipart-files) for
locking, publication, and temporary disk-space requirements.

Source columns are retained, with these additions:

| Column | Meaning |
| --- | --- |
| `label` | `0` = A wins, `1` = B wins, `2` = tie |
| `group_id` | SHA-256 of canonical prompt JSON |
| `split` | `train` or `validation` |
| `is_swapped` | Whether this is the swapped counterpart |

Each ID has one original and one swapped row in the same group/partition.
A/B fields and labels `0`/`1` swap; ties remain `2`. Originals precede swapped
rows in source order. No original ID is dropped. Input loading rejects missing
values, invalid labels, incomplete orientation pairs, swapped-label mismatches,
and prompt groups crossing partitions.

### Conversation serialization

All active profiles use `aligned-prompt-newline-response-v1`: chronological
`prompt\nresponse` turns joined by blank lines, without a chat template.
Prompt and response values must be JSON arrays with the same nonzero turn count.
Prompts must be strings; responses may be strings or `null` (an empty response).

Frozen extraction may right-truncate to its context limit. LoRA rejects empty
or over-limit branches instead; see [input limits](lora.md#hardware-and-input-limits).

## Migration from five folds

Older fold configurations and datasets are not interpreted as a holdout.
See [migration](migration.md#five-folds-to-one-holdout) before regenerating data
or reporting historical runs. Continue with [model workflows](models.md).
