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

Token counting uses the training conversation serializer, configured tokenizer's
chat template, and shared [text cache](caching.md#shared-token-encodings), without
padding or truncation. Template output is encoded with `add_special_tokens=False`
because it already contains the control tokens. Whole groups with identical
three-class count vectors are exchanged to meet the validation token target.
This preserves initial row and class counts exactly; at most 1,000 improving
exchanges are allowed.

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

SR2 uses `sr2-qwen3-chat-template-v1`. Each A/B branch becomes one complete list
of alternating `user` prompt and `assistant` response messages, rendered with
the configured tokenizer's `apply_chat_template(tokenize=False,
add_generation_prompt=False, continue_final_message=False)`. No system message
or tools are supplied. The complete output, including its trailing newline, is
encoded with `add_special_tokens=False`.

The [official SR2 template](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-4B/blob/main/chat_template.jinja)
uses ChatML role/control tokens and inserts a `<think>\n\n</think>\n\n` block before
the final assistant response when it has no reasoning content. This default
behavior is retained. Earlier assistant messages use the official template's
normal multi-turn handling. Missing or unusable templates fail explicitly.

Prompt and response values must be JSON arrays with the same nonzero turn count.
Prompts must be strings; responses may be strings or `null`. A null response is
an empty assistant message, retaining its position in the conversation. The
Parquet stores source arrays rather than materialized chat text; model
repositories render them with the same tokenizer used for encoding.

Current presets count tokens with `Skywork/Skywork-Reward-V2-Qwen3-4B` at revision
`fd958fef475f323f4e6b195930e3dd918485c668`. Switching from the old plain-text
serializer requires rebuilding the split to balance the new token counts; some
IDs can change partitions even when the fraction and random seed are unchanged.

SR2 rejects empty or over-limit branches and never truncates; see
[input limits](lora.md#hardware-and-input-limits). Historical frozen-model
behavior is recorded in [migration](migration.md#historical-frozen-model).

### Saved SR2 split (2026-10-27)

The default saved artifact was rebuilt from all 57,477 source IDs with the
official template above, validation fraction `0.2`, and seed `42`. It contains
114,954 augmented rows across 51,734 prompt groups, preserving source ID order.
Token balancing made nine whole-group exchanges. Compared with the previous
saved split, two original IDs moved to validation and two moved to training.

| Partition | Original rows | Augmented rows | Prompt groups | Canonical A+B tokens |
| --- | ---: | ---: | ---: | ---: |
| Train | 45,982 | 91,964 | 41,436 | 35,855,325 |
| Validation | 11,495 | 22,990 | 10,298 | 9,018,402 |

The validation row share is 19.999304% and token share is 20.097288%. Their
relative deviations from the 20% target are 0.003480% and 0.486438%, respectively,
both strictly below 0.5%. The reconstructed Parquet SHA-256 is
`2b06417150aea3d9320b0d40032feae154b6f072164fca09b05f58f64faf8f90`.
The previous artifact's SHA-256 was
`f2e9efb77dd2888609fb49c64b8b6884fd0deb05418e951151386667a64ab787`.
See [measured lengths](lora.md#measured-input-lengths) for the matching tokenizer
statistics and current preset ceiling.

## Migration from five folds

Older fold configurations and datasets are not interpreted as a holdout.
See [migration](migration.md#five-folds-to-one-holdout) before regenerating data
or reporting historical runs. Continue with [model workflows](models.md).
