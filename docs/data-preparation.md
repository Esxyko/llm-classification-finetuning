# Prepare training data

[Documentation index](../README.md)

Complete [setup](getting-started.md), then run the combined pipeline:

```shell
uv run init
```

To download files only, use `uv run data`. To rebuild from local raw files, use
`uv run preprocess`. Paths and fold settings come from the
[`data` and `cross_validation` sections](configuration.md#configuration-sections).

The preparation pipeline:

1. Downloads `train.csv`, `test.csv`, and `sample_submission.csv` only when they
   are missing.
2. Keeps identical canonical prompts in the same stratified fold to prevent
   leakage.
3. Adds one A/B-swapped copy of every training row to reduce position bias.
4. Writes a Zstandard-compressed Parquet file.

The processed file preserves the source columns and adds:

| Column | Meaning |
| --- | --- |
| `label` | `0` = model A wins, `1` = model B wins, `2` = tie |
| `group_id` | SHA-256 hash of the canonical prompt JSON |
| `fold` | Zero-based validation-fold index |
| `is_swapped` | Whether the row is the generated A/B-swapped copy |

Original and swapped rows share an `id`, `group_id`, and fold, so duplicate IDs
are expected. Original rows appear first, followed by swapped rows in the same
source order. Re-running preparation skips downloads for existing files and
atomically replaces the processed file. A ZIP-wrapped raw file, including an
existing one, is unpacked in place if its only member is the requested CSV.

Both active models serialize each branch as chronological `prompt\nresponse`
turns joined by a blank line, without a chat template. `prompt`, `response_a`,
and `response_b` must be JSON arrays with the same nonzero turn count. Prompts
must be strings; response entries may be strings or `null`, with `null`
serialized as an empty response. Training requires exactly one original and one
swapped row per ID in the same fold, with labels translated back consistently.
These input checks happen when the model loads the processed data.

## Count tokens in each saved fold

From the project root, run:

```shell
uv run python scripts/count_fold_tokens.py --local-files-only
```

The script uses the selected model's tokenizer and the same conversation
serialization as training, preserving the saved folds. It counts special tokens
and excludes padding and truncation. The default model is `qwen3-4b-lora`;
`--model qwen3-4b`, `--config PATH`, `--revision REVISION`, and `--output-dir PATH`
are also supported. Omit `--local-files-only` to allow downloading tokenizer files.
No model weights are loaded.

Reports default to `data/processed/fold_token_stats/`: `per_id.csv` contains A/B
lengths, `folds.csv` contains fold totals and branch-length percentiles, and
`summary.json` records the source fingerprint and tokenizer revision. Each fold's
`tokens_total` counts its original A/B inputs once; `tokens_augmented` includes
both orientations. `train_tokens` counts the other folds using the configured
training orientations (`aug` doubles the count; `inf` does not). These are input
volumes for one pass, excluding epoch repetition, padding, warmup, validation,
and cache reuse. The frozen model trains its head on embeddings rather than
re-encoding these tokens each epoch. Over-limit branches are reported without
truncating or rejecting them.

Continue with [model training](models.md). Code ownership and the invariants
that preprocessing must preserve are documented in [structure.md](../structure.md).
