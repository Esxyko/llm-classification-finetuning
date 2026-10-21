# Troubleshooting

[Documentation index](../README.md)

- **Kaggle download fails:** Accept the competition rules, verify
  `KAGGLE_API_TOKEN`, and check network access.
- **Configuration file is missing:** Run from the intended working directory or
  pass the correct file path with `--config`.
- **A configuration field is missing or unknown:** Each command validates the
  complete YAML, including present model sections it does not use. Files do not
  inherit values from the root preset; see
  [required fields and defaults](configuration.md#configuration-sections).
- **`train.csv` is missing:** Run `uv run data`, or use `uv run init` for both
  download and preprocessing.
- **CUDA or precision is unsupported:** Select an available `cuda:N` device and
  set it in `gpu.device` for training, or select two distinct available devices
  in `gpu.submit_devices` for submissions. Both submission GPUs must support the
  selected precision; use `fp16` or `fp32` if either does not support `bf16`.
- **Embedding extraction runs out of memory:** Reduce
  `qwen3_4b.extraction_batch_size`. Lowering `qwen3_4b.max_length` also reduces
  input length by truncating longer branches; a saved frozen head requires the
  same length and compute settings used at build time. Rebuild it if those
  compatibility fields change.
- **Frozen head training or inference runs out of memory:** Reduce
  `qwen3_4b.training_batch_size`; the frozen profile uses it for head training,
  validation, and submission inference.
- **LoRA or QLoRA runs out of memory:** Lower the selected model section's `warmup_extraction_batch_size`,
  `warmup_training_batch_size`, `training_batch_size`, or `head_inference_batch_size`
  for the affected stage, or lower `training_token_budget` when enabled.
  `warmup_extraction_batch_size` also controls validation/submission backbone
  extraction; `head_inference_batch_size` controls only head inference over the
  extracted CPU features. Keep
  `max_length` at least as large as the longest required branch. Enable
  `gradient_checkpointing` and use FP16/BF16 instead of FP32 where supported.
  Token budgets do not cap an over-budget singleton, and checkpointing applies
  only to training. See [memory tuning](lora.md#memory-tuning).
- **LoRA or QLoRA input exceeds `max_length`:** Use the reported ID, branch, and actual
  token count to raise the limit before rerunning. Inputs are never silently
  truncated; lowering batch sizes or the token budget preserves their content.
- **QLoRA quantized loading fails:** Run `uv sync` to install the locked
  bitsandbytes dependency and check that its CUDA backend supports your runtime
  and GPU. QLoRA trains on a 4-bit base but submits on a non-quantized base,
  so submission memory requirements and predictions can differ from validation.
- **A checkpoint is incompatible:** Match the model, serialization, and
  classifier architecture used for its build or rebuild with the current config.
  LoRA/QLoRA checkpoints also check classifier dropout, adapter settings, and
  training quantization; files from those profiles are not interchangeable. All
  checkpoint types allow changes to precision, attention implementation, TF32,
  `max_length`, and `A/B_swap`.
- **Report rows or aggregation do not match:** Use the processed Parquet and
  fold count corresponding to that run, preserve both orientations in each CSV,
  and keep `metrics.json` beside the CSVs. See [report requirements](reports.md).

For the complete setup, see [getting started](getting-started.md). For hardware
settings and memory tuning, see [GPU configuration](configuration.md#gpu-configuration)
and [LoRA memory tuning](lora.md#memory-tuning).
