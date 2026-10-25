# Troubleshooting

[Documentation index](../README.md)

- **Kaggle download fails:** Accept the competition rules, verify
  `KAGGLE_API_TOKEN`, and check network access.
- **Warmup or model download fails with `CAS Client Error` and `401 Unauthorized`:**
  The Xet client failed to authenticate while downloading model weights. Project
  commands now default to HTTP downloads when using `https://hf-mirror.com`.
  Restart the command with the updated code. If Xet was explicitly enabled,
  disable it before starting the command, for example in PowerShell:

  ```powershell
  $env:HF_HUB_DISABLE_XET = "1"
  uv run model warmup --build
  ```

  For notebooks, restart the kernel and set the variable before importing
  `transformers` or `huggingface_hub`. A `401` from the Hub itself can still
  require a valid `HF_TOKEN` and access to the configured repository. See
  [Hugging Face downloads](getting-started.md#hugging-face-downloads).
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
- **T4 submission fails with `No available kernel`:** Use the
  [submission environment requirements](getting-started.md#submission-environment-pytorch-211--cuda-128),
  `gpu.precision: fp16`, and `gpu.attention_implementation: sdpa`. The submission
  pipeline applies the K/V-head expansion workaround inside the inference
  process, including subprocesses. A patch applied only in a notebook does not
  affect a separate Python process. Math fallback remains disabled, so
  unsupported fused-kernel inputs still fail explicitly.
- **Embedding extraction runs out of memory:** Reduce
  `qwen3_4b.extraction_budget`, the padded-token limit per extraction micro-batch
  (active presets: `524288`). A branch above the budget still runs alone.
  Lowering `qwen3_4b.max_length` also reduces
  input length by truncating longer branches. Frozen checkpoints permit changes
  to `max_length`, precision, attention implementation, and TF32 without a
  rebuild. Precision changes select a different embedding cache; length changes
  reuse vectors only when the effective token IDs still match. Predictions can
  change with the new inputs or compute settings; see
  [checkpoint compatibility](models.md#checkpoint-compatibility).
- **Frozen head training or inference runs out of memory:** Reduce
  `qwen3_4b.training_batch_size`; the frozen profile uses it for head training,
  validation, and submission inference.
- **LoRA runs out of memory:** Lower `warmup.extraction_budget` for warmup
  backbone extraction or `warmup.warmup_training_batch_size` for warmup head
  training. Lower the selected model section's `extraction_budget` for
  joint-validation/submission extraction or `head_inference_batch_size` for head
  inference. For joint training, lower a positive
  `gradient_checkpointing_threshold` or lower `gradient_checkpointing_budget`
  (`0` disables singleton merging). With a zero threshold, lowering
  `training_batch_size` reduces initial micro-batches. With a positive threshold,
  micro-batch sizes follow token packing; see [memory tuning](lora.md#memory-tuning).
  `head_inference_batch_size` controls only head inference over the
  extracted CPU features. Keep
  `max_length` at least as large as the longest required branch. Gradient
  checkpointing is automatic for batches above the threshold; at threshold `0`,
  all joint-training batches use it. Batches at or below a positive threshold
  disable it. The budget only limits merging checkpointed singletons within
  an optimizer update, and an individual sample above it still runs alone.
  Use FP16/BF16 instead of FP32 where supported. Neither setting shortens
  an input, and checkpointing applies only to joint
  training. See [memory tuning](lora.md#memory-tuning).
- **LoRA input exceeds `max_length`:** Use the reported ID, branch, and actual
  token count to raise the limit before rerunning. Inputs are never silently
  truncated; lowering batch sizes or the token budget preserves their content.
- **A checkpoint is incompatible:** Match the model, serialization, and
  classifier architecture used for its build or rebuild with the current config.
  LoRA checkpoints also check classifier dropout, adapter settings, and
  training quantization; historical quantized checkpoints are incompatible. All
  checkpoint types allow changes to precision, attention implementation, TF32,
  and `max_length`. Frozen checkpoints also allow changes to `qwen3_4b.A/B_swap`.
- **Report rows or aggregation do not match:** Use the processed Parquet and
  saved split corresponding to that run, and keep its holdout `metrics.json`.
  Frozen `validation.csv` files contain both orientations; EXPT2 and SR2 contain
  only original validation rows. Preserve the expected order for that policy.
  Old five-fold data and results require migration; regenerate the split and run
  the model again. See [report requirements](reports.md).

For the complete setup, see [getting started](getting-started.md). For hardware
settings and memory tuning, see [GPU configuration](configuration.md#gpu-configuration)
and [LoRA memory tuning](lora.md#memory-tuning).
