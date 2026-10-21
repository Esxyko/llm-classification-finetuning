# Train with QLoRA

[Documentation index](../README.md)

Prepare the [training data](data-preparation.md), then select the independent
`qwen3_4b_qlora` section in the shared configuration:

```shell
uv run model qwen3-4b-qlora
uv run model qwen3-4b-qlora --build
uv run model qwen3-4b-qlora --submit
```

The root configuration and active presets copy the corresponding LoRA
hyperparameters into this section. The sections are independent: changing one
does not change the other. QLoRA supports `--config`, `--refresh-cache`, and
`--checkpoint-tag TAG` with the same meanings as [LoRA](lora.md).

## Quantized training and validation

QLoRA loads the base directly on `gpu.device` with bitsandbytes 4-bit NF4 and
double quantization. These settings are fixed by the model profile and are not
YAML options. `gpu.precision` controls the quantized layers' computation dtype
(`bf16`, `fp16`, or `fp32`), rather than their 4-bit weight storage. The project
dependencies include `bitsandbytes>=0.50.2`; install the locked environment with
`uv sync`. Its CUDA backend must support the selected training GPU and runtime.
The loading and preparation sequence follows the official
[Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
and [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
guides.

Before adapter injection, PEFT prepares and freezes the quantized base. Adapters
and the pairwise classifier retain FP32 parameters, gradients, and AdamW moment
tensors. With `gpu.precision: bf16`, the project restores the frozen input
embedding and every Qwen3 RMSNorm weight to BF16 after PEFT preparation and
before adapter injection. This includes attention Q/K norms, both decoder-layer
norms, and the final norm. RMSNorm retains FP32 variance calculation and returns
BF16 outputs, keeping residual states and activation-checkpoint layer inputs in
BF16. The NF4 weights and double-quantization state are unchanged. FP16/FP32
preparation retains its existing behavior. Gradient checkpointing
keeps the existing non-reentrant implementation and fused SDPA context during
both forward execution and backward recomputation.

QLoRA reuses the [LoRA training workflow](lora.md): head-only warmup, joint
adapter/head training, AdamW, linear decay, gradient accumulation, token-budget
batch planning, and `A/B_swap: aug` or `inf`. On a warmup cache miss, warmup
extraction uses the quantized base with adapter forward computation disabled;
a compatible LoRA cache skips that stage. Warmup epoch timing excludes this
preparation. Extraction batches count individual A/B branches, as in the frozen
model. Every joint-validation pass uses the current quantized base and trained
adapters. Inputs are independently
right-padded without padding masks and are never truncated. Batch sizes,
sequence limits, epochs, and learning rates are not automatically changed.

## Non-quantized submissions

`--submit` explicitly disables quantization and reloads the checkpoint's exact
Hub base revision at `gpu.precision`. It restores the adapter and classifier
onto that non-quantized base, using the existing
[two-GPU layer placement](configuration.md#gpu-configuration). No quantized-base
dequantization or adapter merge is performed. Gradient checkpointing is disabled.

Submission therefore has a different base-weight representation from validation;
its predictions can differ, and its base-weight memory use is higher. Both
submission GPUs must support the selected precision. The classifier remains
FP32 in parameter storage and uses BF16 autocast when `gpu.precision: bf16`.
Pooled and temporary CPU features are then BF16; logits and probability
averaging remain FP32, as in LoRA. FP16/FP32 submissions retain FP32 head
computation and pooled/CPU features.

## Checkpoints, caches, and results

Builds atomically save `models/qwen3_4b_qlora/adapter.pt`, or
`adapter_TAG.pt` with `--checkpoint-tag TAG`. The file contains adapters, the
classifier, base revision, and compatibility metadata, without base weights,
quantization-state tensors, or optimizer state. The `training_quantization`
metadata value is `bnb-nf4-double`; LoRA uses `none`. Loading rejects a checkpoint
from the other training profile, including when files have been copied between
directories. Older LoRA checkpoints without this field are treated as `none`.
Other compatibility rules follow [LoRA checkpoints](lora.md#checkpoints).

QLoRA shares content-addressed text tokens with the other profiles and shares
`adapter_batches/plans/` with LoRA beside the processed Parquet. Batch plans are
reused when selected IDs/orientations, branch lengths, budget, update-group size,
and algorithm version match, independently of training quantization. For
completed head warmup and its training state, QLoRA first checks the
non-quantized `qwen3_4b_lora_warmup/` cache, including entries created with
`model warmup --build`. It then checks `qwen3_4b_qlora_warmup/`; if neither
matches, it trains and saves its own quantized warmup. The shared
[warmup reuse rules](caching.md#loraqlora-head-warmup) apply, including
cross-precision reuse. `--refresh-cache` skips both sources and rebuilds
QLoRA warmup. Extracted features remain temporary and are never
reused across the quantized training and non-quantized submission stages.

Validation results use `results/qwen3-4b-qlora-*/`; submissions use
`results/submit/qwen3-4b-qlora-*/submission.csv`. Existing report commands consume
the validation output without a special QLoRA format.

`model ALL` includes QLoRA when its configuration section is present. The
[batch scripts](configuration.md#batch-experiments) continue to select only LoRA
by default; add `qwen3-4b-qlora` to their `MODELS` tuples to select QLoRA.
