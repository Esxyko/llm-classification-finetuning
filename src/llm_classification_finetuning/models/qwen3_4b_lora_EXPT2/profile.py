"""Active LoRA identity shared by training and standalone head warmup."""

from ..profile import ModelProfile

PROFILE = ModelProfile(
	selector="qwen3-4b-lora-EXPT2",
	config_key="qwen3_4b_lora_EXPT2",
	artifact_stem="qwen3_4b_lora_EXPT2",
	result_prefix="qwen3-4b-lora-EXPT2",
	head_architecture="swap_equivariant_tie_from_wins_v1",
	ab_swap_override="none",
)
