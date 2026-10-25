"""Selectable frozen and LoRA comparison model profiles and pipelines."""

from dataclasses import dataclass

from .common import ModelPipeline, ModelPipelineInterface
from .profile import ModelProfile
from .qwen3_4b import PROFILE as QWEN3_4B_PROFILE
from .qwen3_4b_lora import PROFILE as QWEN3_4B_LORA_PROFILE
from .qwen3_4b_lora import LoRAModelPipeline
from .qwen3_4b_lora_EXPT import PROFILE as QWEN3_4B_LORA_EXPT_PROFILE


@dataclass(frozen=True, slots=True)
class ModelRegistration:
	profile: ModelProfile
	pipeline_type: type[ModelPipelineInterface]


MODEL_REGISTRY = {
	registration.profile.selector: registration
	for registration in (
		ModelRegistration(QWEN3_4B_PROFILE, ModelPipeline),
		ModelRegistration(QWEN3_4B_LORA_PROFILE, LoRAModelPipeline),
		ModelRegistration(QWEN3_4B_LORA_EXPT_PROFILE, LoRAModelPipeline),
	)
}

__all__ = (
	"MODEL_REGISTRY",
	"QWEN3_4B_LORA_EXPT_PROFILE",
	"QWEN3_4B_LORA_PROFILE",
	"QWEN3_4B_PROFILE",
	"ModelProfile",
	"ModelRegistration",
)
