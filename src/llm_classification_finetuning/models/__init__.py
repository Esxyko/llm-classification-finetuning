"""Active Skywork LoRA comparison model registration."""

from dataclasses import dataclass

from .common import ModelPipelineInterface
from .profile import ModelProfile
from .SR2_Qwen3_4B_lora import PROFILE as SR2_QWEN3_4B_LORA_PROFILE
from .SR2_Qwen3_4B_lora import LoRAModelPipeline


@dataclass(frozen=True, slots=True)
class ModelRegistration:
	profile: ModelProfile
	pipeline_type: type[ModelPipelineInterface]


MODEL_REGISTRY = {
	registration.profile.selector: registration
	for registration in (
		ModelRegistration(SR2_QWEN3_4B_LORA_PROFILE, LoRAModelPipeline),
	)
}

__all__ = (
	"MODEL_REGISTRY",
	"SR2_QWEN3_4B_LORA_PROFILE",
	"ModelProfile",
	"ModelRegistration",
)
