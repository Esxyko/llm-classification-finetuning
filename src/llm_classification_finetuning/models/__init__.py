"""Active frozen and LoRA Skywork comparison model registration."""

from dataclasses import dataclass

from .common import ModelPipelineInterface
from .profile import ModelProfile
from .SR2_Qwen3_4B import PROFILE as SR2_QWEN3_4B_PROFILE
from .SR2_Qwen3_4B_lora import PROFILE as SR2_QWEN3_4B_LORA_PROFILE


@dataclass(frozen=True, slots=True)
class ModelRegistration:
	profile: ModelProfile
	pipeline_type: type[ModelPipelineInterface]


def __getattr__(name: str):
	"""Load execution classes only when the registry is requested, keeping data CPU-only."""
	if name not in {"MODEL_REGISTRY", "LoRAModelPipeline", "SR2ModelPipeline"}:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	from .SR2_Qwen3_4B import SR2ModelPipeline
	from .SR2_Qwen3_4B_lora import LoRAModelPipeline

	registry = {
		SR2_QWEN3_4B_PROFILE.selector: ModelRegistration(
			SR2_QWEN3_4B_PROFILE, SR2ModelPipeline
		),
		SR2_QWEN3_4B_LORA_PROFILE.selector: ModelRegistration(
			SR2_QWEN3_4B_LORA_PROFILE, LoRAModelPipeline
		),
	}
	globals().update(
		MODEL_REGISTRY=registry,
		LoRAModelPipeline=LoRAModelPipeline,
		SR2ModelPipeline=SR2ModelPipeline,
	)
	return globals()[name]


__all__ = (
	"MODEL_REGISTRY",
	"SR2_QWEN3_4B_LORA_PROFILE",
	"SR2_QWEN3_4B_PROFILE",
	"ModelProfile",
	"ModelRegistration",
)
