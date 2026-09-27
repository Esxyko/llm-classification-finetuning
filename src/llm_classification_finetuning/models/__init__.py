"""Selectable frozen-backbone comparison model profiles and pipelines."""

from dataclasses import dataclass

from .common import ModelPipeline
from .profile import ModelProfile
from .qwen3_1_7b import PROFILE as QWEN3_1_7B_PROFILE
from .qwen3_1_7b_mono_input import PROFILE as QWEN3_1_7B_MONO_INPUT_PROFILE
from .qwen3_1_7b_mono_input import MonoInputModelPipeline
from .qwen3_4b import PROFILE as QWEN3_4B_PROFILE


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    profile: ModelProfile
    pipeline_type: type[ModelPipeline] | type[MonoInputModelPipeline]


MODEL_REGISTRY = {
    registration.profile.selector: registration
    for registration in (
        ModelRegistration(QWEN3_1_7B_PROFILE, ModelPipeline),
        ModelRegistration(QWEN3_1_7B_MONO_INPUT_PROFILE, MonoInputModelPipeline),
        ModelRegistration(QWEN3_4B_PROFILE, ModelPipeline),
    )
}

__all__ = (
    "MODEL_REGISTRY",
    "ModelRegistration",
    "QWEN3_1_7B_MONO_INPUT_PROFILE",
    "QWEN3_1_7B_PROFILE",
    "QWEN3_4B_PROFILE",
    "ModelProfile",
)
