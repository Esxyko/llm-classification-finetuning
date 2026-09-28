"""Selectable frozen-backbone comparison model profiles and pipelines."""

from dataclasses import dataclass

from .common import ModelPipeline
from .profile import ModelProfile
from .qwen3_4b import PROFILE as QWEN3_4B_PROFILE


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    profile: ModelProfile
    pipeline_type: type[ModelPipeline]


MODEL_REGISTRY = {
    registration.profile.selector: registration
    for registration in (
        ModelRegistration(QWEN3_4B_PROFILE, ModelPipeline),
    )
}

__all__ = (
    "MODEL_REGISTRY",
    "ModelRegistration",
    "QWEN3_4B_PROFILE",
    "ModelProfile",
)
