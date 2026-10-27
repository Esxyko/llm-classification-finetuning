"""Skywork-Reward-V2-Qwen3-4B LoRA profile and training interfaces."""

from importlib import import_module

from .batching import (
	LoRATrainingBatchRepository,
	PreparedTrainingSplit,
	TrainingBatchPlan,
)
from .profile import PROFILE
from .warmup_cache import WarmupCacheContext


def __getattr__(name: str):
	"""Keep CPU batch preparation independent of PEFT's CUDA-aware imports."""
	modules = {"LoRAModelPipeline": ".pipeline", "LoRATrainer": ".trainer"}
	if name not in modules:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	value = getattr(import_module(modules[name], __name__), name)
	globals()[name] = value
	return value


__all__ = (
	"PROFILE",
	"LoRAModelPipeline",
	"LoRATrainer",
	"LoRATrainingBatchRepository",
	"PreparedTrainingSplit",
	"TrainingBatchPlan",
	"WarmupCacheContext",
)
