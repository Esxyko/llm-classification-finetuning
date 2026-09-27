"""Immutable identities for selectable model implementations."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig, ModelConfig


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Connect one CLI selector to configuration and artifact namespaces."""

    selector: str
    config_key: str
    artifact_stem: str
    result_prefix: str

    def resolve_config(self, config: AppConfig) -> ModelConfig:
        """Return this profile's validated model configuration."""
        value = getattr(config, self.config_key)
        if not isinstance(value, ModelConfig):
            raise TypeError(f"{self.config_key} is not a model configuration.")
        return value
