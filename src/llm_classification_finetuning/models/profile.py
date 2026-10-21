"""Immutable identities for selectable model implementations."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig, LoRAModelConfig, ModelConfig
from ..errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ModelProfile:
	"""Connect one CLI selector to configuration and artifact namespaces."""

	selector: str
	config_key: str
	artifact_stem: str
	result_prefix: str
	quantize_training: bool = False

	def is_configured(self, config: AppConfig) -> bool:
		"""Return whether the selected YAML includes this model's section."""
		return getattr(config, self.config_key, None) is not None

	def resolve_config(self, config: AppConfig) -> ModelConfig | LoRAModelConfig:
		"""Return this profile's validated model configuration."""
		value = getattr(config, self.config_key)
		if value is None:
			raise ConfigurationError(
				f"Model {self.selector} requires a '{self.config_key}' section in "
				"the selected configuration. Select a matching YAML with --config."
			)
		if not isinstance(value, (ModelConfig, LoRAModelConfig)):
			raise TypeError(f"{self.config_key} is not a model configuration.")
		return value
