"""Domain exceptions for data preparation."""


class DataPreparationError(Exception):
    """Base exception for expected data-preparation failures."""


class ConfigurationError(DataPreparationError):
    """Raised when the application configuration is invalid."""


class DownloadError(DataPreparationError):
    """Raised when competition data cannot be obtained safely."""


class ResultSynthesisError(DataPreparationError):
    """Raised when fold predictions cannot be synthesized safely."""


class ModelExecutionError(DataPreparationError):
    """Raised when model embedding extraction or training cannot complete."""
