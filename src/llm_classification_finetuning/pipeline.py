"""Orchestrate data download, holdout preparation, and A/B augmentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .config import AppConfig
from .data import CompetitionDataDownloader, DownloadResult
from .preprocess import SplitPreparationResult, SplitPreprocessor


@dataclass(frozen=True, slots=True)
class PrepareDataResult:
	download: DownloadResult
	splits: SplitPreparationResult


class PrepareDataPipeline:
	"""Run the complete raw-data-to-training-artifact workflow."""

	def __init__(self, config: AppConfig) -> None:
		self._config = config
		self._downloader = CompetitionDataDownloader(config.data)
		self._preprocessor = SplitPreprocessor(
			config.validation, cache_dir=config.cache_dir
		)

	def run(
		self, on_download: Callable[[DownloadResult], None] | None = None
	) -> PrepareDataResult:
		"""Ensure raw inputs and preprocess them."""
		download_result = self._downloader.ensure_available()
		if on_download is not None:
			on_download(download_result)
		split_result = self._preprocessor.prepare(
			train_path=download_result.file_paths["train.csv"],
			output_path=self._config.data.processed_path,
		)
		return PrepareDataResult(download=download_result, splits=split_result)
