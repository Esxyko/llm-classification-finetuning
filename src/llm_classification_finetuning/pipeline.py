"""Orchestrate data download, fold preparation, and A/B augmentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .config import AppConfig
from .data import CompetitionDataDownloader, DownloadResult
from .preprocess import FoldPreparationResult, FoldPreprocessor


@dataclass(frozen=True, slots=True)
class PrepareDataResult:
    download: DownloadResult
    folds: FoldPreparationResult


class PrepareDataPipeline:
    """Run the complete raw-data-to-training-artifact workflow."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._downloader = CompetitionDataDownloader(config.data)
        self._preprocessor = FoldPreprocessor(config.cross_validation)

    def run(
        self, on_download: Callable[[DownloadResult], None] | None = None
    ) -> PrepareDataResult:
        """Ensure raw inputs and preprocess them."""
        download_result = self._downloader.ensure_available()
        if on_download is not None:
            on_download(download_result)
        fold_result = self._preprocessor.prepare(
            train_path=download_result.file_paths["train.csv"],
            output_path=self._config.data.processed_path,
        )
        return PrepareDataResult(download=download_result, folds=fold_result)
