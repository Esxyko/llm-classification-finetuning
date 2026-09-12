"""Orchestrate data download, fold preparation, and A/B augmentation."""

from __future__ import annotations

from .config import AppConfig
from .data import CompetitionDataDownloader
from .preprocess import FoldPreparationResult, FoldPreprocessor


class PrepareDataPipeline:
    """Run the complete raw-data-to-training-artifact workflow."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._downloader = CompetitionDataDownloader(config.data)
        self._preprocessor = FoldPreprocessor(config.cross_validation)

    def run(self) -> FoldPreparationResult:
        """Ensure raw inputs, preprocess them, and report their distribution."""
        download_result = self._downloader.ensure_available()

        for file_name in download_result.skipped:
            print(f"Skipped existing raw file: {file_name}")
        for file_name in download_result.downloaded:
            print(f"Downloaded raw file: {file_name}")

        result = self._preprocessor.prepare(
            train_path=download_result.file_paths["train.csv"],
            output_path=self._config.data.processed_path,
        )
        self._print_summary(result)
        return result

    @staticmethod
    def _print_summary(result: FoldPreparationResult) -> None:
        print(f"Prepared {result.rows:,} rows across {result.groups:,} prompt groups.")
        print("Fold distribution:")
        print("fold  rows     groups   model_a   model_b   tie")
        for summary in result.folds:
            class_percentages = tuple(
                count / summary.rows * 100 for count in summary.class_counts
            )
            print(
                f"{summary.fold:>4}  {summary.rows:>8,}  {summary.groups:>7,}  "
                f"{class_percentages[0]:>7.2f}%  "
                f"{class_percentages[1]:>7.2f}%  "
                f"{class_percentages[2]:>6.2f}%"
            )
        print(f"Wrote processed data: {result.output_path}")
