"""Shared command-line summaries for data preparation commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data.downloader import DownloadResult
    from .preprocess.folds import FoldPreparationResult


def print_download_result(result: DownloadResult) -> None:
    for file_name in result.skipped:
        print(f"Skipped existing raw file: {file_name}")
    for file_name in result.downloaded:
        print(f"Downloaded raw file: {file_name}")


def print_fold_result(result: FoldPreparationResult) -> None:
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
