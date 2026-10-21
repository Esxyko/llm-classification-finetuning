"""Shared command-line summaries for data preparation commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .data.downloader import DownloadResult
	from .preprocess.splits import SplitPreparationResult


def print_download_result(result: DownloadResult) -> None:
	for file_name in result.skipped:
		print(f"Skipped existing raw file: {file_name}")
	for file_name in result.downloaded:
		print(f"Downloaded raw file: {file_name}")


def print_split_result(result: SplitPreparationResult) -> None:
	print(f"Prepared {result.rows:,} rows across {result.groups:,} prompt groups.")
	print("Split distribution (A/B augmented):")
	print("split           rows     groups   model_a   model_b   tie")
	for summary in result.splits:
		class_percentages = tuple(
			count / summary.rows * 100 for count in summary.class_counts
		)
		print(
			f"{summary.split:<10}  {summary.rows:>8,}  {summary.groups:>7,}  "
			f"{class_percentages[0]:>7.2f}%  "
			f"{class_percentages[1]:>7.2f}%  "
			f"{class_percentages[2]:>6.2f}%"
		)
	print("Canonical inputs (before A/B augmentation):")
	print(
		"split       source_rows  model_a  model_b     tie     A+B tokens  row share  row deviation  token share  token deviation"
	)
	for summary in result.splits:
		print(
			f"{summary.split:<10}  {summary.source_rows:>11,}  "
			f"{summary.source_class_counts[0]:>7,}  "
			f"{summary.source_class_counts[1]:>7,}  "
			f"{summary.source_class_counts[2]:>6,}  "
			f"{summary.tokens_total:>13,}  {summary.row_share_pct:>8.4f}%  "
			f"{summary.row_deviation_pct:>+12.4f}%  {summary.token_share_pct:>10.4f}%  "
			f"{summary.token_deviation_pct:>+14.4f}%"
		)
	print(f"Wrote processed data: {result.output_path}")
