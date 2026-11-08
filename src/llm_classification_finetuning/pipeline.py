"""Orchestrate prebuilt data availability and raw-data preparation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .data import CompetitionDataDownloader, DownloadResult
from .errors import DataPreparationError
from .models.common import CacheStorage
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


class EnsureProcessedDataPipeline:
	"""Reuse local processed data or fill missing files from a project mirror."""

	def __init__(
		self,
		config: AppConfig,
		*,
		project_root: Path,
		source_root: Path | None = None,
	) -> None:
		self._storage = CacheStorage(config.data.processed_path)
		self._project_root = project_root.resolve()
		# Resolve the source only after a local miss, avoiding access to Drive on hits.
		self._source_root = source_root

	def run(
		self, on_copy: Callable[[Path, Path], None] | None = None
	) -> tuple[Path, ...]:
		"""Ensure the selected artifact is complete and return copied local paths."""
		source_path: Path | None = None
		try:
			missing = self._storage.missing_components()
			if not missing:
				return ()
			missing_paths = ", ".join(str(path) for path in missing)
			if self._source_root is None:
				raise DataPreparationError(
					f"Missing or wrong-sized processed data: {missing_paths}. "
					f"Expected {self._storage.path} or {self._storage.manifest_path}. "
					"No --data-source-root configured."
				)
			try:
				relative_path = self._storage.path.relative_to(self._project_root)
			except ValueError as error:
				raise DataPreparationError(
					f"Missing or wrong-sized processed data: {missing_paths}. "
					f"Cannot map {self._storage.path} to the source root: "
					f"it is outside the project root {self._project_root}."
				) from error
			source_root = self._source_root.expanduser()
			if not source_root.is_absolute():
				source_root = self._project_root / source_root
			source_path = source_root / relative_path
			return self._storage.copy_missing_from(
				CacheStorage(source_path), on_copy=on_copy
			)
		except OSError as error:
			source_detail = f" from {source_path}" if source_path is not None else ""
			raise DataPreparationError(
				f"Could not prepare processed data {self._storage.path}{source_detail}: {error}"
			) from error
