"""Download the raw Kaggle competition files."""

from __future__ import annotations

import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import kagglehub

from ..config import DataConfig
from ..errors import DownloadError


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Paths and actions produced while ensuring raw data availability."""

    file_paths: dict[str, Path]
    downloaded: tuple[str, ...]
    skipped: tuple[str, ...]


class CompetitionDataDownloader:
    """Ensure official competition files exist without replacing raw data."""

    REQUIRED_FILES = ("train.csv", "test.csv", "sample_submission.csv")

    def __init__(self, config: DataConfig) -> None:
        self._config = config

    def ensure_available(self) -> DownloadResult:
        """Download missing competition files."""
        self._prepare_raw_directory()

        downloaded: list[str] = []
        skipped: list[str] = []
        file_paths: dict[str, Path] = {}

        for file_name in self.REQUIRED_FILES:
            destination = self._config.raw_dir / file_name
            if destination.exists():
                if not destination.is_file():
                    raise DownloadError(
                        f"Raw data path exists but is not a file: {destination}"
                    )
                skipped.append(file_name)
            else:
                self._download_file(file_name, destination)
                downloaded.append(file_name)

            self._extract_single_file_archive(destination)
            file_paths[file_name] = destination

        return DownloadResult(
            file_paths=file_paths,
            downloaded=tuple(downloaded),
            skipped=tuple(skipped),
        )

    def _prepare_raw_directory(self) -> None:
        raw_dir = self._config.raw_dir
        if raw_dir.exists() and not raw_dir.is_dir():
            raise DownloadError(f"data.raw_dir is not a directory: {raw_dir}")
        try:
            raw_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise DownloadError(
                f"Could not create raw data directory {raw_dir}: {error}"
            ) from error

    def _download_file(self, file_name: str, destination: Path) -> None:
        try:
            downloaded_path = Path(
                kagglehub.competition_download(
                    self._config.competition,
                    path=file_name,
                    output_dir=str(self._config.raw_dir),
                )
            ).resolve()
        except Exception as error:
            raise DownloadError(
                f"Failed to download {file_name} from Kaggle competition "
                f"'{self._config.competition}'. Confirm that Kaggle credentials are "
                "configured, the competition rules have been accepted, and the "
                f"network is available. Kaggle reported: {error}"
            ) from error

        if not destination.is_file():
            raise DownloadError(
                f"Kaggle reported a successful download to {downloaded_path}, but "
                f"the expected file is unavailable at {destination}."
            )

    @staticmethod
    def _extract_single_file_archive(destination: Path) -> None:
        """Normalize a ZIP-wrapped Kaggle file to the requested CSV path."""
        if not zipfile.is_zipfile(destination):
            return

        temporary_path: Path | None = None
        try:
            with zipfile.ZipFile(destination) as archive:
                members = [
                    member for member in archive.infolist() if not member.is_dir()
                ]
                archived_name = Path(members[0].filename).name if members else None
                if len(members) != 1 or archived_name != destination.name:
                    raise DownloadError(
                        f"Unexpected archive contents downloaded for {destination.name}."
                    )

                with tempfile.NamedTemporaryFile(
                    dir=destination.parent,
                    prefix=f".{destination.name}-",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    with archive.open(members[0]) as archived_file:
                        while chunk := archived_file.read(1024 * 1024):
                            temporary_file.write(chunk)

            os.replace(temporary_path, destination)
        except DownloadError:
            raise
        except (OSError, zipfile.BadZipFile) as error:
            raise DownloadError(
                f"Could not extract downloaded archive {destination}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
