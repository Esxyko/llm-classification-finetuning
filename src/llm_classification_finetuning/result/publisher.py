"""Publish a complete pair of comprehensive report artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import ResultSynthesisError
from .writers import ConfusionMatrixWriter, RecordsWorkbookWriter


class ResultPublisher:
    """Replace the current report while retaining a recoverable previous copy."""

    OUTPUT_DIR_NAME = "comprehensive"
    WORKBOOK_NAME = "records.xlsx"
    CONFUSION_MATRIX_NAME = "confusion_matrix.png"
    WINDOWS_RENAME_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8)

    def __init__(self, results_root: Path) -> None:
        self._results_root = results_root.resolve()
        self._matrix_writer = ConfusionMatrixWriter()
        self._workbook_writer = RecordsWorkbookWriter()

    def publish(
        self, records: pd.DataFrame, confusion_matrix: np.ndarray
    ) -> tuple[Path, Path]:
        output_dir = self._results_root / self.OUTPUT_DIR_NAME
        backup_dir = self._results_root / f".{self.OUTPUT_DIR_NAME}-backup"
        if output_dir.is_symlink() or backup_dir.is_symlink():
            raise ResultSynthesisError(
                "Comprehensive output and backup cannot be symlinks."
            )
        if output_dir.exists() and not output_dir.is_dir():
            raise ResultSynthesisError(
                f"Comprehensive output path is not a directory: {output_dir}"
            )
        if backup_dir.exists() and not backup_dir.is_dir():
            raise ResultSynthesisError(
                f"Comprehensive backup path is not a directory: {backup_dir}"
            )
        if backup_dir.exists():
            if output_dir.exists():
                raise ResultSynthesisError(
                    f"A previous comprehensive report backup needs review: {backup_dir}"
                )
            try:
                self._replace_directory(backup_dir, output_dir)
            except OSError as error:
                raise ResultSynthesisError(
                    f"Could not restore the previous report from {backup_dir}: {error}"
                ) from error

        try:
            with tempfile.TemporaryDirectory(
                dir=self._results_root,
                prefix=f".{self.OUTPUT_DIR_NAME}-stage-",
            ) as temporary_dir:
                staging_dir = Path(temporary_dir)
                try:
                    self._workbook_writer.write(
                        records, staging_dir / self.WORKBOOK_NAME
                    )
                    self._matrix_writer.write(
                        confusion_matrix, staging_dir / self.CONFUSION_MATRIX_NAME
                    )
                except Exception as error:
                    raise ResultSynthesisError(
                        f"Could not generate comprehensive artifacts: {error}"
                    ) from error

                previous_moved = False
                try:
                    if output_dir.exists():
                        self._replace_directory(output_dir, backup_dir)
                        previous_moved = True
                    self._replace_directory(staging_dir, output_dir)
                except OSError as error:
                    if previous_moved:
                        try:
                            self._replace_directory(backup_dir, output_dir)
                        except OSError as restore_error:
                            raise ResultSynthesisError(
                                "Could not publish the new report or restore the previous "
                                f"one. Previous report backup: {backup_dir}. "
                                f"Publish error: {error}; restore error: {restore_error}"
                            ) from restore_error
                    raise ResultSynthesisError(
                        f"Could not publish comprehensive artifacts to {output_dir}: {error}"
                    ) from error

                if previous_moved:
                    try:
                        shutil.rmtree(backup_dir)
                    except OSError as error:
                        raise ResultSynthesisError(
                            f"New report is available at {output_dir}, but the previous "
                            f"report backup could not be removed from {backup_dir}: {error}"
                        ) from error
        except ResultSynthesisError:
            raise
        except OSError as error:
            raise ResultSynthesisError(
                f"Could not stage comprehensive artifacts under {self._results_root}: {error}"
            ) from error

        return output_dir / self.WORKBOOK_NAME, output_dir / self.CONFUSION_MATRIX_NAME

    @classmethod
    def _replace_directory(cls, source: Path, destination: Path) -> None:
        """Retry brief Windows sharing races while preserving rename semantics."""
        for delay in (*cls.WINDOWS_RENAME_RETRY_DELAYS, None):
            try:
                os.replace(source, destination)
                return
            except OSError as error:
                if os.name != "nt" or error.winerror not in (5, 32) or delay is None:
                    raise
                time.sleep(delay)
