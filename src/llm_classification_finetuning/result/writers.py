"""Write comprehensive result artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter


class ConfusionMatrixWriter:
    """Render a labeled confusion matrix as a PNG heatmap."""

    CLASS_NAMES = ("0 (model A)", "1 (model B)", "2 (tie)")

    def write(self, matrix: np.ndarray, output_path: Path) -> None:
        """Write raw matrix counts to a compact annotated heatmap."""
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(output_path.parent / ".matplotlib"),
        )
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(6.4, 5.4), dpi=150)
        canvas = FigureCanvasAgg(figure)
        axes = figure.subplots()
        image = axes.imshow(matrix, interpolation="nearest", cmap="Blues")
        figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)

        axes.set(
            title="Cross-validation confusion matrix",
            xlabel="Actual label",
            ylabel="Expected label",
            xticks=np.arange(3),
            yticks=np.arange(3),
            xticklabels=self.CLASS_NAMES,
            yticklabels=self.CLASS_NAMES,
        )

        threshold = float(matrix.max()) / 2 if matrix.size else 0.0
        for expected_label in range(3):
            for actual_label in range(3):
                count = int(matrix[expected_label, actual_label])
                axes.text(
                    actual_label,
                    expected_label,
                    f"{count:,}",
                    ha="center",
                    va="center",
                    color="white" if count > threshold else "#172033",
                    fontweight="bold",
                )

        figure.tight_layout()
        canvas.print_png(output_path)
        figure.clear()


class RecordsWorkbookWriter:
    """Write reduced validation records to a filterable Excel workbook."""

    HEADERS = (
        "id",
        "fold",
        "expected_labels",
        "actual_labels",
        "incorrects",
        "contributed_loss",
    )

    def write(self, records: pd.DataFrame, output_path: Path) -> None:
        """Write records and a filter-aware average-loss row."""
        workbook = xlsxwriter.Workbook(str(output_path))
        try:
            workbook.set_calc_mode("auto")
            worksheet = workbook.add_worksheet("Records")
            worksheet.hide_gridlines(2)
            worksheet.freeze_panes(1, 0)
            worksheet.set_tab_color("#2563EB")

            integer_format = workbook.add_format({"num_format": "#,##0"})
            centered_format = workbook.add_format({"align": "center"})
            loss_format = workbook.add_format({"num_format": "0.000000"})
            incorrect_format = workbook.add_format(
                {"bg_color": "#FEE2E2", "font_color": "#991B1B"}
            )
            average_label_format = workbook.add_format(
                {
                    "bold": True,
                    "top": 2,
                    "top_color": "#2563EB",
                    "font_color": "#1E3A8A",
                }
            )
            average_value_format = workbook.add_format(
                {
                    "bold": True,
                    "top": 2,
                    "top_color": "#2563EB",
                    "num_format": "0.000000",
                    "font_color": "#1E3A8A",
                }
            )

            worksheet.set_column("A:A", 14, integer_format)
            worksheet.set_column("B:B", 9, centered_format)
            worksheet.set_column("C:C", 18, centered_format)
            worksheet.set_column("D:D", 18, centered_format)
            worksheet.set_column("E:E", 12, centered_format)
            worksheet.set_column("F:F", 20, loss_format)
            worksheet.set_row(0, 22)

            worksheet.write_column(1, 0, [int(value) for value in records["id"]])
            worksheet.write_column(1, 1, [int(value) for value in records["fold"]])
            worksheet.write_column(
                1,
                2,
                [int(value) for value in records["expected_labels"]],
            )
            worksheet.write_column(1, 3, records["actual_labels"].tolist())
            worksheet.write_column(
                1,
                4,
                [int(value) for value in records["incorrects"]],
            )
            worksheet.write_column(
                1,
                5,
                [float(value) for value in records["contributed_loss"]],
            )

            last_data_row = len(records)
            worksheet.add_table(
                0,
                0,
                last_data_row,
                len(self.HEADERS) - 1,
                {
                    "name": "RecordsTable",
                    "style": "Table Style Medium 2",
                    "columns": [{"header": header} for header in self.HEADERS],
                },
            )
            worksheet.conditional_format(
                1,
                0,
                last_data_row,
                len(self.HEADERS) - 1,
                {
                    "type": "formula",
                    "criteria": "=$E2>0",
                    "format": incorrect_format,
                },
            )

            average_row = last_data_row + 1
            last_excel_row = last_data_row + 1
            formula = f'=IFERROR(SUBTOTAL(101,F2:F{last_excel_row}),"")'
            worksheet.write(average_row, 0, "AVG", average_label_format)
            for column in range(1, 5):
                worksheet.write_blank(average_row, column, None, average_label_format)
            worksheet.write_formula(
                average_row,
                5,
                formula,
                average_value_format,
                float(records["contributed_loss"].mean()),
            )
            worksheet.set_row(average_row, 20)
        finally:
            workbook.close()
