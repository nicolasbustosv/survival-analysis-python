"""Save figures and write Excel tables."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def save_figure(
    fig: matplotlib.figure.Figure,
    basename: str | Path,
    formats: Iterable[str] = ("png", "svg"),
    dpi: int = 300,
) -> list[Path]:
    """Save figure in one or more formats. Returns list of written paths."""
    basename = Path(basename)
    basename.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        out = basename.with_suffix(f".{fmt}")
        try:
            fig.savefig(out, dpi=dpi, bbox_inches="tight", format=fmt)
            paths.append(out)
        except Exception as exc:
            warnings.warn(f"Could not save {out}: {exc}")
    plt.close(fig)
    return paths


def write_xlsx(tables: dict[str, pd.DataFrame], path: str | Path) -> None:
    """Write multiple DataFrames as sheets in one Excel workbook."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
