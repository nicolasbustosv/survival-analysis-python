"""Publication style and JCO palette (matches ggsci jco)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl

JCO_PALETTE = [
    "#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC",
    "#003C67", "#8F7700", "#3B3B3B", "#A73030", "#4A6990",
]


def set_publication_style() -> None:
    """Apply publication-ready matplotlib rcParams."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def jco_colors(n: int) -> list[str]:
    """Return n colors cycling through JCO palette."""
    return [JCO_PALETTE[i % len(JCO_PALETTE)] for i in range(n)]
