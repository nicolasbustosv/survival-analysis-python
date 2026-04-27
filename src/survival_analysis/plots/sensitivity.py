"""HR sensitivity panel — ported from HR_sensiblity.R."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from .style import set_publication_style


def load_hr_sensitivity(path: str) -> pd.DataFrame:
    """Read hr_sensibility.csv (semicolon-delimited, comma decimal mark)."""
    df = pd.read_csv(path, sep=";", decimal=",")
    period_renames = {
        "1825 - (1990-2010)": "5 years (1990-2010)",
        "1825 - (1990-2020)": "5 years (1990-2020)",
        "1825 - (2010-2020)": "5 years (2010-2020)",
        "3500 - (1990-2010)": "10 years (1990-2010)",
        "3500 - (1990-2020)": "10 years (1990-2020)",
        "3500 - (2010-2020)": "10 years (2010-2020)",
    }
    df["Period"] = df["Period"].replace(period_renames)
    return df


PERIOD_ORDER = [
    "10 years (2010-2020)",
    "10 years (1990-2010)",
    "10 years (1990-2020)",
    "5 years (2010-2020)",
    "5 years (1990-2010)",
    "5 years (1990-2020)",
]


def plot_hr_sensitivity(
    hr_df: pd.DataFrame,
    model_filter: str,
    *,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Faceted HR errorbar panel for one model, matching R's HR_sensiblity.R."""
    set_publication_style()

    sub = hr_df[hr_df["Model"] == model_filter].copy()
    if sub.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data for: {model_filter}", ha="center", va="center")
        return fig

    variables = sub["Variable"].unique()
    nrows = int(np.ceil(len(variables) / ncols))
    if figsize is None:
        figsize = (ncols * 5, nrows * 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, var in enumerate(variables):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        vdf = sub[sub["Variable"] == var].copy()

        # Set period order
        period_cat = pd.Categorical(vdf["Period"], categories=PERIOD_ORDER, ordered=True)
        vdf = vdf.copy()
        vdf["Period"] = period_cat
        vdf = vdf.sort_values("Period")

        y_pos = range(len(vdf))
        labels = vdf["Period"].tolist()

        ax.errorbar(
            vdf["HR"],
            y_pos,
            xerr=[vdf["HR"] - vdf["LB"], vdf["UP"] - vdf["HR"]],
            fmt="o",
            color="#0073C2",
            capsize=3,
            elinewidth=1.2,
            capthick=1.2,
            markersize=5,
        )
        ax.axvline(x=1, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("HR (95% CI)", fontsize=7)
        ax.set_title(var, fontsize=8, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for idx in range(len(variables), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(model_filter, fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig
