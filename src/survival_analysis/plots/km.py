"""Kaplan-Meier survival curves (ggsurvplot equivalent)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from .style import jco_colors, set_publication_style


def plot_km(
    df: pd.DataFrame,
    factor_col: str,
    duration_col: str = "duration",
    event_col: str = "event",
    *,
    figsize_cm: tuple[float, float] = (12, 10),
    pval_coord: tuple[float, float] = (1500, 0.55),
    legend_loc: tuple[float, float] | str = (0.78, 0.78),
    ylim: tuple[float, float] = (0, 1),
    title: str = "",
    xlabel: str = "Time (days)",
) -> matplotlib.figure.Figure:
    """Stratified KM plot matching R create_km() function."""
    set_publication_style()

    cm = 1 / 2.54
    w, h = figsize_cm
    fig, ax = plt.subplots(figsize=(w * cm, h * cm))

    # Log-rank p-value
    groups = df[factor_col].dropna()
    levels = sorted(groups.unique())
    colors = jco_colors(len(levels))

    try:
        lr = multivariate_logrank_test(df[duration_col], df[factor_col], df[event_col])
        p_val = lr.p_value
    except Exception:
        p_val = np.nan

    for level, color in zip(levels, colors):
        sub = df[df[factor_col] == level]
        kmf = KaplanMeierFitter(label=str(level))
        kmf.fit(sub[duration_col], event_observed=sub[event_col])
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            show_censors=False,
            color=color,
            linewidth=1.5,
        )
        # Median line (horizontal + vertical dashes)
        med = kmf.median_survival_time_
        if np.isfinite(med):
            ax.hlines(0.5, xmin=0, xmax=med, colors=color, linestyles="--",
                      linewidth=0.8, alpha=0.5)
            ax.vlines(med, ymin=0, ymax=0.5, colors=color, linestyles="--",
                      linewidth=0.8, alpha=0.5)

    # p-value annotation
    if np.isfinite(p_val):
        p_text = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        ax.text(pval_coord[0], pval_coord[1], p_text, fontsize=7,
                va="center", ha="left")

    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Survival probability", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)

    if isinstance(legend_loc, (list, tuple)):
        ax.legend(loc="upper right", bbox_to_anchor=legend_loc, fontsize=7,
                  framealpha=0.8)
    else:
        ax.legend(loc=legend_loc, fontsize=7, framealpha=0.8)

    fig.tight_layout()
    return fig
