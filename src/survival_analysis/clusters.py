"""Cluster analysis — boxplot by Disease_Group (temporal k-means not reproducible)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from .plots.style import JCO_PALETTE, set_publication_style


def boxplot_by_disease_group(
    df: pd.DataFrame,
    duration_col: str = "duration",
    group_col: str = "Disease_Group",
    *,
    figsize: tuple[float, float] = (8, 5),
    title: str = "Duration by Disease Group",
    log_scale: bool = True,
) -> matplotlib.figure.Figure:
    """Boxplot of trial duration by Disease_Group."""
    set_publication_style()

    sub = df[[group_col, duration_col]].dropna()
    groups = sorted(sub[group_col].unique())
    palette = {g: JCO_PALETTE[i % len(JCO_PALETTE)] for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=sub,
        x=group_col,
        y=duration_col,
        hue=group_col,
        palette=palette,
        order=groups,
        width=0.5,
        linewidth=0.8,
        fliersize=2,
        legend=False,
        ax=ax,
    )

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Duration (days, log scale)", fontsize=9)
    else:
        ax.set_ylabel("Duration (days)", fontsize=9)

    ax.set_xlabel("Disease Group", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    medians = sub.groupby(group_col)[duration_col].median()
    for i, g in enumerate(groups):
        ax.text(
            i, medians[g] * 1.05 if log_scale else medians[g] + 5,
            f"Mdn={medians[g]:.0f}",
            ha="center", va="bottom", fontsize=7,
        )

    fig.tight_layout()
    return fig


def boxplot_without_oncology(
    df: pd.DataFrame,
    duration_col: str = "duration",
    group_col: str = "Disease_Group",
    *,
    figsize: tuple[float, float] = (8, 5),
) -> matplotlib.figure.Figure:
    """Boxplot excluding Oncology (mirrors cluster_analysis.R 'wo_oncology')."""
    sub = df[df[group_col] != "Oncology"].copy()
    return boxplot_by_disease_group(
        sub,
        duration_col=duration_col,
        group_col=group_col,
        figsize=figsize,
        title="Duration by Disease Group (excluding Oncology)",
    )


def endpoint_distribution(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (10, 4),
) -> matplotlib.figure.Figure:
    """Side-by-side histograms of primary and secondary endpoint counts."""
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, col, label in zip(
        axes,
        ["n_primary_endpoint", "n_secondary_endpoint"],
        ["Primary Endpoints", "Secondary Endpoints"],
    ):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        data = df[col].dropna()
        ax.hist(data, bins=20, color=JCO_PALETTE[0], edgecolor="white", linewidth=0.4)
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Endpoint Count Distributions", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig
