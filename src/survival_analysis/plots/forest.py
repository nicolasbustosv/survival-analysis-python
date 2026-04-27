"""ggforest-equivalent forest plot using matplotlib.

Layout mirrors survminer::ggforest:
  - Left column:  variable group label + level label + N count
  - Centre:       dot + errorbar on log-HR x-axis, vertical ref line at 1
  - Right column: HR (95% CI) + p-value text
"""
from __future__ import annotations

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lifelines import CoxPHFitter
from typing import Any

from .style import JCO_PALETTE, set_publication_style
from .._constants import DURATION_COL, EVENT_COL


def _get_forest_rows(
    cph: CoxPHFitter,
    data: pd.DataFrame,
    duration_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
) -> list[dict]:
    """Build row records from fitted model for forest plot rendering."""
    rows = []
    covariates = [c for c in data.columns if c not in {duration_col, event_col}]
    n_total = len(data)

    summary = cph.summary.copy()

    for cov in covariates:
        col_data = data[cov]
        is_categorical = col_data.dtype == object or str(col_data.dtype) == "category"

        if is_categorical:
            levels = sorted(col_data.dropna().unique())
            # Add variable header row (no estimate)
            rows.append({
                "label": cov,
                "sub_label": "",
                "is_header": True,
                "hr": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "p": np.nan,
                "n": n_total,
                "is_ref": False,
            })
            # Reference level = first level (treatment contrast)
            ref_level = levels[0]
            rows.append({
                "label": "",
                "sub_label": f"  {ref_level}",
                "is_header": False,
                "hr": 1.0,
                "lower": np.nan,
                "upper": np.nan,
                "p": np.nan,
                "n": int((col_data == ref_level).sum()),
                "is_ref": True,
            })
            for level in levels[1:]:
                # Find matching row in summary using partial name match
                pattern = re.escape(str(cov)) + r".*" + re.escape(str(level))
                matches = [
                    idx for idx in summary.index
                    if re.search(pattern, str(idx), re.IGNORECASE)
                    or str(idx) == f"{cov}[T.{level}]"
                    or str(idx).endswith(str(level))
                ]
                if matches:
                    row_key = matches[0]
                    hr = summary.loc[row_key, "exp(coef)"]
                    lo = summary.loc[row_key, "exp(coef) lower 95%"]
                    hi = summary.loc[row_key, "exp(coef) upper 95%"]
                    p = summary.loc[row_key, "p"]
                else:
                    hr = lo = hi = p = np.nan

                n_level = int((col_data == level).sum())
                rows.append({
                    "label": "",
                    "sub_label": f"  {level}",
                    "is_header": False,
                    "hr": hr,
                    "lower": lo,
                    "upper": hi,
                    "p": p,
                    "n": n_level,
                    "is_ref": False,
                })
        else:
            # Continuous variable — single row
            matches = [idx for idx in summary.index if str(idx) == cov]
            if matches:
                row_key = matches[0]
                hr = summary.loc[row_key, "exp(coef)"]
                lo = summary.loc[row_key, "exp(coef) lower 95%"]
                hi = summary.loc[row_key, "exp(coef) upper 95%"]
                p = summary.loc[row_key, "p"]
            else:
                hr = lo = hi = p = np.nan

            rows.append({
                "label": cov,
                "sub_label": "",
                "is_header": False,
                "hr": hr,
                "lower": lo,
                "upper": hi,
                "p": p,
                "n": n_total,
                "is_ref": False,
            })

    return rows


def plot_forest(
    cph: CoxPHFitter,
    data: pd.DataFrame,
    *,
    title: str = "",
    duration_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
    figsize: tuple[float, float] = (8, 5),
    fontsize: float = 8,
    palette: list[str] | None = None,
) -> matplotlib.figure.Figure:
    """Create a publication-quality forest plot matching R's ggforest layout."""
    set_publication_style()
    palette = palette or JCO_PALETTE
    rows = _get_forest_rows(cph, data, duration_col, event_col)
    n_rows = len(rows)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[2.5, 2.5, 1.8],
        wspace=0.05,
    )
    ax_left = fig.add_subplot(gs[0])
    ax_mid = fig.add_subplot(gs[1])
    ax_right = fig.add_subplot(gs[2])

    for ax in [ax_left, ax_mid, ax_right]:
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.invert_yaxis()
        ax.axis("off")

    # ── Column headers ────────────────────────────────────────────────────────
    header_y = -0.7
    ax_left.text(0.0, header_y, "Variable", fontsize=fontsize + 1,
                 fontweight="bold", va="center", ha="left")
    ax_left.text(0.82, header_y, "N", fontsize=fontsize + 1,
                 fontweight="bold", va="center", ha="right")
    ax_right.text(0.0, header_y, "HR (95% CI)", fontsize=fontsize,
                  fontweight="bold", va="center", ha="left")
    ax_right.text(0.7, header_y, "p", fontsize=fontsize,
                  fontweight="bold", va="center", ha="left")

    # ── Rows ─────────────────────────────────────────────────────────────────
    dot_color = palette[0]

    # Determine x-axis range for the midpanel (log scale)
    valid_hrs = [r["hr"] for r in rows if not np.isnan(r["hr"]) and not r["is_ref"]]
    valid_lo = [r["lower"] for r in rows if not np.isnan(r.get("lower", np.nan))]
    valid_hi = [r["upper"] for r in rows if not np.isnan(r.get("upper", np.nan))]

    all_vals = [v for v in valid_hrs + valid_lo + valid_hi if v > 0]
    xmin = max(0.05, min(all_vals) * 0.5) if all_vals else 0.1
    xmax = max(all_vals) * 2.0 if all_vals else 10.0
    xmin, xmax = min(xmin, 0.5), max(xmax, 2.0)

    ax_mid.set_xlim(xmin, xmax)
    ax_mid.set_xscale("log")
    ax_mid.axis("on")
    ax_mid.set_ylim(-0.5, n_rows - 0.5)
    ax_mid.invert_yaxis()
    ax_mid.spines["top"].set_visible(False)
    ax_mid.spines["left"].set_visible(False)
    ax_mid.spines["right"].set_visible(False)
    ax_mid.yaxis.set_visible(False)
    ax_mid.set_xlabel("Hazard Ratio (95% CI)", fontsize=fontsize)

    # Reference line at HR=1
    ax_mid.axvline(x=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    for i, row in enumerate(rows):
        y = i
        # ── Left panel: variable name + N ─────────────────────────────────
        label = row["label"] or row["sub_label"]
        weight = "bold" if row["is_header"] else "normal"
        style = "italic" if row["is_ref"] else "normal"
        color = "#555555" if row["is_header"] else "black"
        ax_left.text(0.0, y, label, fontsize=fontsize, fontweight=weight,
                     fontstyle=style, color=color, va="center", ha="left")
        ax_left.text(0.82, y, str(row["n"]), fontsize=fontsize,
                     va="center", ha="right", color="#333333")

        if row["is_header"] or row["is_ref"] or np.isnan(row["hr"]):
            if row["is_ref"]:
                ax_mid.plot([1.0], [y], "D", color="#888888", markersize=5)
            continue

        # ── Mid panel: dot + errorbar ──────────────────────────────────────
        hr, lo, hi = row["hr"], row["lower"], row["upper"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax_mid.errorbar(
                hr, y,
                xerr=[[hr - lo], [hi - hr]],
                fmt="o",
                color=dot_color,
                markersize=5,
                capsize=3,
                elinewidth=1.2,
                capthick=1.2,
            )
        else:
            ax_mid.plot(hr, y, "o", color=dot_color, markersize=5)

        # ── Right panel: HR text + p ───────────────────────────────────────
        if np.isfinite(hr):
            if np.isfinite(lo) and np.isfinite(hi):
                hr_text = f"{hr:.2f} ({lo:.2f}–{hi:.2f})"
            else:
                hr_text = f"{hr:.2f}"
            ax_right.text(0.0, y, hr_text, fontsize=fontsize - 0.5,
                          va="center", ha="left")

        p = row["p"]
        if np.isfinite(p):
            p_text = f"{p:.3f}" if p >= 0.001 else "<0.001"
            ax_right.text(0.7, y, p_text, fontsize=fontsize - 0.5,
                          va="center", ha="left")

    if title:
        fig.suptitle(title, fontsize=fontsize + 1, fontweight="bold", y=1.01)

    fig.tight_layout()
    return fig
