"""Masking-agent combinatorial Cox analysis (agent_cphm.R port)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lifelines import CoxPHFitter

from .cox import fit_cox, cox_summary
from .plots.style import JCO_PALETTE, set_publication_style


AGENT_COLS = {
    "subject_masked":          "Subject",
    "caregiver_masked":        "Care_provider",
    "investigator_masked":     "Investigator",
    "outcomes_assessor_masked":"Outcome_assessor",
}

BOOL_MAP = {"TRUE": True, "FALSE": False, "True": True, "False": False,
            "1": True, "0": False, True: True, False: False}


def _to_bool(series: pd.Series) -> pd.Series:
    return series.map(BOOL_MAP).fillna(False).astype(bool)


def prepare_phase3_onco(df: pd.DataFrame) -> pd.DataFrame:
    """Filter Phase 3 Oncology, select agent columns (agent_cphm.R L18-30)."""
    sub = df[
        (df["phase"] == "Phase 3") &
        (df["Disease_Group"] == "Oncology") &
        (df["intervention_model"] != "NA")
    ][["duration", "event"] + list(AGENT_COLS.keys())].copy()
    sub = sub.rename(columns=AGENT_COLS)
    for col in AGENT_COLS.values():
        sub[col] = _to_bool(sub[col])
    return sub.dropna()


def _fit_separate_agents(agents_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One Cox per agent (Subject, Care_provider, Investigator, Outcome_assessor)."""
    results = {}
    for col in AGENT_COLS.values():
        sub = agents_df[["duration", "event", col]].copy()
        # Convert bool to categorical True/False for readable forest plot
        sub[col] = sub[col].map({True: "True", False: "False"})
        try:
            cph = fit_cox(sub, duration_col="duration", event_col="event")
            results[col] = cox_summary(cph)
            results[col]["variable"] = col
        except Exception:
            pass
    return results


def _make_combinations(agents_df: pd.DataFrame) -> pd.DataFrame:
    """Create S/C/I/O combination factor (agent_cphm.R L33-96)."""
    df = agents_df.copy()
    S = df["Subject"]
    C = df["Care_provider"]
    I = df["Investigator"]
    O = df["Outcome_assessor"]

    S_C_I_O = S & C & I & O
    S_C_I   = S & C & I & ~S_C_I_O
    S_I     = S & I & ~S_C_I_O & ~S_C_I

    df["Combination"] = "Other"
    df.loc[S_I,     "Combination"] = "S_I"
    df.loc[S_C_I,   "Combination"] = "S_C_I"
    df.loc[S_C_I_O, "Combination"] = "S_C_I_O"

    df["Combination"] = pd.Categorical(
        df["Combination"],
        categories=["Other", "S_I", "S_C_I", "S_C_I_O"],
    )
    return df[["duration", "event", "Combination"]]


def run_agent_analysis(
    df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], CoxPHFitter, pd.DataFrame]:
    """Full agent analysis. Returns (separate_results, combos_model, combos_df)."""
    agents_df = prepare_phase3_onco(df)

    separate = _fit_separate_agents(agents_df)

    combos_df = _make_combinations(agents_df)
    cph_combos = fit_cox(combos_df, duration_col="duration", event_col="event")

    return separate, cph_combos, combos_df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _simple_forest(
    rows: list[dict],
    title: str = "",
    figsize: tuple[float, float] = (8, 3),
) -> matplotlib.figure.Figure:
    """Generic forest plot for agent analyses."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0.1, 10)
    ax.set_xscale("log")
    ax.axvline(1.0, color="black", linestyle="--", lw=0.8)

    n = len(rows)
    ax.set_ylim(-0.5, n - 0.5)

    for i, row in enumerate(rows):
        y = n - 1 - i
        hr, lo, hi = row.get("hr"), row.get("lower"), row.get("upper")
        label = row.get("label", "")
        if hr is None or not np.isfinite(hr):
            ax.text(-0.05, y, label, transform=ax.get_yaxis_transform(),
                    fontsize=7, ha="right", va="center")
            continue
        ax.errorbar(hr, y,
                    xerr=[[hr - lo], [hi - hr]] if (lo and hi) else None,
                    fmt="o", color=JCO_PALETTE[0], markersize=5,
                    capsize=3, elinewidth=1.2)
        p = row.get("p", np.nan)
        p_str = f"p={p:.3f}" if p and np.isfinite(p) else ""
        ax.text(hi * 1.05 if hi else hr * 1.1, y,
                f"{hr:.2f} {p_str}", fontsize=6.5, va="center")
        ax.text(-0.05, y, label, transform=ax.get_yaxis_transform(),
                fontsize=7, ha="right", va="center")

    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=8)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_agents_separately(
    separate: dict[str, pd.DataFrame],
) -> matplotlib.figure.Figure:
    """Forest plot with one entry per agent (Agents Separately)."""
    rows = []
    for agent, summ in separate.items():
        true_rows = summ[summ["term"].str.contains("True", na=False)]
        for _, r in true_rows.iterrows():
            rows.append({
                "label": f"{agent} (True vs False)",
                "hr": r["HR"],
                "lower": r["lower95"],
                "upper": r["upper95"],
                "p": r["p"],
            })
    return _simple_forest(rows, title="Masking Agents — Separately",
                          figsize=(8, max(2, len(rows) * 0.5 + 1)))


def plot_agent_combinations(
    cph_combos: CoxPHFitter,
    combos_df: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Forest plot for combination factor (Agent Combinations)."""
    from .plots.forest import plot_forest
    return plot_forest(cph_combos, combos_df, title="Agent Combinations",
                       figsize=(8, 3))
