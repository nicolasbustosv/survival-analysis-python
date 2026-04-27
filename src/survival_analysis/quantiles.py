"""Quantile survival summaries from Cox models — port of results_analysis.R."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .cox import fit_cox, cox_summary


def _percentile_survival(
    cph: CoxPHFitter,
    percentiles: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> pd.DataFrame:
    """
    Estimate time at which the baseline survival function crosses each percentile.
    Returns DataFrame with columns: percentile, time_estimate.
    """
    sf = cph.baseline_survival_
    records = []
    for q in percentiles:
        surv_level = 1.0 - q
        crossed = sf[sf["baseline survival"] <= surv_level]
        if not crossed.empty:
            t = crossed.index[0]
        else:
            t = np.nan
        records.append({"percentile": q, "time_estimate": t})
    return pd.DataFrame(records)


def quantile_summary(
    df: pd.DataFrame,
    model_name: str,
    duration_col: str = "duration",
    event_col: str = "event",
    percentiles: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> pd.DataFrame:
    """
    Fit Cox on df, extract HR table + baseline percentile estimates.
    Returns a combined DataFrame tagged with model_name.
    """
    cph = fit_cox(df, duration_col=duration_col, event_col=event_col)
    hr_df = cox_summary(cph)
    hr_df.insert(0, "model", model_name)

    pct_df = _percentile_survival(cph, percentiles)
    pct_df.insert(0, "model", model_name)

    return hr_df, pct_df


def build_quantile_tables(
    model_results: list[tuple[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """
    Aggregate HR and percentile tables across multiple models.

    Parameters
    ----------
    model_results : list of (model_name, prepared_df)
        Each tuple contains the model label and the final analysis-ready DataFrame.

    Returns
    -------
    dict with keys "hr_table" and "percentile_table".
    """
    hr_rows = []
    pct_rows = []

    for name, df in model_results:
        try:
            hr_df, pct_df = quantile_summary(df, model_name=name)
            hr_rows.append(hr_df)
            pct_rows.append(pct_df)
        except Exception as exc:
            print(f"  [quantiles] {name}: skipped — {exc}")

    hr_table = pd.concat(hr_rows, ignore_index=True) if hr_rows else pd.DataFrame()
    pct_table = pd.concat(pct_rows, ignore_index=True) if pct_rows else pd.DataFrame()

    return {"hr_table": hr_table, "percentile_table": pct_table}
