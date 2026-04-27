"""Kaplan-Meier estimation utilities."""
from __future__ import annotations

import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from typing import Any


def fit_km_strata(
    df: pd.DataFrame,
    factor_col: str,
    duration_col: str = "duration",
    event_col: str = "event",
) -> dict[str, KaplanMeierFitter]:
    """Fit a KM curve per stratum of factor_col.

    Returns {level: KaplanMeierFitter} in sorted order.
    """
    fits: dict[str, KaplanMeierFitter] = {}
    for level in sorted(df[factor_col].dropna().unique()):
        sub = df[df[factor_col] == level]
        kmf = KaplanMeierFitter(label=str(level))
        kmf.fit(sub[duration_col], event_observed=sub[event_col])
        fits[str(level)] = kmf
    return fits


def median_survival(km_fits: dict[str, KaplanMeierFitter]) -> pd.DataFrame:
    """Return median survival time per stratum."""
    rows = []
    for level, kmf in km_fits.items():
        rows.append({"level": level, "median_days": kmf.median_survival_time_})
    return pd.DataFrame(rows)


def logrank_p(
    df: pd.DataFrame,
    factor_col: str,
    duration_col: str = "duration",
    event_col: str = "event",
) -> float:
    """Log-rank test p-value across all strata of factor_col."""
    result = multivariate_logrank_test(
        df[duration_col], df[factor_col], df[event_col]
    )
    return float(result.p_value)
