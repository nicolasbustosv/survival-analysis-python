"""Cox proportional hazards modeling: fit, screen, backward AIC, VIF."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Core fit
# ─────────────────────────────────────────────────────────────────────────────

def _prep_for_lifelines(
    df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
) -> pd.DataFrame:
    """Expand categorical/string columns to dummies (treatment contrasts, drop_first).

    Lifelines without a formula expects all-numeric input.
    Numeric and boolean columns pass through unchanged.
    Duration and event columns are never dummified.
    """
    protect = {duration_col, event_col}
    cat_cols = [
        c for c in df.columns
        if c not in protect and (str(df[c].dtype) in ("category", "object") or df[c].dtype == bool)
    ]
    bool_cols = [c for c in cat_cols if df[c].dtype == bool]
    str_cat_cols = [c for c in cat_cols if c not in bool_cols]

    out = df.copy()
    # Bool -> int (0/1): no dummies needed
    for c in bool_cols:
        out[c] = out[c].astype(int)

    if str_cat_cols:
        dummies = pd.get_dummies(out[str_cat_cols], drop_first=True)
        out = out.drop(columns=str_cat_cols)
        out = pd.concat([out, dummies], axis=1)

    return out


def fit_cox(
    df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
    formula: str | None = None,
    penalizer: float = 0.0,
) -> CoxPHFitter:
    """Fit a Cox PH model.

    event_col should be all-1 (completed trials). If formula is given it is
    passed to lifelines as a patsy formula string.
    """
    df = _prep_for_lifelines(df, duration_col=duration_col, event_col=event_col)
    cph = CoxPHFitter(penalizer=penalizer)
    if formula:
        cph.fit(df, duration_col=duration_col, event_col=event_col, formula=formula)
    else:
        cph.fit(df, duration_col=duration_col, event_col=event_col)
    return cph


def cox_summary(cph: CoxPHFitter) -> pd.DataFrame:
    """Return tidy summary: variable, exp(coef)=HR, lower95, upper95, p."""
    s = cph.summary.copy()
    s = s[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    s.columns = ["HR", "lower95", "upper95", "p"]
    s.index.name = "term"
    s = s.reset_index()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Univariate screening  (uni_surv_models from survival_analysis_functions.R)
# ─────────────────────────────────────────────────────────────────────────────

def univariate_screen(
    df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
    p_threshold: float = 0.1,
) -> list[str]:
    """Return variable names where any level has p ≤ p_threshold in univariate Cox."""
    covariates = [c for c in df.columns if c not in {duration_col, event_col}]
    significant = []
    for col in covariates:
        subset = df[[col, duration_col, event_col]].dropna()
        if subset[col].nunique() < 2:
            continue
        try:
            cph = fit_cox(subset, duration_col=duration_col, event_col=event_col)
            if (cph.summary["p"] <= p_threshold).any():
                significant.append(col)
        except Exception:
            pass
    return significant


# ─────────────────────────────────────────────────────────────────────────────
# Backward AIC selection  (selectCox with rule="aic" from rms)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_aic(cph: CoxPHFitter) -> float:
    """AIC = -2 * partial log-likelihood + 2 * number of parameters."""
    ll = cph.log_likelihood_
    k = len(cph.params_)
    return -2 * ll + 2 * k


def _get_categorical_groups(df: pd.DataFrame, covariates: list[str]) -> dict[str, list[str]]:
    """Map each original variable name to the dummy column names it expands into.

    For a numeric column, the group is just [col_name].
    For a categorical column, we track it as a group so the whole factor is
    dropped or kept together (matching R's selectCox behaviour).
    """
    groups: dict[str, list[str]] = {}
    for col in covariates:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            groups[col] = [col]
        else:
            groups[col] = [col]
    return groups


def backward_aic(
    df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
    max_iter: int = 100,
) -> tuple[CoxPHFitter, list[str]]:
    """Backward AIC elimination, dropping whole variables at each step.

    Matches R's selectCox(rule='aic') semantics: at each step, try dropping
    every remaining variable; keep the drop that gives the lowest AIC.
    Stop when no drop improves AIC.

    Returns (final_model, remaining_variable_names).
    """
    covariates = [c for c in df.columns if c not in {duration_col, event_col}]
    current = list(covariates)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        full_df = df[current + [duration_col, event_col]].copy()
        try:
            base_cph = fit_cox(full_df, duration_col, event_col)
            current_aic = _compute_aic(base_cph)
            current_model = base_cph
        except Exception:
            return CoxPHFitter(), current

        for _ in range(max_iter):
            best_aic = current_aic
            best_drop = None
            best_model = current_model

            for col in current:
                candidate = [c for c in current if c != col]
                if not candidate:
                    continue
                sub = df[candidate + [duration_col, event_col]].dropna()
                try:
                    cph = fit_cox(sub, duration_col, event_col)
                    aic = _compute_aic(cph)
                    if aic < best_aic:
                        best_aic = aic
                        best_drop = col
                        best_model = cph
                except Exception:
                    continue

            if best_drop is None:
                break
            current.remove(best_drop)
            current_aic = best_aic
            current_model = best_model

    return current_model, current


# ─────────────────────────────────────────────────────────────────────────────
# VIF  (car::vif — diagnostic only, numeric columns)
# ─────────────────────────────────────────────────────────────────────────────

def vif_table(
    df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
) -> pd.DataFrame:
    """Compute VIF for numeric covariates. Returns DataFrame with vif and gen_r2."""
    num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in {duration_col, event_col}
    ]
    if len(num_cols) < 2:
        return pd.DataFrame(columns=["variable", "vif", "gen_r2"])

    X = df[num_cols].dropna()
    rows = []
    for i, col in enumerate(num_cols):
        try:
            v = variance_inflation_factor(X.values, i)
            rows.append({"variable": col, "vif": v, "gen_r2": 1 - 1 / v})
        except Exception:
            rows.append({"variable": col, "vif": np.nan, "gen_r2": np.nan})
    return pd.DataFrame(rows)
