"""Smoke tests for Cox model fitting."""
import pytest
import numpy as np
import pandas as pd

from survival_analysis.cox import fit_cox, cox_summary, univariate_screen


@pytest.fixture
def simple_df():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1, n)
    duration = np.exp(0.3 * x + rng.normal(0, 0.5, n)) * 500
    return pd.DataFrame({
        "duration": duration,
        "event":    np.ones(n),
        "x":        x,
        "group":    pd.Categorical(["A", "B"] * (n // 2)),
    })


def test_fit_cox_returns_fitter(simple_df):
    cph = fit_cox(simple_df, duration_col="duration", event_col="event")
    assert cph is not None


def test_cox_summary_columns(simple_df):
    cph = fit_cox(simple_df, duration_col="duration", event_col="event")
    summ = cox_summary(cph)
    assert "HR" in summ.columns
    assert "lower95" in summ.columns
    assert "upper95" in summ.columns
    assert "p" in summ.columns


def test_cox_hr_positive(simple_df):
    cph = fit_cox(simple_df, duration_col="duration", event_col="event")
    summ = cox_summary(cph)
    assert (summ["HR"] > 0).all()


def test_univariate_screen_returns_list(simple_df):
    kept = univariate_screen(
        simple_df,
        duration_col="duration",
        event_col="event",
        p_threshold=0.20,
    )
    assert isinstance(kept, list)
    assert "x" in kept
