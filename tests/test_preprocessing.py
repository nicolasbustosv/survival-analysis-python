"""Tests for preprocessing utilities."""
import pytest
import numpy as np
import pandas as pd

from survival_analysis.preprocessing import range_normalize


def _make_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "duration": rng.uniform(100, 2000, 50),
        "event":    np.ones(50),
        "num_a":    rng.uniform(0, 10, 50),
        "num_b":    rng.uniform(-5, 5, 50),
        "cat_c":    pd.Categorical(["A", "B"] * 25),
    })


def test_range_normalize_bounds():
    df = _make_df()
    normed = range_normalize(df, duration_col="duration", event_col="event")
    numeric_cols = ["num_a", "num_b"]
    for col in numeric_cols:
        assert normed[col].min() >= 0.0 - 1e-9, f"{col} min below 0"
        assert normed[col].max() <= 1.0 + 1e-9, f"{col} max above 1"


def test_range_normalize_preserves_categoricals():
    df = _make_df()
    normed = range_normalize(df, duration_col="duration", event_col="event")
    assert list(normed["cat_c"]) == list(df["cat_c"])


def test_range_normalize_preserves_duration():
    df = _make_df()
    normed = range_normalize(df, duration_col="duration", event_col="event")
    pd.testing.assert_series_equal(normed["duration"], df["duration"], check_names=True)


def test_range_normalize_preserves_event():
    df = _make_df()
    normed = range_normalize(df, duration_col="duration", event_col="event")
    pd.testing.assert_series_equal(normed["event"], df["event"], check_names=True)
