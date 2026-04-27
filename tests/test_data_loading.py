"""Smoke tests for data loading."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

DATA_CSV = Path("data/raw/survival_data_complemented.csv")


@pytest.fixture(scope="module")
def raw_df():
    if not DATA_CSV.exists():
        pytest.skip(f"Data file not found: {DATA_CSV}")
    from survival_analysis.data import load_complemented
    return load_complemented(DATA_CSV)


def test_load_returns_dataframe(raw_df):
    assert isinstance(raw_df, pd.DataFrame)


def test_event_column_added(raw_df):
    assert "event" in raw_df.columns
    assert (raw_df["event"] == 1).all()


def test_duration_positive(raw_df):
    assert (raw_df["duration"] > 0).all()


def test_required_columns_present(raw_df):
    required = ["duration", "event", "phase", "Disease_Group"]
    for col in required:
        assert col in raw_df.columns, f"Missing column: {col}"


def test_pre_logged_columns_not_relogged(raw_df):
    # enrollment should be ~log(n_enrolled), so values should be in [0, 12] range
    # not raw counts like 1000+
    if "enrollment" in raw_df.columns:
        assert raw_df["enrollment"].max() < 20, (
            "enrollment looks un-logged (values > 20); check pre_logged_columns config"
        )
