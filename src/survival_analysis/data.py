"""Load and validate the primary survival dataset."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

EXPECTED_COLS = {
    "nct_id", "phase", "enrollment", "number_of_arms", "has_dmc",
    "Drug_Class", "allocation", "intervention_model", "primary_purpose",
    "masking", "subject_masked", "caregiver_masked", "investigator_masked",
    "outcomes_assessor_masked", "Experimental", "Control", "Other",
    "n_countries", "lead_agency", "n_collaborator", "us_international",
    "intervention_type", "Disease_Group", "duration",
    "Therapy", "Pivotal", "SPA", "Orphan", "Breakthrough", "RMAT", "QIDP",
    "Fast_Track", "n_primary_endpoint", "n_secondary_endpoint",
}


def load_complemented(path: str | Path) -> pd.DataFrame:
    """Read survival_data_complemented.csv.

    The CSV was produced by R after log-transforming numeric columns; enrollment,
    n_countries, and n_collaborator are already on log scale. duration is raw days.
    """
    df = pd.read_csv(path, index_col=0)
    df = _apply_string_types(df)
    df = _add_event_column(df)
    return df


def _apply_string_types(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from string columns and convert bool-like values."""
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    return df


def _add_event_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add event=1 for all rows.

    R's Surv(time=duration) with no event col treats every row as an event.
    All trials in the CSV are pre-filtered to overall_status == 'Completed'.
    """
    df = df.copy()
    df["event"] = 1
    return df


def validate_schema(df: pd.DataFrame, expected_cols: set[str] | None = None) -> None:
    expected = expected_cols or EXPECTED_COLS
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
