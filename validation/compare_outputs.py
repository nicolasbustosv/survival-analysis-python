"""Compare Python Cox outputs against R reference."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

R_REF   = Path("validation/r_reference/cox_summaries.csv")
PY_OUT  = Path("results/tables/cox_summaries.xlsx")
TOL_HR  = 1e-3
TOL_P   = 1e-3


def load_python() -> pd.DataFrame:
    xl = pd.read_excel(PY_OUT, sheet_name=None)
    frames = []
    for model_name, df in xl.items():
        df = df.copy()
        df.insert(0, "model", model_name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    if not R_REF.exists():
        print(f"R reference not found: {R_REF}")
        print("Run validation/export_reference.R first.")
        sys.exit(1)

    if not PY_OUT.exists():
        print(f"Python output not found: {PY_OUT}")
        print("Run 'sa-cox' or 'sa-all' first.")
        sys.exit(1)

    r_df = pd.read_csv(R_REF)
    py_df = load_python()

    # Normalise term names (lifelines may use [T.level] notation)
    r_df["term_clean"]  = r_df["term"].str.replace(r"\[T\.", "_", regex=True).str.rstrip("]")
    py_df["term_clean"] = py_df["term"].str.replace(r"\[T\.", "_", regex=True).str.rstrip("]")

    merged = pd.merge(
        r_df[["model", "term_clean", "HR", "lower95", "upper95", "p"]],
        py_df[["model", "term_clean", "HR", "lower95", "upper95", "p"]],
        on=["model", "term_clean"],
        suffixes=("_r", "_py"),
    )

    if merged.empty:
        print("No matching (model, term) pairs found. Check column names.")
        sys.exit(1)

    merged["diff_HR"] = (merged["HR_r"] - merged["HR_py"]).abs()
    merged["diff_p"]  = (merged["p_r"]  - merged["p_py"]).abs()

    fails_hr = merged[merged["diff_HR"] > TOL_HR]
    fails_p  = merged[merged["diff_p"]  > TOL_P]

    print(f"Compared {len(merged)} (model, term) pairs.")
    print(f"HR tolerance  : {TOL_HR}   — {len(fails_hr)} pairs exceed")
    print(f"p  tolerance  : {TOL_P}   — {len(fails_p)} pairs exceed")

    if not fails_hr.empty:
        print("\nHR mismatches:")
        print(fails_hr[["model", "term_clean", "HR_r", "HR_py", "diff_HR"]].to_string(index=False))

    if not fails_p.empty:
        print("\np mismatches:")
        print(fails_p[["model", "term_clean", "p_r", "p_py", "diff_p"]].to_string(index=False))

    if fails_hr.empty and fails_p.empty:
        print("\nAll outputs within tolerance.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
