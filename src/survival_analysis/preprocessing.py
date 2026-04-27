"""Data preprocessing: level renames, subsets, normalization, factor encoding."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Any

from ._constants import DURATION_COL, EVENT_COL
from .config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Global level renames  (ported from sensibility_analysis.R L322-387)
# ─────────────────────────────────────────────────────────────────────────────

def apply_global_renames(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply global level renames from covariates.yaml global.level_renames."""
    df = df.copy()
    global_cfg = cfg.global_cfg
    renames = global_cfg.get("level_renames", {})

    for col, mapping in renames.items():
        if col not in df.columns:
            continue
        if "__keep_only__" in mapping:
            keep = mapping["__keep_only__"]
            df[col] = df[col].where(df[col] == keep, "Other")
        elif "__bin__" in mapping or "__bin_ordered__" in mapping:
            # handled separately in _bin_endpoints
            pass
        else:
            df[col] = df[col].replace(mapping)

    # Masking consolidation (R L69-76) — also fills actual NaN → "None"
    masking_renames = global_cfg.get("masking_renames", {})
    if "masking" in df.columns:
        if masking_renames:
            df["masking"] = df["masking"].replace(masking_renames)
        df["masking"] = df["masking"].fillna("None").replace("nan", "None").replace("NA", "None")

    # FDA designations: replace NaN with "No" (R L317-321)
    for col in ["Fast_Track", "Orphan", "Breakthrough", "RMAT", "QIDP"]:
        if col in df.columns:
            df[col] = df[col].replace({"nan": "No", "N": "No", "Y": "Yes"}).fillna("No")

    # Bin endpoints  (R L361-385)
    df = _bin_endpoints(df)

    return df


def _bin_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "n_primary_endpoint" in df.columns:
        col = pd.to_numeric(df["n_primary_endpoint"], errors="coerce")
        df["n_primary_endpoint"] = np.where(col > 4, ">4", "1-3")

    if "n_secondary_endpoint" in df.columns:
        col = pd.to_numeric(df["n_secondary_endpoint"], errors="coerce")
        bins = []
        for v in col:
            if pd.isna(v):
                bins.append(np.nan)
            elif v > 20:
                bins.append(">20")
            elif v >= 14:
                bins.append("14-20")
            elif v >= 8:
                bins.append("8-13")
            elif v >= 3:
                bins.append("3-7")
            else:
                bins.append("1-2")
        df["n_secondary_endpoint"] = bins

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Disease_Group_2 derived column (for KM curves)
# ─────────────────────────────────────────────────────────────────────────────

def add_disease_group_2(df: pd.DataFrame) -> pd.DataFrame:
    """Ported from sensibility_analysis.R L1935-1940."""
    df = df.copy()
    mapping = {
        "Oncology": "Oncology",
        "Infectious Disease": "Infectious Disease",
    }
    df["Disease_Group_2"] = df["Disease_Group"].map(mapping).fillna("Other")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-model subset preparation
# ─────────────────────────────────────────────────────────────────────────────

def apply_subset_rules(
    df: pd.DataFrame,
    model_cfg: dict[str, Any],
    for_bw: bool = False,
) -> pd.DataFrame:
    """Filter rows, drop columns, collapse levels — ported from model blocks in R."""
    df = df.copy()

    # Row filters: keep matching values
    for col, val in model_cfg.get("filter", {}).items():
        if col in df.columns:
            df = df[df[col] == val]

    # Row filters: exclude matching values
    for col, vals in model_cfg.get("filter_out", {}).items():
        if col in df.columns:
            df = df[~df[col].isin(vals)]

    # Drop columns
    drop = [c for c in model_cfg.get("drop_cols", []) if c in df.columns]
    df = df.drop(columns=drop)

    # Level collapses
    for col, rules in model_cfg.get("level_collapses", {}).items():
        if col not in df.columns:
            continue
        if rules is None:
            continue
        if "__keep_only__" in rules:
            keep = rules["__keep_only__"]
            if keep is None:
                # collapse a specific set to Other
                for_collapse = rules.get("other_values", [])
                df[col] = df[col].apply(lambda x: "Other" if x in for_collapse else x)
            else:
                df[col] = df[col].where(df[col] == keep, "Other")
        elif "__keep_set__" in rules:
            keep_set = rules["__keep_set__"]
            other_label = rules.get("other_label", "Other")
            df[col] = df[col].apply(lambda x: x if x in keep_set else other_label)
        elif "other_values" in rules:
            for v in rules["other_values"]:
                df[col] = df[col].replace(v, "Other")

    # Backward-selection-specific drops (columns with certain substrings)
    if for_bw:
        for pattern in model_cfg.get("bw_subset_drop", []):
            cols_to_drop = [c for c in df.columns if pattern in c]
            df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Range normalization  (caret::preProcess(method="range"))
# ─────────────────────────────────────────────────────────────────────────────

def range_normalize(
    df: pd.DataFrame,
    duration_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
) -> pd.DataFrame:
    """MinMax-scale numeric columns (excluding duration and event)."""
    df = df.copy()
    skip = {duration_col, event_col}
    num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in skip
    ]
    if not num_cols:
        return df
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Rename final-model covariates to human-readable names
# ─────────────────────────────────────────────────────────────────────────────

def rename_for_plot(
    df: pd.DataFrame,
    final_covariates: dict[str, str],
    duration_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
) -> tuple[pd.DataFrame, list[str]]:
    """Keep only final covariates + duration/event; rename to human-readable."""
    keep_raw = list(final_covariates.keys())
    keep_cols = [c for c in keep_raw if c in df.columns] + [duration_col, event_col]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    rename_map = {k: v for k, v in final_covariates.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    pretty_covs = [final_covariates[k] for k in keep_raw if k in rename_map]
    return df, pretty_covs
