"""Shared model registry used by cox and quantiles CLI commands."""
from __future__ import annotations

MODEL_KEYS: list[str] = [
    "oncology_phase2",
    "oncology_phase3",
    "infectious_phase2",
    "infectious_phase3",
    "cardiovascular_phase2",
    "cardiovascular_phase3",
]

FOREST_LABELS: dict[str, str] = {
    "oncology_phase2":       "ggforest_O_P2",
    "oncology_phase3":       "ggforest_O_P3",
    "infectious_phase2":     "ggforest_ID_P2",
    "infectious_phase3":     "ggforest_ID_P3",
    "cardiovascular_phase2": "ggforest_C_P2",
    "cardiovascular_phase3": "ggforest_C_P3",
}
