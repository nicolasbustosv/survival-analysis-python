"""CLI: run all Cox PH models + forest plots."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import load_config
from ..data import load_complemented
from ..preprocessing import apply_global_renames, apply_subset_rules, range_normalize
from ..cox import fit_cox, cox_summary, univariate_screen, backward_aic
from ..plots.forest import plot_forest
from ..io import save_figure, write_xlsx


MODEL_KEYS = [
    "oncology_phase2",
    "oncology_phase3",
    "infectious_phase2",
    "infectious_phase3",
    "cardiovascular_phase2",
    "cardiovascular_phase3",
]

FOREST_LABELS = {
    "oncology_phase2":      "ggforest_O_P2",
    "oncology_phase3":      "ggforest_O_P3",
    "infectious_phase2":    "ggforest_ID_P2",
    "infectious_phase3":    "ggforest_ID_P3",
    "cardiovascular_phase2":"ggforest_C_P2",
    "cardiovascular_phase3":"ggforest_C_P3",
}


def _run_model(
    model_key: str,
    df: pd.DataFrame,
    model_cfg: dict,
    result_dir: Path,
) -> pd.DataFrame | None:
    """Run one Cox model end-to-end; return summary DataFrame or None on failure."""
    print(f"  [{model_key}] filtering ...")
    sub = apply_subset_rules(df, model_cfg)
    if sub.empty or len(sub) < 20:
        print(f"  [{model_key}] insufficient rows ({len(sub)}), skipping.")
        return None

    final_covariates = model_cfg.get("final_covariates", [])
    if not final_covariates:
        print(f"  [{model_key}] no final_covariates configured, skipping.")
        return None

    keep_cols = ["duration", "event"] + [
        c for c in final_covariates if c in sub.columns
    ]
    sub = sub[keep_cols].dropna()

    sub_norm = range_normalize(sub, duration_col="duration", event_col="event")

    print(f"  [{model_key}] fitting Cox (n={len(sub_norm)}) ...")
    try:
        cph = fit_cox(sub_norm, duration_col="duration", event_col="event")
    except Exception as exc:
        print(f"  [{model_key}] Cox failed: {exc}")
        return None

    summ = cox_summary(cph)
    summ.insert(0, "model", model_key)

    label = FOREST_LABELS[model_key]
    fdir = result_dir / "forest"
    fdir.mkdir(parents=True, exist_ok=True)
    fig = plot_forest(
        cph, sub_norm,
        title=model_key.replace("_", " ").title(),
        figsize=tuple(model_cfg.get("forest", {}).get("figsize", [8, 6])),
    )
    save_figure(fig, fdir / label)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  [{model_key}] forest plot saved → {label}.*")

    return summ


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Cox PH models + forest plots")
    parser.add_argument(
        "--config", default="configs/covariates.yaml",
        help="Path to covariates.yaml",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df_raw = load_complemented(cfg.input_cfg["csv"])
    df = apply_global_renames(df_raw, cfg)

    result_dir = Path(cfg.result_dir("forest")).parent
    tables_dir = result_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, pd.DataFrame] = {}
    for key in MODEL_KEYS:
        model_cfg = cfg.models.get(key)
        if model_cfg is None:
            print(f"  [{key}] not found in config, skipping.")
            continue
        summ = _run_model(key, df, model_cfg, result_dir)
        if summ is not None:
            summaries[key] = summ

    if summaries:
        out_path = tables_dir / "cox_summaries.xlsx"
        write_xlsx(summaries, out_path)
        print(f"\nCox summaries written → {out_path}")
    else:
        print("\nNo models produced output.")


if __name__ == "__main__":
    main()
