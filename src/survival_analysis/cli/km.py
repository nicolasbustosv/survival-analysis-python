"""CLI: Kaplan-Meier stratified survival curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ..config import load_config
from ..data import load_complemented
from ..preprocessing import apply_global_renames, add_disease_group_2
from ..plots.km import plot_km
from ..io import save_figure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kaplan-Meier stratified plots")
    parser.add_argument(
        "--config", default="configs/covariates.yaml",
        help="Path to covariates.yaml",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df_raw = load_complemented(cfg.input_cfg["csv"])
    df = apply_global_renames(df_raw, cfg)
    df = add_disease_group_2(df)

    km_dir = Path(cfg.result_dir("km"))
    km_dir.mkdir(parents=True, exist_ok=True)

    km_cfg = cfg.km_cfg
    # Combine main stratifications + industry variants
    all_strats = list(km_cfg.get("stratifications", []))
    for strat in km_cfg.get("industry_stratifications", []):
        s = dict(strat)
        s.setdefault("subset_col", "lead_agency")
        s.setdefault("subset_val", "Industry")
        all_strats.append(s)

    for strat in all_strats:
        factor_col  = strat.get("factor_col") or strat.get("col")
        if not factor_col:
            continue
        label       = strat.get("label", factor_col)
        title       = strat.get("title", label)
        pval_coord  = strat.get("pval_coord", [1500, 0.55])
        legend_loc  = strat.get("legend_loc", [0.78, 0.78])
        subset_col  = strat.get("subset_col")
        subset_val  = strat.get("subset_val")
        filename    = strat.get("filename", f"km_{label}")

        sub = df.copy()
        if subset_col and subset_val:
            sub = sub[sub[subset_col] == subset_val]

        if factor_col not in sub.columns:
            print(f"  [km] {factor_col} not in data, skipping.")
            continue
        sub = sub.dropna(subset=[factor_col, "duration", "event"])
        if sub.empty:
            print(f"  [km] {factor_col}: no rows after dropna, skipping.")
            continue

        print(f"  [km] {factor_col} (n={len(sub)}) ...")
        fig = plot_km(
            sub,
            factor_col=factor_col,
            title=title,
            pval_coord=tuple(pval_coord),
            legend_loc=tuple(legend_loc),
        )
        save_figure(fig, km_dir / filename)
        plt.close(fig)
        print(f"  [km] saved -> {filename}.*")


if __name__ == "__main__":
    main()
