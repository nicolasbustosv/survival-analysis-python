"""CLI: cluster / distribution plots by Disease_Group."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ..config import load_config
from ..data import load_complemented
from ..preprocessing import apply_global_renames
from ..clusters import (
    boxplot_by_disease_group,
    boxplot_without_oncology,
    endpoint_distribution,
)
from ..io import save_figure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cluster / distribution plots")
    parser.add_argument(
        "--config", default="configs/covariates.yaml",
        help="Path to covariates.yaml",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df_raw = load_complemented(cfg.input_cfg["csv"])
    df = apply_global_renames(df_raw, cfg)

    out_dir = Path(cfg.result_dir("clusters"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Plotting duration by Disease_Group ...")
    fig1 = boxplot_by_disease_group(df)
    save_figure(fig1, out_dir / "boxplot_by_disease_group")
    plt.close(fig1)

    print("Plotting duration by Disease_Group (excluding Oncology) ...")
    fig2 = boxplot_without_oncology(df)
    save_figure(fig2, out_dir / "wo_oncology")
    plt.close(fig2)

    print("Plotting endpoint distributions ...")
    fig3 = endpoint_distribution(df)
    save_figure(fig3, out_dir / "endpoint_distributions")
    plt.close(fig3)

    print(f"Cluster plots saved → {out_dir}")


if __name__ == "__main__":
    main()
