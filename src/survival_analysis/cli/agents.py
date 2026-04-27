"""CLI: Phase 3 Oncology masking-agent analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ..config import load_config
from ..data import load_complemented
from ..preprocessing import apply_global_renames
from ..agents import run_agent_analysis, plot_agents_separately, plot_agent_combinations
from ..io import save_figure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Masking-agent Cox analysis (Phase 3 Onco)")
    parser.add_argument(
        "--config", default="configs/covariates.yaml",
        help="Path to covariates.yaml",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df_raw = load_complemented(cfg.input_cfg["csv"])
    df = apply_global_renames(df_raw, cfg)

    agents_dir = Path(cfg.result_dir("agents"))
    agents_dir.mkdir(parents=True, exist_ok=True)

    print("Running agent analysis ...")
    try:
        separate, cph_combos, combos_df = run_agent_analysis(df)
    except Exception as exc:
        print(f"Agent analysis failed: {exc}")
        return

    fig1 = plot_agents_separately(separate)
    save_figure(fig1, agents_dir / "agents_separately")
    plt.close(fig1)
    print("Saved -> agents_separately.*")

    fig2 = plot_agent_combinations(cph_combos, combos_df)
    save_figure(fig2, agents_dir / "agent_combinations")
    plt.close(fig2)
    print("Saved -> agent_combinations.*")


if __name__ == "__main__":
    main()
