"""CLI: quantile survival summaries → quantile_models.xlsx."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..data import load_complemented
from ..preprocessing import apply_global_renames, apply_subset_rules, range_normalize
from ..quantiles import build_quantile_tables
from ..io import write_xlsx


MODEL_KEYS = [
    "oncology_phase2",
    "oncology_phase3",
    "infectious_phase2",
    "infectious_phase3",
    "cardiovascular_phase2",
    "cardiovascular_phase3",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quantile survival summaries")
    parser.add_argument(
        "--config", default="configs/covariates.yaml",
        help="Path to covariates.yaml",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    df_raw = load_complemented(cfg.input_cfg["csv"])
    df = apply_global_renames(df_raw, cfg)

    tables_dir = Path(cfg.result_dir("tables"))
    tables_dir.mkdir(parents=True, exist_ok=True)

    model_results = []
    for key in MODEL_KEYS:
        model_cfg = cfg.models.get(key)
        if model_cfg is None:
            continue
        sub = apply_subset_rules(df, model_cfg)
        if sub.empty or len(sub) < 20:
            print(f"  [{key}] insufficient rows, skipping.")
            continue

        final_covariates = model_cfg.get("final_covariates", [])
        if not final_covariates:
            continue
        keep_cols = ["duration", "event"] + [
            c for c in final_covariates if c in sub.columns
        ]
        sub = sub[keep_cols].dropna()
        sub = range_normalize(sub, duration_col="duration", event_col="event")
        model_results.append((key, sub))
        print(f"  [{key}] prepared (n={len(sub)})")

    print("Building quantile tables ...")
    tables = build_quantile_tables(model_results)

    out_path = tables_dir / "quantile_models.xlsx"
    write_xlsx(tables, out_path)
    print(f"Quantile models written → {out_path}")


if __name__ == "__main__":
    main()
