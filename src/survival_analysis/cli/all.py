"""CLI: run all analyses end-to-end."""
from __future__ import annotations

import argparse
import sys
import traceback

from . import cox, km, agents, hr_sensitivity, clusters, quantiles


_STEPS = [
    ("cox",             cox.main,            "Cox PH models + forest plots"),
    ("km",              km.main,             "Kaplan-Meier stratified curves"),
    ("agents",          agents.main,         "Masking-agent analysis (Phase 3 Onco)"),
    ("hr_sensitivity",  hr_sensitivity.main, "HR sensitivity plots"),
    ("clusters",        clusters.main,       "Cluster / distribution plots"),
    ("quantiles",       quantiles.main,      "Quantile survival summaries"),
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run all survival analyses")
    parser.add_argument(
        "--config", default="configs/covariates.yaml",
        help="Path to covariates.yaml",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Steps to skip (e.g. --skip hr_sensitivity clusters)",
    )
    args = parser.parse_args(argv)

    config_args = ["--config", args.config]
    failed = []

    for step_name, step_fn, description in _STEPS:
        if step_name in args.skip:
            print(f"\n[{step_name}] skipped.")
            continue
        print(f"\n{'='*60}")
        print(f"[{step_name}] {description}")
        print("="*60)
        try:
            step_fn(config_args)
        except SystemExit:
            pass
        except Exception:
            print(f"[{step_name}] FAILED:")
            traceback.print_exc()
            failed.append(step_name)

    print("\n" + "="*60)
    if failed:
        print(f"Completed with errors in: {failed}")
        sys.exit(1)
    else:
        print("All steps completed successfully.")


if __name__ == "__main__":
    main()
