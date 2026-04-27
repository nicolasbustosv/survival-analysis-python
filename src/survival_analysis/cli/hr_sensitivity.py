"""CLI: faceted HR sensitivity panel from hr_sensibility.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ..config import load_config
from ..plots.sensitivity import load_hr_sensitivity, plot_hr_sensitivity
from ..io import save_figure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="HR sensitivity plots from hr_sensibility.csv"
    )
    parser.add_argument(
        "--config", default="configs/covariates.yaml",
        help="Path to covariates.yaml",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    hr_csv = cfg.input_cfg.get("hr_sensitivity_csv", "data/raw/hr_sensibility.csv")
    out_dir = Path(cfg.result_dir("hr_sensitivity"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HR sensitivity data from {hr_csv} ...")
    try:
        hr_df = load_hr_sensitivity(hr_csv)
    except FileNotFoundError:
        print(f"hr_sensibility.csv not found at {hr_csv}. Skipping.")
        return

    models = hr_df["Model"].unique()
    print(f"  Found {len(models)} models: {list(models)}")

    for model in models:
        print(f"  [{model}] plotting ...")
        fig = plot_hr_sensitivity(hr_df, model_filter=model)
        safe_name = model.replace(" ", "_").replace("/", "-")
        save_figure(fig, out_dir / f"hr_sensitivity_{safe_name}", formats=("png", "svg"))

        pdf_path = out_dir / f"hr_sensitivity_{safe_name}.pdf"
        with PdfPages(str(pdf_path)) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{model}] saved → hr_sensitivity_{safe_name}.*")

    # Consolidated PDFs by duration bucket (5yr / 10yr) matching R output names
    for bucket_label, duration_tag in [("5 years", "ggforest_5_years"), ("10 years", "ggforest_10_years")]:
        bucket_models = [m for m in models if bucket_label in m]
        if not bucket_models:
            continue
        pdf_path = out_dir / f"{duration_tag}.pdf"
        with PdfPages(str(pdf_path)) as pdf:
            for m in bucket_models:
                fig = plot_hr_sensitivity(hr_df, model_filter=m)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        print(f"Consolidated PDF → {duration_tag}.pdf")


if __name__ == "__main__":
    main()
