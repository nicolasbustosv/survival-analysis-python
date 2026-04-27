"""Load and validate YAML configurations."""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    covariates: dict[str, Any]
    plots: dict[str, Any]
    paths: dict[str, Any]

    @property
    def models(self) -> dict[str, Any]:
        return self.covariates.get("models", {})

    @property
    def global_cfg(self) -> dict[str, Any]:
        return self.covariates.get("global", {})

    @property
    def km_cfg(self) -> dict[str, Any]:
        return self.covariates.get("km", {})

    @property
    def hr_sensitivity_cfg(self) -> dict[str, Any]:
        return self.covariates.get("hr_sensitivity", {})

    @property
    def input_cfg(self) -> dict[str, Any]:
        return self.covariates.get("input", {})

    @property
    def jco_palette(self) -> list[str]:
        return self.plots.get("jco_palette", [
            "#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC",
            "#003C67", "#8F7700", "#3B3B3B", "#A73030", "#4A6990",
        ])

    def result_dir(self, key: str) -> Path:
        return Path(self.paths["results"].get(key, f"results/{key}"))


def load_config(
    covariates_path: str | Path = "configs/covariates.yaml",
    plots_path: str | Path = "configs/plots.yaml",
    paths_path: str | Path = "configs/paths.yaml",
) -> Config:
    def _load(p: str | Path) -> dict:
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f)

    return Config(
        covariates=_load(covariates_path),
        plots=_load(plots_path),
        paths=_load(paths_path),
    )
