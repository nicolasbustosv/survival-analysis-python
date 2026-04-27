# Clinical Trials Survival Analysis

Python port of the R survival analysis pipeline for Phase 2 and Phase 3 clinical trial
time-to-completion modeling. Covers Cox PH regression, Kaplan-Meier curves, masking-agent
analysis, and HR sensitivity plots for Oncology, Infectious Disease, and Cardiovascular
disease groups.

## Setup

```bash
pip install -e .
```

Requires Python 3.10+. See `requirements.txt` for pinned dependency versions.

## Data

Place the input files in `data/raw/` before running:

| File | Source |
|------|--------|
| `survival_data_complemented.csv` | R ETL output (see `data/raw/README.md`) |
| `hr_sensibility.csv` | Precomputed HR sensitivity table |

See `data/raw/README.md` for column descriptions and format notes.

> **Note**: `enrollment`, `n_countries`, and `n_collaborator` are already
> log-transformed in the CSV. The pipeline does not re-apply log to these columns.

## Running

All outputs are written to `results/` (gitignored; recreated on each run).

```bash
# All analyses at once
sa-all

# Individual steps
sa-cox             # Cox models + forest plots
sa-km              # Kaplan-Meier curves
sa-agents          # Masking-agent analysis (Phase 3 Oncology)
sa-hr-sens         # HR sensitivity plots
sa-clusters        # Distribution plots by Disease Group
sa-quantiles       # Quantile survival tables

# Skip steps
sa-all --skip hr_sensitivity clusters

# Custom config path
sa-all --config configs/covariates.yaml
```

All CLIs also accept `python -m survival_analysis.cli.<name>`.

## Output structure

```
results/
├── forest/         # Forest plots: ggforest_O_P2.*, O_P3.*, ID_P2.*, ID_P3.*, C_P2.*, C_P3.*
├── km/             # KM curves per stratification variable
├── agents/         # agents_separately.*, agent_combinations.*
├── hr_sensitivity/ # ggforest_5_years.pdf, ggforest_10_years.pdf, per-model PNGs
├── clusters/       # boxplot_by_disease_group.*, wo_oncology.*
└── tables/         # cox_summaries.xlsx, quantile_models.xlsx
```

Figures are saved in PNG and SVG formats (300 DPI).

## Validation

1. Run the R reference export once:
   ```r
   Rscript validation/export_reference.R
   ```
2. Run all Python analyses:
   ```bash
   sa-all
   ```
3. Compare numeric outputs:
   ```bash
   python validation/compare_outputs.py
   ```

Target tolerance: max absolute diff in HR < 0.001, p-values < 0.001.

## Tests

```bash
pytest -q
```

## Design notes

- **Input is post-log-transform**: the R ETL applies `mutate_if(is.numeric, log)` before
  writing `survival_data_complemented.csv`. This pipeline does NOT re-log any columns.
- **`event = 1` for all rows**: all trials are pre-filtered to "Completed" status, so
  this is regression on time-to-completion, not true right-censored survival. The column
  is synthesized on load to match R's `Surv(time = duration)` (no event argument).
- **Backward AIC**: hand-rolled to drop entire variables (not individual dummies), matching
  R's `rms::selectCox` behavior. Falls back to the hardcoded `final_covariates` list in
  `configs/covariates.yaml` if needed.
- **HR sensitivity**: the Python pipeline reads the precomputed `hr_sensibility.csv` for
  plotting only. Recomputing the window-subset Cox models requires re-running the R script.
- **Cardiovascular models**: the R source has this block commented out, but outputs exist
  in the Final Results folder from a prior run. These models are included in the Python
  pipeline; if the filtered dataset is too small, the step is skipped gracefully.
- **Cluster analysis**: the temporal k-means from `cluster_analysis.R` is not reproducible
  because `start_date` was dropped before writing `survival_data_complemented.csv`.
  The clusters module produces boxplots by Disease_Group instead.
- **OneDrive**: if running inside OneDrive, pause sync during `sa-all` to avoid file-lock
  conflicts during parallel matplotlib saves.
