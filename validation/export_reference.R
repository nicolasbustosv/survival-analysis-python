# export_reference.R
# Run ONCE in R to export Cox model coefficients for numeric comparison.
# Output: validation/r_reference/cox_summaries.csv
#
# Usage (from project root):
#   Rscript validation/export_reference.R
#
# Requires that sensibility_analysis.R has been run and the fitted model
# objects are still in the session, OR that final_survival_models.rData exists.
#
# If loading from RData:
#   load("../Codes/Survival_scripts/new_data/New Data Base/final_survival_models.rData")

library(dplyr)
library(tibble)

# Helper to extract tidy summary from a coxph object
extract_cox <- function(model, model_name) {
  s <- summary(model)
  coef_df <- as.data.frame(s$coefficients)
  ci_df   <- as.data.frame(s$conf.int)

  tibble(
    model    = model_name,
    term     = rownames(coef_df),
    coef     = coef_df[, "coef"],
    HR       = ci_df[, "exp(coef)"],
    lower95  = ci_df[, "lower .95"],
    upper95  = ci_df[, "upper .95"],
    p        = coef_df[, "Pr(>|z|)"]
  )
}

# Adjust model object names to match what sensibility_analysis.R creates.
# Common names from sensibility_analysis.R:
#   model_phase2_onco_bw    -> oncology_phase2
#   model_phase3_onco_bw    -> oncology_phase3
#   model_phase2_id_bw      -> infectious_phase2
#   model_phase3_id_bw      -> infectious_phase3
model_map <- list(
  oncology_phase2     = "model_phase2_onco_bw",
  oncology_phase3     = "model_phase3_onco_bw",
  infectious_phase2   = "model_phase2_id_bw",
  infectious_phase3   = "model_phase3_id_bw"
)

rows <- list()
for (python_name in names(model_map)) {
  r_name <- model_map[[python_name]]
  if (exists(r_name)) {
    rows[[python_name]] <- extract_cox(get(r_name), python_name)
  } else {
    message("Model not found: ", r_name, " — skipping")
  }
}

if (length(rows) == 0) {
  stop("No models found. Run sensibility_analysis.R first, or load an RData file.")
}

out <- bind_rows(rows)
out_path <- file.path("validation", "r_reference", "cox_summaries.csv")
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write.csv(out, out_path, row.names = FALSE)
message("Written: ", out_path, "  (", nrow(out), " rows)")
