# Responsible and Interpretable Machine Learning for Prostate Cancer Data

Undergraduate Research Honors Program project by Keely Morris, Department of Computing, East Tennessee State University.

This repository supports a two-part thesis about how dataset structure changes what machine-learning results can legitimately claim.

## Thesis structure

### Part 1: Post-diagnostic log-PSA pilot

The current 97-patient dataset contains tumor-burden and pathology-related variables. It is useful for comparing regression algorithms and studying model stability, but it cannot validate prostate-cancer screening, early detection, or cancer-risk claims.

Part 1 compares five algorithms:

- Linear regression
- Decision tree
- Random forest
- Gradient boosting
- Neural network

The thesis-grade analysis uses five-fold cross-validation repeated ten times, keeps preprocessing inside each fold, removes the source dataset's `train` split indicator from the predictors, and calculates held-out permutation importance.

### Part 2: Pre-diagnostic risk-modeling framework

Part 2 is a literature-grounded framework for a PLCO-based study using variables that exist before diagnosis, including age, race and ethnicity, family history, longitudinal PSA results, DRE results, screening history, diagnostic follow-up, biopsy linkage, and biopsy-confirmed outcomes. It is intentionally separated from Part 1 because predicting log PSA from post-diagnostic variables is not the same task as predicting future or biopsy-confirmed cancer risk.

## Run the validated pilot analysis

```bash
python -m pip install -r requirements.txt
python run_research_analysis.py
```

The script writes reproducible outputs to `results/research/`:

- `model_cv_summary.csv`: repeated cross-validation means and standard deviations
- `fold_metrics.csv`: all held-out fold results
- `permutation_importance_summary.csv`: feature importance aggregated across held-out folds
- `permutation_importance_raw.csv`: fold-level permutation results
- `run_metadata.json`: analysis role, outcome, predictors, exclusions, and seed

## Current verified pilot result

Linear regression ranks first by mean held-out RMSE. Across 50 held-out evaluations, its mean R² is 0.535 (SD 0.185), mean RMSE is 0.733 (SD 0.124), and mean MAE is 0.566 (SD 0.106). Log cancer volume is the most stable held-out permutation feature.

These values are research results for a small, post-diagnostic log-PSA dataset. They are not evidence of clinical utility, cancer detection, or screening performance.

## Repository map

- `src/research_analysis.py`: leakage-safe repeated cross-validation and interpretability analysis
- `run_research_analysis.py`: command-line entry point for the thesis pilot
- `src/models/`: original educational single-split model implementations
- `docs/thesis_draft.md`: revised two-part paper with IEEE headings, numbered in-text citations, and 24 references
- `docs/literature_review_matrix.md`: 24-source evidence matrix aligned to IEEE citation order
- `docs/advisor_feedback_plan.md`: Dr. Husari's feedback translated into project actions
- `tests/`: regression tests for metadata exclusion and analysis coverage

## Reproducibility and limitations

- The `train` column is source-publication metadata and is never used as a predictor.
- The outcome is `lpsa`, the natural logarithm of PSA, not a cancer diagnosis.
- The dataset has only 97 observations, so performance estimates remain uncertain.
- Model comparison is exploratory and requires external validation before any clinical interpretation.
- Part 2 requires approved access to PLCO data through the National Cancer Institute's Cancer Data Access System.

## Research standards

The thesis plan follows TRIPOD+AI for transparent reporting and uses PROBAST+AI concepts to discuss risk of bias and applicability. The PLCO variable plan is grounded in the official NCI data dictionary rather than inferred from unrelated datasets.
