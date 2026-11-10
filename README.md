# Prostate Cancer Machine Learning Research

This is my undergraduate research project at ETSU. I use Python to compare machine-learning models for predicting log PSA from a prostate cancer dataset.

## What's in the repo

- `data/` contains the dataset.
- `src/` contains the analysis code.
- `run_thesis_analysis.py` runs the larger model comparison.
- `make_thesis_figures.py` creates the figures from saved results and the predictor data.
- `results/thesis/` contains the main results and figures.
- `run_research_analysis.py` runs the earlier five-model comparison.
- `tests/` contains checks for the analysis.

## Running it

Use Python 3.12 from the project folder:

```bash
python -m pip install -r requirements-research.txt
python run_thesis_analysis.py --output-dir results/check/thesis
python tests/verify_results.py --results-dir results/check/thesis
python make_thesis_figures.py --results-dir results/check/thesis
```

The main comparison uses 97 records, eight predictors, and repeated five-fold cross-validation. It predicts natural-log PSA (`lpsa`). The source file's `train` marker is kept as metadata and is not used as a predictor.

To recreate the figures from the saved thesis results, run:

```bash
python make_thesis_figures.py
```

The script saves six figures as PNG, PDF, and SVG files in `results/thesis/figures/`. It does not refit the models. The two additions are:

- `05_r_squared_comparison`: mean R² across the 50 held-out folds for each model.
- `06_feature_correlations`: Pearson correlations among the eight predictors, using all 97 records. The values are also saved in `results/thesis/predictor_correlations.csv`.

The figure filenames keep their original sequence; their numbers differ from the figure numbers in the Word thesis.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tests/verify_results.py
```

The checks recalculate the saved metrics and verify the held-out predictions and fold assignments.

The written thesis is kept separately in a Word document.

## Age and prostate weight comparison

```bash
python run_age_weight_analysis.py
```

This follow-up keeps only `age` and `lweight` (log prostate weight) and uses the same standardized ridge model with `alpha=1.0`. It checks that its 50 folds match the saved thesis partitions and reproduces the original full ridge and mean-baseline results before saving the new comparison in `results/age_weight/`.

Mean fold RMSE is 1.059 for age and weight, 0.731 for full ridge, and 1.154 for the mean-only baseline, all in natural-log PSA units. The reduced model improves on the baseline by 8.2%, but its error is 44.8% higher than full ridge. This is an exploratory comparison of log-PSA prediction. The documentation does not establish that the stored weight was available before surgery, and the results do not demonstrate early cancer detection.

