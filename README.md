# Prostate Cancer Machine Learning Research

This is my undergraduate research project at ETSU. I use Python to compare machine-learning models for predicting log PSA from a prostate cancer dataset.

## What's in the repo

- `data/` contains the dataset.
- `src/` contains the analysis code.
- `run_thesis_analysis.py` runs the larger model comparison.
- `make_thesis_figures.py` creates the figures from saved results.
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

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tests/verify_results.py
```

The checks recalculate the saved metrics and verify the held-out predictions and fold assignments.

The written thesis is kept separately in a Word document.
