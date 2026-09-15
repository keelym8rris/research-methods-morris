# Prostate Cancer Machine Learning Research

This is my undergraduate research project at ETSU. I'm using Python to explore a prostate cancer dataset and compare how different machine learning models predict log PSA, a transformed version of the PSA measurement.

I started by looking at the data and trying a few models, then expanded the comparison to see how the results change across different training and testing groups. So far, the simpler linear models have performed about the same as each other and better than the more complex models I tested. These results are about predicting PSA in this dataset, not detecting cancer.

## What's in the repo

- `data/` contains the dataset.
- `src/` contains the code for loading data, building models, and evaluating results.
- `run_thesis_analysis.py` runs the expanded model comparison.
- `make_thesis_figures.py` creates graphs from the saved results.
- `results/thesis/` contains the expanded results and graphs. Earlier results are in `results/research/`.
- `run_research_analysis.py` reproduces the earlier five-model comparison.
- `tests/` contains checks for the analysis and saved results.

## Running the analysis

Use Python 3.12 and run these commands from the project folder:

```bash
python -m pip install -r requirements-research.txt
python tests/verify_results.py
```

To rerun the eight-model thesis comparison without replacing the saved results:

```bash
python run_thesis_analysis.py --output-dir results/check/thesis
python tests/verify_results.py --results-dir results/check/thesis
python make_thesis_figures.py --results-dir results/check/thesis
```

To recreate the four figures from the saved thesis results, run `python make_thesis_figures.py`. Each figure is saved as PNG, PDF, and SVG. Omitting `--output-dir` when running the analysis replaces the files in `results/thesis/`.

The thesis comparison uses 97 records, eight predictors, and the same five-fold splits repeated ten times for all eight models. The outcome is natural-log PSA (`lpsa`); the source file's `train` marker is excluded from the predictors. Ridge, ordinary linear regression, and lasso have mean fold RMSEs of 0.731, 0.733, and 0.735. These small differences do not establish a reliable winner.

The original run information is in `results/thesis/run_metadata.json`, including the data hash, package versions, model settings, and warnings. The historical source hash refers to the computation commit named there. A new run writes its own metadata to the selected output folder. The earlier five-model comparison has different neural-network settings and is saved separately in `results/research/`.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tests/verify_results.py
```

The result checks recalculate metrics from the saved held-out predictions and verify fold membership, summaries, the training-mean baseline, and training-only scaling.

### Reproduction check: September 15, 2026

Both analyses completed on Windows with Python 3.12.14 and the pinned research packages. All five tests passed, and the thesis figures were regenerated in PNG, PDF, and SVG. The original data and saved research results were retained.

The earlier five-model results matched within a numerical tolerance of `1e-9`. In the thesis comparison, the seven non-MLP models, fold assignments, descriptive statistics, coefficients, and permutation importance also matched. The MLP's mean RMSE was 0.815131 in this run versus the saved 0.814273, with no recorded fitting warnings and the same model ranking. The precise cause of this small numerical difference was not isolated; the saved outputs remain the source for the manuscript's reported values.

## Data and manuscript

The dataset comes from [The Elements of Statistical Learning teaching data](https://hastie.su.domains/ElemStatLearn/datasets/prostate.info.txt).

I keep the written thesis separately in a Word document.
