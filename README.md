# Prostate Cancer Machine Learning Research

This is my undergraduate research project at ETSU. I'm using Python to explore a prostate cancer dataset and compare how different machine learning models predict log PSA, a transformed version of the PSA measurement.

I started by looking at the data and trying a few models, then expanded the comparison to see how the results change across different training and testing groups. So far, the simpler linear models have performed about the same as each other and better than the more complex models I tested. These results are about predicting PSA in this dataset, not detecting cancer.

## What's in the repo

- `data/` contains the dataset.
- `src/` contains the code for loading data, building models, and evaluating results.
- `Prostate_Cancer_ML_Tutorial.ipynb` contains my earlier exploration.
- `run_thesis_analysis.py` runs the expanded model comparison.
- `make_thesis_figures.py` creates graphs from the saved results.
- `results/thesis/` contains the expanded results and graphs. Earlier results are in `results/research/`.
- `tests/` contains checks for the analysis and saved results.

## Running the analysis

Use Python 3.12 and run these commands from the project folder:

```bash
python -m pip install -r requirements-research.txt
python run_thesis_analysis.py
python make_thesis_figures.py
python tests/verify_results.py
```

The dataset comes from [The Elements of Statistical Learning teaching data](https://hastie.su.domains/ElemStatLearn/datasets/prostate.info.txt).

I keep the written thesis separately in a Word document.
