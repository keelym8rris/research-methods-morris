# From PSA prediction to prostate cancer risk

Keely Morris · Computing · East Tennessee State University · Undergraduate Research Honors Program

I started this project by comparing models for predicting log PSA. The thesis connects that computing work to a broader question: what does a prediction mean when the population, measurements, and timing change?

Part I contains descriptive analysis and an eight-model regression comparison using 97 observations. Part II is a focused literature review and a discussion of dataset reliability before and after diagnosis. A new PLCO experiment is possible future work.

## Manuscript and results

- [Thesis draft with numbered IEEE-style references and reading links](docs/thesis_draft.md)
- [Python-generated figures in PNG, PDF, and SVG](results/thesis/figures)
- [Model comparison](results/thesis/model_cv_summary.csv)
- [Descriptive statistics](results/thesis/descriptive_statistics.csv)
- [Saved predictions, fold assignments, and supporting results](results/thesis)

## Main finding

| Model | Mean fold RMSE | Mean fold R² |
|---|---:|---:|
| Ridge | 0.731 | 0.538 |
| Ordinary linear regression | 0.733 | 0.535 |
| Lasso | 0.735 | 0.540 |
| Random forest | 0.776 | 0.494 |
| Gradient boosting | 0.790 | 0.475 |
| MLP (L-BFGS) | 0.814 | 0.435 |
| Decision tree | 0.914 | 0.289 |
| Training-mean baseline | 1.154 | -0.087 |

Errors are in log-PSA units. All models use the same five-fold partitions repeated ten times. The linear methods perform similarly; their small differences do not establish a statistically superior model. These results concern PSA prediction in this sample and do not demonstrate cancer detection.

## Reproduce the analysis

Use Python 3.12 in a fresh virtual environment, then run from the repository root:

```bash
python -m pip install -r requirements-research.txt
python run_thesis_analysis.py
python make_thesis_figures.py
python tests/verify_results.py
python build_thesis.py --output outputs/Keely_Morris_IEEE_Thesis_Draft.docx
```

The figure script reads saved CSVs without fitting models. The Word builder produces a plain, single-column review draft with IEEE-style references; it is not a journal or ETSU submission template.

The saved results contain 400 fold evaluations and 7,760 held-out predictions for the same 97 observations. Repeated predictions do not increase the number of independent patients. Reported standard deviations describe variation across overlapping folds, not confidence intervals.

The verification script checks prediction membership, recomputes fold metrics, checks the mean baseline, and examines training-only scaling. Run metadata preserves the provenance of the saved computation. The current cleanup changes names and documentation; it does not represent a new fitting run.

## Sources and scope

The manuscript cites only sources whose full text was available for inspection. Research findings are tied to the population and outcome actually studied. Official NCI documentation supports PLCO field definitions; the American Cancer Society supplies risk-factor background. The supplied PET/CT article was read in full through the provided copy; its publisher link may require institutional access.

Part II is a narrative review, not a systematic review or a new PLCO experiment. The distinctions among PSA prediction, future diagnosis, lesion classification, and postdiagnosis prognosis are central to the thesis.

## Earlier analysis

`run_research_analysis.py` and `results/research/` retain the earlier five-model comparison. The notebook and `src/models/` contain the original single-split examples. Current thesis claims use `results/thesis/`.

The `train` column is historical split metadata, not a predictor. [The original teaching-data description](https://hastie.su.domains/ElemStatLearn/datasets/prostate.info.txt) documents the distributed dataset.
