# PSA prediction and the limits of clinical claims

Keely Morris · Computing · East Tennessee State University · Undergraduate Research Honors Program

This two-part thesis asks how dataset structure and predictor timing affect what machine-learning results can support. The current advisor draft is on `codex/advisor-feedback-thesis` in [draft PR #1](https://github.com/keelym8rris/research-methods-morris/pull/1).

**Part I is implemented:** eight regression configurations predict natural-log PSA in a 97-record teaching dataset. **Part II is a critical literature review and proposed PLCO study:** no PLCO patient-level experiment has been performed.

## Start here

- [Advisor draft with 16 IEEE-style references and reading links](docs/thesis_draft.md)
- [Source and claim audit](docs/source_claim_audit.md)
- [Code/data audit and remaining limitations](docs/audit_report.md)
- [How the two parts fit together](docs/how_it_ties_together.md)
- [Advisor feedback and scope decisions](docs/advisor_feedback_plan.md)
- [Python figures: PNG, PDF, SVG](results/audited/figures)

## Main finding

| Model | Mean fold RMSE, log PSA | Mean fold R² |
|---|---:|---:|
| Ridge | 0.731 | 0.538 |
| Ordinary linear regression | 0.733 | 0.535 |
| Lasso | 0.735 | 0.540 |
| Random forest | 0.776 | 0.494 |
| Gradient boosting | 0.790 | 0.475 |
| MLP (L-BFGS) | 0.814 | 0.435 |
| Decision tree | 0.914 | 0.289 |
| Training-mean baseline | 1.154 | -0.087 |

All models use the same 5-fold splits repeated 10 times. Ridge, OLS, and lasso perform similarly; these differences do not establish a statistically superior model. Pathology-related predictors help estimate PSA within this sample. These results do not demonstrate cancer detection or clinical usefulness.

## Reproduce from the repository root

Use Python 3.12 in a fresh virtual environment:

```bash
python -m pip install -r requirements-audit.txt
python run_audited_analysis.py
python make_thesis_figures.py
PYTHONPATH=. python tests/verify_audited_outputs.py
python build_advisor_draft.py --output outputs/Keely_Morris_IEEE_Thesis_Draft.docx
```

`run_audited_analysis.py` writes `results/audited/`. `make_thesis_figures.py` reads saved CSVs and does not refit models. The Word builder uses a plain single-column review layout with IEEE-style numbered references; it is not a specific journal or ETSU submission template. `build_ieee_paper.py` remains a compatibility entry point to the same builder.

## Evidence and validation

`results/audited/` includes 400 fold-metric rows, 7,760 held-out prediction rows, split membership, repeat-level metrics, descriptive summaries, OLS coefficients, and permutation importance. The predictions represent repeated evaluations of 97 people, not thousands of independent patients. Standard deviations describe split variation, not confidence intervals. `run_metadata.json` records settings, versions, warnings and hashes.

The verification script independently recomputes metrics from saved predictions, checks test coverage and train/test disjointness, verifies the mean baseline, and checks training-only scaling and OLS predictions on a fold.

## Earlier work

`run_research_analysis.py` and `results/research/` preserve the earlier five-model experiment. Its original summary was reproduced during this audit. The `src/models/` implementations and tutorial notebook are educational single-split examples. They are not the source of current thesis results; stale notebook outputs were cleared. The separate `improvements` branch was reviewed as exploratory work and remains separate.

The original clinical publication describes 102 men; the distributed teaching file contains 97 observations. Do not conflate those counts. The `train` field is historical split metadata and is excluded from predictors. Data are compared to [the official teaching file](https://hastie.su.domains/ElemStatLearn/datasets/prostate.data). See the audit for details and source access limitations.
