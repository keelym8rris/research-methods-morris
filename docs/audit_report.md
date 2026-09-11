# Research audit: September 11, 2026

## Scope and baseline

Reviewed the thesis branch at `2cb40f8e4b50f2f1f1681f98405d3e5e9680a703`, including analysis code, CSV, saved results, draft, references and tutorial notebook. Compared the older `main` branch (`76ed33004033c76cb10f6bf8c5cc1c8147626c1b`) and divergent `improvements` branch (`d108a6bdfebbf59ebf1e29ca06ffc9e002806c6b`). Changes belong to the existing advisor branch and draft PR; this audit does not merge those branches.

This is a code, data, claim and reproducibility review. It is not an exhaustive security audit, formal clinical validation, systematic literature review, or advisor approval.

## Findings and actions

| Finding | Action and evidence |
|---|---|
| Original study and teaching-file sample counts were conflated. | Draft now distinguishes 102 men in the original abstract from 97 records in the teaching file. No undocumented subset explanation is invented. |
| Older educational workflow included historical `train` metadata; the advisor pipeline already excluded it. | Verified eight true predictors. Corrected the notebook feature selection and retained exclusion checks. The magnitude of any prior score inflation is unknown. |
| Single-split educational scores were easy to confuse with research results. | README identifies the audited pipeline as the thesis source. Notebook outputs were cleared, misleading interpretations corrected, and its historical role labeled. TensorFlow notebook execution was not repeated. |
| Earlier five-model summary needed independent reproduction. | Reran the July pipeline and compared against the original saved summary; numeric agreement within absolute tolerance 0.000000001. Historical output files are preserved. |
| Neural-network internal validation was not isolated from scaling. | Outer test folds were already excluded. For the audited extension, the MLP uses L-BFGS with no early-stopping split. Multiple settings changed; improved scores cannot be attributed to one change alone. |
| A five-model ranking lacked simple reference comparisons. | Added training-mean, ridge and lasso baselines. The new comparison is exploratory because this dataset has been examined previously. |
| Calling one model conclusively best overstated close results. | Reported all eight models, overlapping fold variability, and the near tie among linear models. No significance test using correlated folds or clinical-superiority claim. |
| Source interpretations blurred screening, risk and prognosis. | Separated outcomes, populations and timing; narrowed references to 16 supported uses. Full-text limitations are explicit in `source_claim_audit.md`. |
| PLCO biopsy fields do not automatically produce biopsy-confirmed outcomes. | Proposed primary endpoint is recorded confirmed diagnosis. Biopsy confirmation requires linkage and usable result fields; missing results cannot mean negative. |
| Figures and draft needed traceable inputs. | Four figures generated from saved CSVs in PNG/PDF/SVG; three embedded in the draft. No invented PLCO performance charts. |
| Old command-line output option and Word builder were inconsistent with docs. | Added `--output-dir` to the historical analysis entry point; old Word builder delegates to the current builder. |
| Legacy plots could imply causal direction or suppress negative R². | Corrected affected labels/axis restrictions and neutralized importance coloring. |

## Verification performed

- Compared every numeric data value with the official ESL-hosted teaching file at absolute tolerance 0.000000000001 and relative tolerance zero; verified T/F split mapping. Found 97 rows, eight predictors, no missing values or duplicate rows.
- Ran the full audited comparison: 8 models × 50 folds; zero captured fitting warnings. Exact package versions, model settings and data/script hashes are in `results/audited/run_metadata.json`.
- Independently recalculated all 400 metric rows from the 7,760 saved predictions; checked each observation is held out once per repeat/model, all train/test partitions are disjoint, targets match the CSV, and the mean baseline equals each training mean.
- Checked training-only OLS scaler statistics and predictions on the first fold against the saved output; ran the existing predictor-exclusion and small repeated-CV checks.
- Parsed all notebook code cells after clearing stale outputs. The historical TensorFlow workflow remains unexecuted in this audit.
- Checked all Python-generated figure files, draft table values, reference numbering and links. Rendered the revised Word file and inspected every page.

## What the evidence establishes

In this sample and at the fixed evaluated settings, linear models have lower mean log-PSA error than the evaluated nonlinear alternatives. Ridge, ordinary linear regression, and lasso are close. OLS reliance is greatest for log cancer volume. The experiment demonstrates an auditable computing workflow and a carefully bounded result.

## Remaining limits and advisor decisions

The sample is small, retrospective, selected, and contains pathology-related measurements. Repeated CV does not supply independent external patients, causal evidence, screening outcomes, or prospective validation. The revised configurations were not preregistered. Permutation interpretation shares evaluation folds and is descriptive. Correlated predictors complicate coefficient and importance interpretation.

Part II is a literature review and protocol. It contains no patient-level PLCO analysis, empirical fairness analysis, calibration curve, or decision-curve result. No claim of novelty over the full literature is justified by a focused narrative search. The advisor must confirm whether this scope fulfills the thesis or whether an approved PLCO project and additional analyses are required. Formal ETSU/venue formatting, institutional requirements and final advisor approval remain separate from this review.
