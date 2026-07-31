# Dr. Husari Feedback Implementation Plan

Source: email dated July 4, 2026, forwarded to Gmail on July 24, 2026.

## Advisor direction

Dr. Husari endorsed the two-part thesis and described the project as a combination of data science, machine learning, and public-health analytics. His suggested structure is:

1. Complete the descriptive and predictive analysis of the 97-patient dataset, compare machine-learning algorithms, and discuss the strengths and weaknesses of each method.
2. Add a chapter based on a dataset with pre-diagnostic variables such as DRE, demographic factors, and risk factors. Use a literature review and discuss how dataset type affects the reliability of pre- and post-diagnostic claims.

## Changes implemented now

- Removed the source dataset's `train` indicator from all predictors.
- Added a reproducible comparison of five algorithms using five-fold cross-validation repeated ten times.
- Moved scaling inside model pipelines so each held-out fold remains isolated.
- Replaced a single train/test winner claim with mean and standard deviation across 50 held-out evaluations per model.
- Added held-out permutation importance for the best-performing model.
- Added tests that prevent the `train` indicator or outcome from leaking into the feature matrix.
- Added explicit language distinguishing log-PSA prediction from screening, diagnosis, and cancer-risk prediction.
- Added a PLCO chapter plan grounded in the official NCI prostate data dictionary.
- Added literature anchors for interpretability, prediction-model reporting, risk of bias, and sample-size limitations.
- Expanded the evidence base from 10 to 24 sources and converted the working paper to IEEE structure and reference style.

## Recommended next research meeting decisions

1. Confirm whether Part 2 will be a fully implemented PLCO analysis or a literature review and preregistered framework.
2. Choose one Part 2 estimand before requesting data. The strongest option is five-year risk of biopsy-confirmed prostate cancer among participants cancer-free at baseline.
3. Define the prediction time origin, prediction horizon, and handling of repeated PSA and DRE measurements.
4. Decide whether high-grade prostate cancer will be a secondary outcome.
5. Confirm access timing and the NCI project-proposal process.
6. Agree on a subgroup reporting plan for race and ethnicity, with uncertainty and fairness discussed explicitly.

## Non-duplication strategy for Part 2

Existing PLCO research already models future prostate-cancer risk and describes longitudinal DRE trends. The thesis should not attempt to reproduce those papers with a new algorithm and the same outcome. Its original contribution should be a claim-validity comparison:

- Apply the same transparent evaluation principles to two datasets with different temporal structures.
- Identify which predictors are available at the moment a prediction would actually be made.
- Create a claim-to-data audit showing why post-diagnostic predictors can support association or log-PSA prediction but not early-detection claims.
- Evaluate model performance, calibration, stability, and subgroup behavior only for outcomes supported by each dataset.
- Use a dataset card and a clinical-claim boundary for every experiment.

This keeps the computing contribution clear while preserving the public-health importance of data provenance and clinical timing.
