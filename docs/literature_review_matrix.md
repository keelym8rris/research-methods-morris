# Literature Review Matrix

## Priority sources

### [1] G. S. Collins et al., “TRIPOD+AI statement,” BMJ, 2024

Link: https://doi.org/10.1136/bmj-2023-078378

Use: Reporting framework for both thesis parts. It supports transparent definitions of the target population, outcome, predictors, data preparation, validation, missing-data handling, performance measures, fairness, and limitations.

### [2] K. G. M. Moons et al., “PROBAST+AI,” BMJ, 2025

Link: https://doi.org/10.1136/bmj-2024-082505

Use: Risk-of-bias and applicability framework. It is especially relevant to the 97-patient sample, predictor timing, overfitting, outcome definition, and the difference between internal and external validation.

### [3] O. Efthimiou et al., “Developing clinical prediction models: a step-by-step guide,” BMJ, 2024

Link: https://doi.org/10.1136/bmj-2023-078276

Use: Methodological foundation for designing Part 2 before selecting models. It supports defining the estimand and intended clinical use before data preprocessing or algorithm comparison.

### [4] R. D. Riley et al., “Importance of sample size on the quality and utility of AI-based prediction models for healthcare,” The Lancet Digital Health, 2025

Link: https://doi.org/10.1016/S2589-7500(25)00021-4

Use: Justifies conservative interpretation of the 97-patient pilot. A small dataset can produce unstable individual predictions and overly optimistic conclusions even when average cross-validation performance appears acceptable.

### [5] J. A. Gelfond et al., “Prediction of future risk of any and higher-grade prostate cancer based on the PLCO and SELECT trials,” BMC Urology, 2022

Link: https://doi.org/10.1186/s12894-022-00986-w

Use: Closest direct comparator for Part 2. It shows how longitudinal trial data can support one- to five-year risk estimation and why calibration across cohorts matters. It also defines work the thesis should avoid duplicating.

### [6] K. Guan et al., “Ten-Year Trends in Digital Rectal Exam Results and Prostate Cancer Detection: Insights from the PLCO Trial,” Research and Reports in Urology, 2025

Link: https://doi.org/10.2147/RRU.S542550

Use: The paper recommended by Dr. Husari. It analyzes 34,756 PLCO participants, including 1,713 Black and 33,043 White men, and demonstrates how longitudinal DRE patterns and demographic context can be studied without treating a single exam as a complete risk model.

### [7] J.-E. Bibault et al., “Development and Validation of an Interpretable Artificial Intelligence Model to Predict 10-Year Prostate Cancer Mortality,” Cancers, 2021

Link: https://doi.org/10.3390/cancers13123064

Use: Strong example of an interpretable gradient-boosting model using prospective, multicenter PLCO data. It is post-diagnosis and predicts mortality, so it provides an important contrast with screening and biopsy-risk models.

### [8] M. Aladwani et al., “Prediction models for prostate cancer to be used in the primary care setting: a systematic review,” BMJ Open, 2020

Link: https://doi.org/10.1136/bmjopen-2019-034661

Use: Establishes what primary-care prediction models require and helps separate clinically available pre-diagnostic variables from pathology variables that would not exist at the prediction time.

### [9] P. Dhiman et al., “Risk of bias of prognostic models developed using machine learning: a systematic review in oncology,” Diagnostic and Prognostic Research, 2022

Link: https://doi.org/10.1186/s41512-022-00126-w

Use: Supports the thesis's focus on methodological validity. It documents recurring risk-of-bias problems in oncology machine-learning studies and strengthens the rationale for leakage-safe validation and explicit claim boundaries.

## Authoritative data documentation

### [10] National Cancer Institute, PLCO Cancer Data Access System

Links:

- https://cdas.cancer.gov/plco/
- https://cdas.cancer.gov/files/download/pcvhtvubq8/pros-dictionary-t20241011.pdf

Use: Primary source for available PLCO variables and access requirements. The prostate dataset includes longitudinal PSA and DRE results, biopsy linkage after positive screens, confirmed prostate-cancer outcomes, demographic variables, screening history, and time-to-event fields.

## Source selection guidance

The core thesis should prioritize [1]-[7] and [10]. Sources [8]-[9] are valuable for literature synthesis and bias discussion. Studies focused only on MRI, radiomics, histopathology images, recurrence after treatment, or telesurgery should be excluded unless used briefly to explain why prediction tasks cannot be compared without matching the clinical time point and outcome.
