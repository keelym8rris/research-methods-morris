# Responsible and Interpretable Machine Learning for PSA Prediction and Prostate Cancer Risk Modeling

Keely Morris  
Department of Computing, East Tennessee State University  
Undergraduate Research Honors Program  
Working draft, July 2026

## Abstract

Machine-learning studies in prostate cancer often use the language of screening, detection, prediction, and prognosis as if these tasks were interchangeable. They are not. The validity of a model's claim depends on when its predictors become available, how its outcome is defined, and whether its evaluation represents the population in which the model would be used. This thesis investigates how dataset structure affects the reliability and clinical validity of machine-learning claims in prostate cancer research. Part 1 uses a 97-patient post-diagnostic dataset to compare linear regression, decision tree, random forest, gradient boosting, and neural-network models for predicting log prostate-specific antigen (PSA). Leakage-safe five-fold cross-validation repeated ten times shows that linear regression has the lowest mean held-out error, with mean R² of 0.535, RMSE of 0.733, and MAE of 0.566. However, the dataset includes tumor-volume and pathology-related variables and therefore cannot validate screening or early-detection claims. Part 2 proposes a literature-grounded framework using pre-diagnostic variables from the Prostate, Lung, Colorectal and Ovarian Cancer Screening Trial (PLCO), including longitudinal PSA and digital rectal examination results, demographic factors, screening history, diagnostic follow-up, biopsy linkage, and biopsy-confirmed outcomes. The central argument is that responsible medical machine learning requires claim-to-data alignment before algorithm selection. This thesis contributes a reproducible pilot analysis, an interpretable evaluation workflow, and a framework for separating post-diagnostic prediction from clinically valid risk modeling.

Keywords: prostate cancer, machine learning, PSA, PLCO, clinical prediction, interpretability, data leakage, risk of bias

## 1. Introduction

Prostate cancer research includes several related but distinct prediction tasks. A model may estimate a laboratory value, identify cancer at biopsy, predict future diagnosis, estimate tumor aggressiveness, or predict mortality after diagnosis. Each task has a different population, prediction time, outcome, and set of valid predictors. A model trained with variables collected after diagnosis may accurately predict another post-diagnostic measurement, but it cannot be described as an early-detection model because the required information would not exist when screening decisions are made.

This distinction became clear during preliminary modeling with a 97-patient prostate dataset. The dataset contains log cancer volume, log prostate weight, age, benign prostatic hyperplasia, seminal vesicle invasion, capsular penetration, Gleason score, percentage of Gleason pattern 4 or 5, and log PSA. Several predictors describe tumor burden or pathology after cancer has already been evaluated. These variables may explain variation in log PSA, but they do not form a pre-diagnostic screening feature set.

The thesis therefore treats dataset structure as part of the research question rather than as a technical detail. Part 1 asks what can be learned responsibly from the available post-diagnostic data. Part 2 asks how a dataset designed around pre-diagnostic screening and longitudinal follow-up changes the type of claim that can be evaluated.

### 1.1 Research question

How does the temporal and clinical structure of a dataset affect the reliability, interpretability, and clinical validity of machine-learning models for PSA prediction and prostate cancer risk modeling?

### 1.2 Subquestions

1. How do five regression algorithms compare when predicting log PSA from a small post-diagnostic dataset under repeated, leakage-safe cross-validation?
2. Which patient and tumor variables contribute most consistently to held-out log-PSA prediction?
3. Which claims are and are not supported by post-diagnostic predictors?
4. How should a PLCO-based study define its population, predictors, time origin, outcome, and validation plan to support a pre-diagnostic risk claim?

### 1.3 Thesis statement

Dataset structure determines the boundary of a valid machine-learning claim. Post-diagnostic tumor and pathology variables can support exploratory log-PSA modeling, but they cannot validate screening or early-detection claims. Pre-diagnostic, longitudinal data with clearly timed biopsy-confirmed outcomes are necessary for clinically meaningful prostate-cancer risk modeling.

### 1.4 Contributions

This thesis makes four contributions. First, it provides a reproducible comparison of five machine-learning algorithms on the pilot dataset. Second, it corrects an analysis flaw by removing the dataset's `train` indicator from the predictors and placing preprocessing inside each validation fold. Third, it adds held-out permutation importance to examine whether feature influence is stable outside the training data. Fourth, it develops a PLCO-based claim-validity framework that separates pre-diagnostic risk modeling from post-diagnostic association and prognosis.

## 2. Background and Related Work

Transparent clinical prediction research begins by defining the intended use, target population, outcome, predictors, and evaluation design. TRIPOD+AI provides a reporting framework for prediction models developed with regression or machine-learning methods [1]. PROBAST+AI complements that framework by assessing quality, risk of bias, and applicability [2]. Both are relevant because a model may be computationally correct while still answering the wrong clinical question.

Sample size is a central limitation in medical artificial intelligence. Small development datasets can produce unstable models, imprecise individual predictions, and optimistic performance estimates [4]. This concern is especially important in the 97-patient pilot, where a single holdout split contains only about 19 or 20 patients. Repeated cross-validation cannot create new information, but it reveals how performance varies across multiple plausible partitions and is more informative than reporting one favorable split.

PLCO is an NCI-sponsored randomized screening trial with approximately 155,000 participants and extensive screening, diagnosis, treatment, incidence, and mortality data [10]. The prostate data dictionary contains repeated PSA and DRE measures, screening histories, follow-up biopsy linkage, confirmed cancer outcomes, and time-to-event variables. These fields allow predictors to be restricted to information available before a chosen prediction time.

Existing PLCO studies demonstrate the range of valid but different clinical tasks. A future-risk model built with PLCO and SELECT data estimated one- to five-year risk of any and higher-grade prostate cancer and evaluated calibration across cohorts [5]. A longitudinal analysis of 34,756 PLCO participants examined abnormal DRE patterns and prostate-cancer detection, including differences by race [6]. An interpretable gradient-boosting study used PLCO participants already diagnosed with prostate cancer to predict ten-year mortality [7]. These studies should not be compared by performance alone because they use different populations, outcomes, and prediction times.

The gap addressed by this thesis is methodological rather than algorithmic. The purpose is not to claim that a new algorithm outperforms established prostate-cancer calculators. Instead, the thesis evaluates how the same principles of temporal alignment, leakage prevention, internal validation, interpretability, and calibrated language change what can responsibly be concluded from different datasets.

## 3. Part 1 Methods: Post-Diagnostic Log-PSA Pilot

### 3.1 Dataset and study role

Part 1 uses 97 patient records and predicts `lpsa`, the natural logarithm of PSA. The candidate predictors are log cancer volume, log prostate weight, age, log benign prostatic hyperplasia, seminal vesicle invasion, log capsular penetration, Gleason score, and percentage of Gleason pattern 4 or 5. The dataset also contains a Boolean `train` field inherited from the source publication. Because this field describes a prior data split rather than a patient characteristic, it is excluded from all analyses.

The study is a post-diagnostic regression pilot. It does not estimate cancer probability, determine biopsy need, identify cancer in an asymptomatic population, or establish clinical utility.

### 3.2 Algorithms

Five pre-specified models are compared: ordinary least-squares linear regression, a regularized-depth decision tree, random forest, gradient boosting, and a small feed-forward neural network. The linear model provides the most interpretable baseline. Tree ensembles represent nonlinear interactions. The neural network tests whether additional flexibility helps despite the small sample, while its expected overfitting risk is acknowledged.

### 3.3 Validation and preprocessing

Each algorithm is evaluated with five-fold cross-validation repeated ten times using a fixed random seed. All models use the same 50 train-test partitions. Scaling for linear regression and the neural network occurs inside the model pipeline and is fitted only on each training fold. Tree-based algorithms do not require scaling. Hyperparameters are pre-specified rather than selected on the held-out folds.

The primary ranking metric is mean held-out root mean squared error (RMSE). Mean absolute error (MAE) and R² are secondary metrics. The mean and standard deviation across the 50 evaluations are reported to show both average performance and instability. These fold results are correlated because observations appear in multiple repeated partitions; the standard deviations are descriptive measures of split sensitivity rather than formal confidence intervals.

### 3.4 Interpretability

The best model by mean RMSE is evaluated with permutation importance in a separate shuffled five-fold procedure. Within each fold, the fitted model is evaluated on held-out patients while one feature is repeatedly permuted. A feature is more important when permutation increases RMSE. Reporting the mean, standard deviation, and proportion of positive held-out importance values helps distinguish consistent signal from unstable rankings.

### 3.5 Reproducibility

The repository stores the complete fold metrics, model summary, raw and summarized permutation importance, predictor list, excluded metadata, validation design, selection metric, and random seed. Automated tests verify that the target and `train` metadata do not enter the predictor matrix.

## 4. Part 1 Results

Linear regression ranks first by mean held-out RMSE. Its mean R² is 0.535 with a standard deviation of 0.185, mean RMSE is 0.733 with a standard deviation of 0.124, and mean MAE is 0.566 with a standard deviation of 0.106 across 50 evaluations. Random forest ranks second with mean R² of 0.494 and mean RMSE of 0.776. Gradient boosting ranks third with mean R² of 0.475 and mean RMSE of 0.790. The decision tree and neural network show lower average performance and greater error.

The ranking indicates that additional model complexity does not improve held-out log-PSA prediction in this small dataset. The result is consistent with the idea that a relatively stable linear signal can outperform flexible models when the number of observations is limited.

Held-out permutation importance identifies log cancer volume as the strongest and most consistent feature. Its mean RMSE importance is 0.404 and is positive in every permutation evaluation. Seminal vesicle invasion and log prostate weight rank next, although their variability is substantially larger. The remaining features have small mean effects and inconsistent positive importance. These results describe predictive contribution within this post-diagnostic dataset; they do not identify causal risk factors.

## 5. Part 1 Discussion

The corrected analysis produces a more conservative conclusion than a single train-test split. Linear regression remains the strongest model, but its repeated-validation R² is lower than the earlier single-split estimate of approximately 0.75. The difference demonstrates why performance from one partition should not be treated as a stable property of a model, especially when the test set is small.

The best-performing predictor, log cancer volume, also exposes the clinical boundary of the experiment. Tumor volume can help explain PSA after the disease has been characterized, but it is not a general screening variable available to an undiagnosed patient. Gleason score, seminal vesicle invasion, and capsular penetration create the same problem. A high-performing model based on these fields would still not support an early-detection claim.

Part 1 therefore succeeds as a computing study of algorithm behavior, validation, and interpretability. It also serves as a negative methodological lesson: predictive performance cannot repair a mismatch between predictor timing and the clinical claim.

## 6. Part 2 Proposed Methods: Pre-Diagnostic Risk Modeling

### 6.1 Objective and estimand

The recommended primary objective is to estimate five-year risk of biopsy-confirmed prostate cancer among men without prostate cancer at the prediction time. A secondary outcome may be high-grade cancer if the biopsy-grade definition and event count are sufficient. The prediction time should be fixed at baseline or at a clearly defined screening visit.

This objective must be confirmed with Dr. Husari before data extraction because changing the time origin or prediction horizon changes which records are valid predictors and which participants are eligible.

### 6.2 Candidate predictors

Candidate predictors should be limited to information recorded at or before the prediction time. The official PLCO prostate dictionary supports age, race and ethnicity, screening-arm information, PSA screening history, prior DRE history, longitudinal PSA levels and results, longitudinal DRE results, and other questionnaire-based risk factors. Family history and additional demographic or behavioral variables should be included only when the exact PLCO fields and missingness are verified.

Biopsy linkage, confirmed cancer status, diagnosis date, Gleason grade, and treatment fields should define outcomes or follow-up. They must not be used as baseline predictors when the goal is pre-diagnostic risk estimation.

### 6.3 Cohort and temporal alignment

Eligible participants should have no prostate-cancer diagnosis before the prediction time and must have sufficient follow-up to determine the outcome or censoring status. Predictors should be assembled with an explicit cutoff timestamp. Any value measured after the cutoff should be excluded, even if it improves model performance.

The dataset should distinguish a predictor's clinical role from its database location. For example, PSA is a valid pre-diagnostic predictor when measured at the screening visit, while biopsy Gleason score is an outcome-related field available only after diagnostic workup.

### 6.4 Analysis plan

Descriptive analysis should report cohort construction, missingness, outcome prevalence, follow-up, and predictor distributions overall and by clinically relevant groups. Model development should begin with a transparent regression baseline before adding tree-based methods. Missing-data handling, preprocessing, and feature selection must occur inside resampling folds.

Performance reporting should include discrimination and calibration. For a fixed-horizon binary outcome, this includes area under the receiver operating characteristic curve, area under the precision-recall curve when appropriate, Brier score, calibration intercept, calibration slope, and a calibration plot. For time-to-event modeling, concordance and time-dependent calibration should be considered. Decision-curve analysis may be added only if a clinically meaningful threshold range is defined.

Internal validation should use nested cross-validation or bootstrap optimism correction when hyperparameters are tuned. A temporal, site-based, or external cohort validation is preferred if available. Subgroup performance should be reported with uncertainty, especially across racial and ethnic groups, but small subgroup sizes must not be overinterpreted.

### 6.5 Claim-to-data audit

Every experiment should include a short claim-to-data audit with five questions:

1. Who is the target population?
2. When is the prediction made?
3. Which variables are available at that moment?
4. What outcome is observed, and over what period?
5. What validation supports use beyond the development sample?

This audit is the conceptual link between Part 1 and Part 2. It turns data provenance into an explicit component of model validity.

## 7. Expected Significance

The thesis contributes to computing education by demonstrating that responsible machine learning is not limited to selecting an algorithm or maximizing accuracy. It requires precise problem formulation, reproducible evaluation, interpretable outputs, and an understanding of how database construction shapes scientific claims.

For prostate-cancer research, the project offers a clear distinction between three tasks that are frequently conflated: modeling a biomarker from post-diagnostic variables, predicting biopsy-confirmed disease from pre-diagnostic information, and predicting prognosis after diagnosis. Keeping these tasks separate makes conclusions more accurate and makes future models easier to evaluate and compare.

## 8. Limitations

Part 1 contains only 97 observations and eight valid predictors. Repeated cross-validation measures sensitivity to data splitting but does not replace external validation or eliminate overfitting. The permutation results are predictive associations and should not be interpreted causally. The dataset does not represent a screening population and cannot support early-detection or risk-calculator claims.

Part 2 is currently a proposed framework. Final feasibility depends on PLCO access, variable completeness, event counts, missingness, and the time available before thesis completion. PLCO also reflects the population, screening practices, and historical period of the original trial, so transportability to current practice requires careful discussion.

## 9. Conclusion

The initial pilot shows that linear regression outperforms more complex models for held-out log-PSA prediction in the 97-patient dataset, but the more important finding concerns validity rather than ranking. The strongest predictors are post-diagnostic tumor characteristics, so even accurate predictions would not constitute evidence for screening or early detection. A clinically meaningful risk question requires a dataset whose predictors exist before diagnosis and whose outcome is defined through subsequent biopsy-confirmed follow-up.

The two-part thesis makes this boundary visible. Part 1 demonstrates what a small post-diagnostic dataset can support. Part 2 defines how pre-diagnostic longitudinal data can support a different and stronger question. The overall conclusion is that dataset structure is not simply an input to machine learning; it determines the scientific claim the model is allowed to make.

## References

[1] G. S. Collins et al., “TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods,” BMJ, vol. 385, e078378, 2024, doi: 10.1136/bmj-2023-078378.

[2] K. G. M. Moons et al., “PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods,” BMJ, vol. 388, e082505, 2025, doi: 10.1136/bmj-2024-082505.

[3] O. Efthimiou et al., “Developing clinical prediction models: a step-by-step guide,” BMJ, vol. 386, e078276, 2024, doi: 10.1136/bmj-2023-078276.

[4] R. D. Riley et al., “Importance of sample size on the quality and utility of AI-based prediction models for healthcare,” Lancet Digital Health, vol. 7, no. 6, art. 100857, 2025, doi: 10.1016/S2589-7500(25)00021-4.

[5] J. A. Gelfond et al., “Prediction of future risk of any and higher-grade prostate cancer based on the PLCO and SELECT trials,” BMC Urology, vol. 22, art. 45, 2022, doi: 10.1186/s12894-022-00986-w.

[6] K. Guan et al., “Ten-Year Trends in Digital Rectal Exam Results and Prostate Cancer Detection: Insights from the PLCO Trial,” Research and Reports in Urology, vol. 17, pp. 309–320, 2025, doi: 10.2147/RRU.S542550.

[7] J.-E. Bibault et al., “Development and Validation of an Interpretable Artificial Intelligence Model to Predict 10-Year Prostate Cancer Mortality,” Cancers, vol. 13, no. 12, art. 3064, 2021, doi: 10.3390/cancers13123064.

[8] M. Aladwani et al., “Prediction models for prostate cancer to be used in the primary care setting: a systematic review,” BMJ Open, vol. 10, no. 7, e034661, 2020, doi: 10.1136/bmjopen-2019-034661.

[9] P. Dhiman et al., “Risk of bias of prognostic models developed using machine learning: a systematic review in oncology,” Diagnostic and Prognostic Research, vol. 6, art. 13, 2022, doi: 10.1186/s41512-022-00126-w.

[10] National Cancer Institute, “Prostate, Lung, Colorectal and Ovarian Cancer Screening Trial,” Cancer Data Access System. https://cdas.cancer.gov/plco/.
