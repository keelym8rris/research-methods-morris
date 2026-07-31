# Responsible and Interpretable Machine Learning for PSA Prediction and Prostate Cancer Risk Modeling

Keely Morris  
Department of Computing, East Tennessee State University  
Undergraduate Research Honors Program  
Revised working paper, July 2026

## Abstract

Machine-learning studies in prostate cancer often use the terms screening, detection, prediction, and prognosis as if they describe the same task. They do not. A model's clinical meaning depends on when its predictors become available, how its outcome is defined, and whether its evaluation represents the population in which it would be used. This two-part study examines how dataset structure affects the reliability and validity of machine-learning claims. Part I compares five regression algorithms on a 97-patient post-diagnostic dataset for predicting log prostate-specific antigen (PSA). Five-fold cross-validation repeated ten times shows that linear regression has the lowest mean held-out error, with mean R² of 0.535, root mean squared error (RMSE) of 0.733, and mean absolute error (MAE) of 0.566. Because the predictors include tumor-volume and pathology variables, however, the experiment cannot validate screening or early-detection claims. Part II specifies a literature-grounded design for pre-diagnostic risk modeling with the Prostate, Lung, Colorectal and Ovarian Cancer Screening Trial (PLCO). The central argument is that claim-to-data alignment must precede algorithm selection. The study contributes a reproducible pilot analysis, leakage-safe validation, held-out interpretability, and a framework for separating post-diagnostic association from clinically meaningful risk estimation.

**Index Terms—**clinical prediction, data leakage, interpretability, machine learning, PLCO, prostate cancer, PSA, risk of bias.

## I. Introduction

Prostate cancer research includes several distinct prediction tasks. A model may estimate a laboratory value, identify cancer at biopsy, predict future diagnosis, estimate aggressiveness, or predict mortality after diagnosis. Each task has a different population, prediction time, outcome, and set of valid predictors. Variables collected after diagnostic workup may explain another post-diagnostic measurement, but they cannot support an early-detection claim because they would not exist when a screening decision is made.

The pilot dataset originates from a study of PSA among 97 men treated with radical prostatectomy [1]. It contains log cancer volume, log prostate weight, age, log benign prostatic hyperplasia, seminal vesicle invasion, log capsular penetration, Gleason score, percentage of Gleason pattern 4 or 5, and log PSA. These variables are useful for studying post-diagnostic associations, but several directly describe tumor burden or pathology.

Contemporary guidance makes the distinction between computational performance and clinical validity explicit. TRIPOD+AI requires transparent reporting of the target population, outcome, predictors, data preparation, validation, and intended use [2]. PROBAST+AI asks whether model development and evaluation create bias or limit applicability [3]. This paper applies those principles to a two-part design: Part I evaluates what the 97-patient dataset can responsibly support, and Part II defines the data structure needed for a pre-diagnostic prostate-cancer risk question.

### A. Research Question and Contributions

The primary research question is: How does the temporal and clinical structure of a dataset affect the reliability, interpretability, and clinical validity of machine-learning models for PSA prediction and prostate-cancer risk modeling?

The study makes four contributions. First, it provides a reproducible comparison of five algorithms under repeated, leakage-safe cross-validation. Second, it corrects a prior analysis flaw by excluding the dataset's `train` indicator from the predictors and fitting preprocessing only inside training folds. Third, it evaluates held-out permutation importance rather than training-set importance. Fourth, it defines a PLCO-based claim-to-data framework for a future pre-diagnostic study.

The thesis is that dataset structure determines the boundary of a valid machine-learning claim. Post-diagnostic tumor and pathology variables can support exploratory log-PSA modeling, but screening or biopsy-risk claims require longitudinal predictors recorded before diagnosis and clearly timed, biopsy-confirmed outcomes.

## II. Background and Related Work

### A. Clinical Prediction Methodology

A clinical prediction study should define the intended use, target population, prediction time, outcome, and candidate predictors before model fitting [4]. This sequence prevents algorithm choice from silently redefining the scientific question. It is especially important when database fields occur at different stages of care.

Small datasets create an additional threat. Healthcare AI models developed from limited samples may produce unstable individual predictions and optimistic estimates of utility [5]. For continuous outcomes, sample-size planning should consider shrinkage, optimism, and precision rather than relying on a simple observations-per-predictor rule [6]. The present 97-patient analysis is therefore framed as a pilot rather than a clinical model-development study.

Performance evaluation also requires more than a single accuracy statistic. Discrimination and calibration address different properties and should be reported together when a model estimates risk [7]. Calibration is particularly important because a model can rank patients correctly while systematically overestimating or underestimating absolute risk [8]. If a model will guide action at clinical thresholds, decision-curve analysis can evaluate net benefit rather than accuracy alone [9].

Risk-of-bias reviews show why these safeguards matter. Oncology machine-learning studies frequently have weaknesses in participant selection, sample size, analysis, and validation [10]. A systematic review of prostate-cancer models intended for primary care likewise found substantial variation in predictors, outcomes, validation, and clinical readiness [11]. These findings support conservative language and a clear separation between exploratory computation and clinical utility.

### B. PLCO and Prostate-Cancer Risk Models

PLCO is an NCI-sponsored randomized screening trial with approximately 155,000 participants and longitudinal screening, diagnosis, incidence, treatment, and mortality data [12]. The trial's prostate component provides repeated PSA and digital rectal examination (DRE) measures, follow-up testing, cancer diagnoses, grade information, and time-to-event fields.

PLCO's randomized screening results also show why model claims must be tied to clinical outcomes. Organized annual screening did not demonstrate a prostate-cancer mortality benefit after 13 years compared with usual care in the trial setting [13], and extended follow-up continued to show no significant mortality reduction between the randomized arms [14]. These findings do not imply that PSA has no predictive information; they show that prediction, screening policy, and mortality benefit are separate questions.

Several studies provide direct comparators for Part II. A model using PLCO and SELECT estimated one- to five-year risk of any and higher-grade prostate cancer and evaluated calibration across cohorts [15]. A longitudinal PLCO analysis of 34,756 participants examined DRE patterns and cancer detection, including racial differences [16]. An interpretable gradient-boosting model used diagnosed PLCO patients to predict ten-year prostate-cancer mortality [17]. These studies use different populations and outcomes, so their performance values are not interchangeable.

Established prostate-cancer calculators further demonstrate the value of pre-biopsy variables and external validation. The Prostate Cancer Prevention Trial model combined PSA, DRE, family history, previous biopsy, age, and race to estimate biopsy outcomes [18]. External evaluation showed that performance and calibration can change in a younger and more diverse screened population [19]. An external validation of the European Randomized Study of Screening for Prostate Cancer calculator similarly emphasized transportability across unscreened and clinical populations [20]. A future PLCO model should therefore be compared with existing calculators conceptually and, if feasible, empirically rather than presented as an algorithm-only novelty.

## III. Part I Methods

### A. Dataset and Study Role

Part I uses 97 records and predicts `lpsa`, the natural logarithm of PSA. Eight patient or tumor variables are retained as candidate predictors. The Boolean `train` field inherited from the source dataset is excluded because it records a historical data split rather than a patient characteristic. The target is also excluded from the predictor matrix.

This experiment is a post-diagnostic regression pilot. It does not estimate the probability of cancer, determine biopsy need, identify disease in an asymptomatic population, or establish clinical utility.

### B. Models and Implementation

Five pre-specified regressors are implemented in Python with scikit-learn [21]: ordinary least-squares linear regression, a depth-constrained decision tree, random forest, gradient boosting, and a small feed-forward neural network. Linear regression provides a transparent baseline. Random forests combine randomized trees and can represent nonlinear interactions while supporting permutation-based variable importance [22]. Gradient boosting fits additive learners sequentially to reduce prediction error [23]. The neural network uses back-propagation [24] but is intentionally small because the sample does not support a high-capacity architecture.

### C. Validation and Metrics

Each algorithm is evaluated with five-fold cross-validation repeated ten times using a fixed seed, producing 50 held-out evaluations per model. Every model uses the same partitions. Scaling for linear regression and the neural network occurs inside the pipeline and is fitted only on each training fold. Tree models do not require scaling. Hyperparameters are pre-specified rather than selected from held-out results.

Mean held-out RMSE is the primary ranking metric. MAE and R² are secondary metrics. Standard deviations across the 50 evaluations describe sensitivity to plausible data partitions; they are not treated as confidence intervals because repeated folds are correlated.

### D. Held-Out Interpretability and Reproducibility

The best model by mean RMSE is refitted in a separate shuffled five-fold procedure for permutation importance. Within each fold, a predictor is permuted only in held-out data, and the resulting increase in RMSE is recorded. The analysis reports mean importance, variability, and the fraction of permutations with positive importance.

The repository records fold-level metrics, summary metrics, raw and summarized permutation importance, excluded metadata, predictor names, random seeds, and the validation design. Automated tests verify that `train` and `lpsa` never enter the predictor matrix.

## IV. Part I Results

Linear regression ranks first by mean held-out RMSE. It achieves mean R² of 0.535, RMSE of 0.733, and MAE of 0.566. Random forest and gradient boosting rank second and third. The decision tree and neural network have lower mean R² and greater error.

| Model | Mean R² ± SD | Mean RMSE ± SD | Mean MAE ± SD |
|---|---:|---:|---:|
| Linear Regression | 0.535 ± 0.185 | 0.733 ± 0.124 | 0.566 ± 0.106 |
| Random Forest | 0.494 ± 0.124 | 0.776 ± 0.090 | 0.638 ± 0.077 |
| Gradient Boosting | 0.475 ± 0.140 | 0.790 ± 0.099 | 0.650 ± 0.083 |
| Decision Tree | 0.289 ± 0.215 | 0.914 ± 0.120 | 0.759 ± 0.103 |
| Neural Network | 0.212 ± 0.395 | 0.945 ± 0.217 | 0.742 ± 0.169 |

The ranking indicates that added flexibility does not improve held-out log-PSA prediction in this dataset. It should not be interpreted as proof that linear regression is universally superior; the result is conditional on these 97 observations, predictors, hyperparameters, and resampling design.

Held-out permutation importance identifies log cancer volume as the strongest and most consistent predictor. Its mean RMSE importance is 0.404 and is positive in every evaluation. Seminal vesicle invasion and log prostate weight rank next, with mean importance of 0.108 and 0.080. The remaining features have small or inconsistent held-out effects. These values describe predictive contribution within the dataset and do not establish causality.

## V. Discussion

The repeated-validation estimate is more conservative than the earlier single-split R² of approximately 0.75. The gap illustrates how a favorable partition can overstate model performance when a test set contains only about 20 patients. Repeated cross-validation does not create information or replace external validation, but it makes split sensitivity visible.

The linear model's first-place ranking is plausible because the dataset is small and contains a few strong, structured predictors. Random forest and gradient boosting remain competitive, but their extra flexibility does not compensate for limited sample information. The neural network has the largest R² variability, which is consistent with high variance under small-sample training.

More importantly, the feature ranking exposes the clinical boundary of the experiment. Tumor volume, seminal vesicle invasion, Gleason score, and capsular penetration are unavailable before diagnostic workup. A model using them may predict log PSA but cannot serve as an early-detection or screening model. Predictive accuracy cannot repair a mismatch between predictor timing and intended use.

## VI. Part II Proposed PLCO Study

### A. Objective and Estimand

The recommended primary objective is to estimate five-year risk of biopsy-confirmed prostate cancer among participants without prostate cancer at the prediction time. A secondary outcome may be high-grade cancer if the grade definition, event count, and data completeness are adequate. Dr. Husari should confirm the prediction time and horizon before data extraction because those choices determine cohort eligibility and valid predictors.

### B. Cohort, Predictors, and Outcomes

Eligible participants should have no prostate-cancer diagnosis before the prediction time and sufficient follow-up to determine the outcome or censoring status. Candidate predictors should be restricted to values recorded at or before an explicit cutoff, including age, race and ethnicity, family history when available, screening history, prior biopsy, longitudinal PSA measures, and longitudinal DRE results.

Biopsy linkage, diagnosis date, cancer status, Gleason grade, stage, and treatment fields should define outcomes or follow-up. They must not become baseline predictors. The data pipeline should preserve source timestamps and produce a variable-level audit indicating whether each field exists before, at, or after the prediction time.

### C. Analysis and Validation Plan

The analysis should first report cohort construction, missingness, follow-up, outcome prevalence, and predictor distributions overall and across clinically relevant groups. Model development should begin with a transparent regression baseline before adding nonlinear methods. Imputation, scaling, feature selection, and hyperparameter tuning must occur within resampling folds.

For a fixed-horizon binary outcome, performance reporting should include area under the receiver operating characteristic curve, precision-recall performance when the outcome is uncommon, Brier score, calibration intercept, calibration slope, and a calibration plot. Nested cross-validation or bootstrap optimism correction should be used when tuning is performed. Temporal, site-based, or external validation is preferred when feasible. Decision-curve analysis should be added only after a clinically meaningful threshold range is defined.

Subgroup estimates should include uncertainty and should not overinterpret small racial or ethnic strata. Comparison with PCPT- or ERSPC-style predictors would clarify whether a PLCO model adds value beyond established baselines.

### D. Claim-to-Data Audit

Every experiment should answer five questions: Who is the target population? When is the prediction made? Which variables are available at that moment? What outcome is observed, and over what period? What validation supports use beyond the development sample? This audit forms the conceptual link between the two thesis parts.

## VII. Limitations and Significance

Part I contains only 97 observations and eight valid predictors. Repeated cross-validation measures split sensitivity but cannot eliminate overfitting or support clinical deployment. The permutation results are associations, not causal effects. The dataset does not represent a screening population.

Part II remains a proposed research design. Feasibility depends on PLCO approval, field availability, missingness, event counts, and the thesis schedule. PLCO reflects the participants, screening practices, and historical period of the original trial, so transportability to current care requires explicit evaluation.

Despite these limitations, the two-part structure provides a useful computing contribution. It demonstrates that responsible machine learning involves more than selecting an algorithm or maximizing an accuracy metric. Problem formulation, time-aware data construction, reproducible evaluation, calibration, interpretability, and claim discipline jointly determine whether a result is scientifically meaningful.

## VIII. Conclusion

Linear regression currently provides the lowest repeated held-out error for log-PSA prediction in the 97-patient dataset, while the neural network is the least stable. The more important finding is not the model ranking but the boundary of the claim: the strongest predictors describe disease after diagnostic workup and therefore cannot validate screening.

A clinically meaningful prostate-cancer risk model requires pre-diagnostic predictors, an explicit prediction time, biopsy-confirmed follow-up, calibration assessment, and validation beyond the development sample. Dataset structure is therefore not merely an input to machine learning; it determines what scientific conclusion the model is permitted to support.

## References

[1] T. A. Stamey, J. N. Kabalin, J. E. McNeal, I. M. Johnstone, F. Freiha, E. A. Redwine, and N. Yang, “Prostate specific antigen in the diagnosis and treatment of adenocarcinoma of the prostate. II. Radical prostatectomy treated patients,” *J. Urol.*, vol. 141, no. 5, pp. 1076–1083, 1989, doi: 10.1016/S0022-5347(17)41175-X.

[2] G. S. Collins *et al.*, “TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods,” *BMJ*, vol. 385, e078378, 2024, doi: 10.1136/bmj-2023-078378.

[3] K. G. M. Moons *et al.*, “PROBAST+AI: An updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods,” *BMJ*, vol. 388, e082505, 2025, doi: 10.1136/bmj-2024-082505.

[4] O. Efthimiou *et al.*, “Developing clinical prediction models: A step-by-step guide,” *BMJ*, vol. 386, e078276, 2024, doi: 10.1136/bmj-2023-078276.

[5] R. D. Riley *et al.*, “Importance of sample size on the quality and utility of AI-based prediction models for healthcare,” *Lancet Digit. Health*, vol. 7, no. 6, art. 100857, 2025, doi: 10.1016/S2589-7500(25)00021-4.

[6] R. D. Riley, K. I. E. Snell, J. Ensor, D. L. Burke, F. E. Harrell, K. G. M. Moons, and G. S. Collins, “Minimum sample size for developing a multivariable prediction model: Part I—Continuous outcomes,” *Stat. Med.*, vol. 38, no. 7, pp. 1262–1275, 2019, doi: 10.1002/sim.7993.

[7] E. W. Steyerberg *et al.*, “Assessing the performance of prediction models: A framework for traditional and novel measures,” *Epidemiology*, vol. 21, no. 1, pp. 128–138, 2010, doi: 10.1097/EDE.0b013e3181c30fb2.

[8] B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, and E. W. Steyerberg, “Calibration: The Achilles heel of predictive analytics,” *BMC Med.*, vol. 17, art. 230, 2019, doi: 10.1186/s12916-019-1466-7.

[9] A. J. Vickers and E. B. Elkin, “Decision curve analysis: A novel method for evaluating prediction models,” *Med. Decis. Making*, vol. 26, no. 6, pp. 565–574, 2006, doi: 10.1177/0272989X06295361.

[10] P. Dhiman *et al.*, “Risk of bias of prognostic models developed using machine learning: A systematic review in oncology,” *Diagn. Progn. Res.*, vol. 6, art. 13, 2022, doi: 10.1186/s41512-022-00126-w.

[11] M. Aladwani *et al.*, “Prediction models for prostate cancer to be used in the primary care setting: A systematic review,” *BMJ Open*, vol. 10, no. 7, e034661, 2020, doi: 10.1136/bmjopen-2019-034661.

[12] National Cancer Institute, “Prostate, Lung, Colorectal and Ovarian Cancer Screening Trial,” Cancer Data Access System. [Online]. Available: https://cdas.cancer.gov/plco/. [Accessed: Jul. 31, 2026].

[13] G. L. Andriole *et al.*, “Prostate cancer screening in the randomized Prostate, Lung, Colorectal, and Ovarian Cancer Screening Trial: Mortality results after 13 years of follow-up,” *J. Natl. Cancer Inst.*, vol. 104, no. 2, pp. 125–132, 2012, doi: 10.1093/jnci/djr500.

[14] P. F. Pinsky *et al.*, “Extended mortality results for prostate cancer screening in the PLCO trial with median follow-up of 15 years,” *Cancer*, vol. 123, no. 4, pp. 592–599, 2017, doi: 10.1002/cncr.30474.

[15] J. A. Gelfond *et al.*, “Prediction of future risk of any and higher-grade prostate cancer based on the PLCO and SELECT trials,” *BMC Urol.*, vol. 22, art. 45, 2022, doi: 10.1186/s12894-022-00986-w.

[16] K. Guan *et al.*, “Ten-year trends in digital rectal exam results and prostate cancer detection: Insights from the PLCO Trial,” *Res. Rep. Urol.*, vol. 17, pp. 309–320, 2025, doi: 10.2147/RRU.S542550.

[17] J.-E. Bibault *et al.*, “Development and validation of an interpretable artificial intelligence model to predict 10-year prostate cancer mortality,” *Cancers*, vol. 13, no. 12, art. 3064, 2021, doi: 10.3390/cancers13123064.

[18] I. M. Thompson *et al.*, “Assessing prostate cancer risk: Results from the Prostate Cancer Prevention Trial,” *J. Natl. Cancer Inst.*, vol. 98, no. 8, pp. 529–534, 2006, doi: 10.1093/jnci/djj131.

[19] D. J. Parekh *et al.*, “External validation of the Prostate Cancer Prevention Trial risk calculator in a screened population,” *Urology*, vol. 68, no. 6, pp. 1152–1155, 2006, doi: 10.1016/j.urology.2006.10.022.

[20] H. A. van Vugt *et al.*, “Prediction of prostate cancer in unscreened men: External validation of a risk calculator,” *Eur. J. Cancer*, vol. 47, no. 6, pp. 903–909, 2011, doi: 10.1016/j.ejca.2010.11.012.

[21] F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[22] L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.

[23] J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” *Ann. Stat.*, vol. 29, no. 5, pp. 1189–1232, 2001, doi: 10.1214/aos/1013203451.

[24] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by back-propagating errors,” *Nature*, vol. 323, pp. 533–536, 1986, doi: 10.1038/323533a0.
