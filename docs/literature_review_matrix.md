# Literature Review Matrix

This matrix follows the citation numbering in `docs/thesis_draft.md`. References are ordered by first appearance, as required by IEEE style.

## Dataset, Reporting, and Clinical-Prediction Methods

| Ref. | Source | Role in the Paper |
|---:|---|---|
| [1] | Stamey et al., *J. Urol.*, 1989, doi: 10.1016/S0022-5347(17)41175-X | Primary source for the 97-patient prostatectomy dataset and its post-diagnostic context. |
| [2] | Collins et al., TRIPOD+AI, *BMJ*, 2024, doi: 10.1136/bmj-2023-078378 | Reporting framework for target population, outcomes, predictors, validation, and intended use. |
| [3] | Moons et al., PROBAST+AI, *BMJ*, 2025, doi: 10.1136/bmj-2024-082505 | Risk-of-bias and applicability framework for both thesis parts. |
| [4] | Efthimiou et al., *BMJ*, 2024, doi: 10.1136/bmj-2023-078276 | Stepwise design of clinical prediction studies before algorithm selection. |
| [5] | Riley et al., *Lancet Digit. Health*, 2025, doi: 10.1016/S2589-7500(25)00021-4 | Explains how inadequate sample size harms AI-model quality and utility. |
| [6] | Riley et al., *Stat. Med.*, 2019, doi: 10.1002/sim.7993 | Sample-size principles specifically for continuous-outcome prediction. |
| [7] | Steyerberg et al., *Epidemiology*, 2010, doi: 10.1097/EDE.0b013e3181c30fb2 | Framework for discrimination, calibration, reclassification, and clinical usefulness. |
| [8] | Van Calster et al., *BMC Med.*, 2019, doi: 10.1186/s12916-019-1466-7 | Establishes calibration as a distinct requirement from discrimination. |
| [9] | Vickers and Elkin, *Med. Decis. Making*, 2006, doi: 10.1177/0272989X06295361 | Original decision-curve analysis method for evaluating net clinical benefit. |
| [10] | Dhiman et al., *Diagn. Progn. Res.*, 2022, doi: 10.1186/s41512-022-00126-w | Documents recurring risk-of-bias problems in oncology machine learning. |
| [11] | Aladwani et al., *BMJ Open*, 2020, doi: 10.1136/bmjopen-2019-034661 | Systematic review of prostate-cancer models intended for primary care. |

## PLCO and Prostate-Cancer Risk Modeling

| Ref. | Source | Role in the Paper |
|---:|---|---|
| [12] | National Cancer Institute, PLCO Cancer Data Access System | Primary documentation for PLCO design, variables, follow-up, and access. |
| [13] | Andriole et al., *J. Natl. Cancer Inst.*, 2012, doi: 10.1093/jnci/djr500 | Thirteen-year randomized PLCO prostate-screening mortality results. |
| [14] | Pinsky et al., *Cancer*, 2017, doi: 10.1002/cncr.30474 | Extended PLCO mortality follow-up and screening interpretation. |
| [15] | Gelfond et al., *BMC Urol.*, 2022, doi: 10.1186/s12894-022-00986-w | Closest comparator for one- to five-year PLCO/SELECT prostate-cancer risk. |
| [16] | Guan et al., *Res. Rep. Urol.*, 2025, doi: 10.2147/RRU.S542550 | Dr. Husari's recommended PLCO study of longitudinal DRE patterns and race. |
| [17] | Bibault et al., *Cancers*, 2021, doi: 10.3390/cancers13123064 | Interpretable PLCO model for post-diagnosis ten-year mortality; a contrasting task. |
| [18] | Thompson et al., *J. Natl. Cancer Inst.*, 2006, doi: 10.1093/jnci/djj131 | Development of the PCPT biopsy-risk model using clinically available predictors. |
| [19] | Parekh et al., *Urology*, 2006, doi: 10.1016/j.urology.2006.10.022 | External validation of PCPT in a younger, more diverse screened cohort. |
| [20] | van Vugt et al., *Eur. J. Cancer*, 2011, doi: 10.1016/j.ejca.2010.11.012 | External validation of the ERSPC risk calculator in unscreened men. |

## Computational Foundations

| Ref. | Source | Role in the Paper |
|---:|---|---|
| [21] | Pedregosa et al., *J. Mach. Learn. Res.*, 2011 | Primary software citation for scikit-learn. |
| [22] | Breiman, *Mach. Learn.*, 2001, doi: 10.1023/A:1010933404324 | Foundational random-forest method and permutation importance. |
| [23] | Friedman, *Ann. Stat.*, 2001, doi: 10.1214/aos/1013203451 | Foundational gradient-boosting method. |
| [24] | Rumelhart et al., *Nature*, 1986, doi: 10.1038/323533a0 | Foundational back-propagation method for neural networks. |

## Selection Guidance

The core paper should retain all 24 references because each supports a claim actually made in the manuscript. Sources [1]–[11] establish dataset provenance and methodological validity; [12]–[20] ground the PLCO and prostate-cancer sections; [21]–[24] document the computational methods. MRI, radiomics, pathology-image, treatment-response, and postoperative-recurrence studies should be excluded unless the paper explicitly adds one of those prediction tasks.
