# Claim-to-source audit

Reviewed September 11, 2026. Numbers match `thesis_draft.md` and `references.json`.

This is a focused source verification, not an exhaustive systematic review or a certification of the research. Preference was given to original research, methodological papers, the original teaching-data host, and official NCI documentation. Peer review and a DOI identify a publication; neither makes every statement in it correct. IEEE describes the numbered citation style used here. The ACM paper was included for its relevance to leakage, not merely its publisher.

Each row below identifies the passage or documentation supporting the use in the draft. Claims about this repository come from its own saved results, not from these papers. Paraphrases are not presented as verbatim quotations. Where an inference goes beyond a reported finding, the draft describes it as interpretation or a proposed design.

| Ref. and direct reading link | Evidence inspected and locator | Supported use and boundary |
|---|---|---|
| [1] [Stamey et al., 1989](https://pubmed.ncbi.nlm.nih.gov/2468795/) | PubMed abstract, PMID 2468795; DOI metadata checked. Full article not inspected. | Original study: 102 men, preoperative PSA, prostatectomy specimens. Does not establish why the teaching file contains 97 records. |
| [2] [Hastie, prostate data information](https://hastie.su.domains/ElemStatLearn/datasets/prostate.info.txt) and [data](https://hastie.su.domains/ElemStatLearn/datasets/prostate.data) | Official teaching-data description and downloaded file; compared numeric columns and split indicator. | 97 distributed observations, eight predictors, log PSA, 67/30 split. This is dataset documentation, not a clinical validation paper. |
| [3] [TRIPOD+AI, BMJ 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/) | Full article/checklist: participants, predictors, outcome, evaluation, open science reporting items. DOI metadata checked. | Reporting framework. No completed checklist or TRIPOD certification is claimed. |
| [4] [PROBAST+AI, BMJ 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11931409/) | Full article, development/evaluation assessment domains and applicability. DOI metadata checked. | Framework for quality and risk-of-bias discussion; not a formal scored assessment of this thesis. |
| [5] [Kaufman et al., ACM TKDD 2012](https://dl.acm.org/doi/10.1145/2382577.2382579) | Publisher-deposited abstract and bibliographic metadata via Crossref. ACM full text was inaccessible during audit. | Narrow definition of leakage as illegitimate information about a target. No detailed algorithm or empirical result from the unseen full text is cited. |
| [6] [Riley et al., Statistics in Medicine 2019](https://doi.org/10.1002/sim.7993) | Abstract and bibliographic metadata; full text not inspected. | Continuous-outcome sample-size planning concerns overfitting and precision. No universal threshold or claim that 97 observations meets the method. |
| [7] [Pedregosa et al., JMLR 2011](https://jmlr.org/papers/v12/pedregosa11a.html) | Official JMLR article record/abstract. | Attribution for scikit-learn. Installed package versions and exact estimator settings are documented separately in the repository. |
| [8] [Guan et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12405719/) | Full text: participant selection, DRE visit/time alignment, GEE results. DOI metadata checked. | 34,756 participants; four visits; interaction OR 1.230. Ten-year alignment is not ten annual exams. Prospective-model limitation is this draft's interpretation of the design. |
| [9] [Gelfond et al., BMC Urology 2022](https://link.springer.com/article/10.1186/s12894-022-00986-w) | Full text: external validation results and discussion of SABOR events and ascertainment. DOI metadata checked. | SABOR n=1,790; C-indices 0.76/0.74; only 22 higher-grade events. Endpoint is Gleason >7. Not evidence that our regression model screens for cancer. |
| [10] [Bibault et al., Cancers 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8234681/) | Full text: cohort, predictors, split and outcome. DOI metadata checked. | 8,776 diagnosed patients; ten-year mortality; treatment predictor. Prognosis, not pre-diagnostic detection. Timing concern is our interpretation. |
| [11] [NCI prostate datasets](https://cdas.cancer.gov/datasets/plco/20/) | Official dataset/access documentation. | PLCO resources and controlled access. No participant-level access or analysis occurred in this work. |
| [12] [NCI person dictionary](https://cdas.cancer.gov/files/download/pcvhtvubq8/pros-dictionary-t20241011.pdf) | October 2024 dictionary: p.4 identifier; pp.13–16 center, screens, biopsy linkage and diagnosis fields. | Documents field availability, not that a usable analysis cohort or linked endpoint has been built. |
| [13] [NCI procedure dictionary](https://cdas.cancer.gov/files/download/k1vmy6z99q/pros_proc-dictionary-t20241011.pdf) | October 2024 dictionary: p.4 biopsy/date; p.7 procedure result and form-version limitation. | A biopsy flag is not a biopsy result. Missing procedure results cannot be recoded as negative. |
| [14] [Efthimiou et al., BMJ 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11369751/) | Full text: intended use, outcomes, missing data, development and evaluation guidance. DOI metadata checked. | Supports protocol principles. The proposed five-year landmark design remains this project's plan. |
| [15] [Van Calster et al., BMC Medicine 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6912996/) | Full text: distinction between discrimination and calibration; assessment discussion. DOI metadata checked. | Ranking performance does not establish accurate absolute risk. No cancer-risk calibration experiment exists in this repository. |
| [16] [Vickers and Elkin, 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC2577036/) | Full text: threshold probability, net benefit, comparison of strategies. DOI metadata checked. | Rationale for a future decision-curve analysis; no net-benefit result is claimed. |

## Repository evidence behind the numerical claims

| Draft claim | Reproducible evidence |
|---|---|
| Sample summaries and no missing/duplicate observations | `results/audited/descriptive_statistics.csv`, `run_metadata.json`; `data/prostate.csv` |
| Eight-model table and 36.4% lower OLS RMSE than mean baseline | `results/audited/model_cv_summary.csv`; arithmetic from its unrounded RMSE means |
| 50 evaluations/model; 7,760 repeated held-out predictions | `fold_metrics.csv`, `oof_predictions.csv`, `fold_assignments.json` |
| Feature reliance and coefficient variation | `permutation_importance_raw.csv`, `coefficients_by_fold.csv`; generated figures 3 and 4 |
| Original five-model result reproduced | Preserved `results/research/model_cv_summary.csv`; audit rerun agrees within absolute tolerance 0.000000001 |

## Citation integrity decisions

The earlier 24-item list was narrowed to 16 sources tied to actual statements. Removed references are not thereby declared false; unused, redundant, or insufficiently inspected material was omitted. No secondary search summary or Wikipedia article is cited. No IEEE-published article was added merely to satisfy a publisher label. The ACM article is real, but its use is deliberately restricted to the accessible abstract.

A narrative review can miss relevant research. Before claiming novelty or submitting to a venue, expand the search with documented databases, terms, dates, and inclusion criteria. Recheck any added claim against its actual source passage.
