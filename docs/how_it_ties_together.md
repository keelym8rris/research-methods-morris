# How the thesis fits together

Your project now has one question connecting both chapters: **What can a model actually tell us, given the data it was trained on and when those data became available?**

## Part I: the work you can demonstrate now

You have 97 records with PSA and measurements that include tumor/pathology information. You compared models predicting **log PSA**, checked them on held-out records, and saved the predictions so someone else can check your calculations. The linear models perform similarly and have lower average error than the nonlinear configurations tested here.

That is a useful computing result. A more complex model did not automatically perform better. It is also a lesson about interpretation: an accurate estimate of PSA from information collected around diagnosis and surgery does not establish early cancer detection.

## Part II: what would need to change for future diagnosis

Dr. Husari's suggested PLCO paper helps introduce data collected before diagnosis. The review then asks which measurements would actually be available when a prediction is made, what future outcome is being predicted, and how it would be verified. Those questions lead to your proposed five-year recorded-diagnosis study.

This chapter is currently a review and study design. It does not report a second trained model. That distinction protects the paper from promising results your repository does not contain.

## How to talk through the figures

| Figure | What to say | What it does not establish |
|---|---|---|
| 1: model performance | “The three linear models are very close. These dots show how error changes across held-out splits.” | A statistically proven winner or cancer-detection accuracy. |
| 2: predictions/residuals | “These are 97 held-out predictions from the first repeat. We can see the size and pattern of errors.” | Calibration of cancer-risk probabilities. |
| 3: permutation importance | “The linear model relies most on cancer volume when predicting log PSA.” | That cancer volume causes a particular PSA change. |
| 4: coefficient variation | “These are conditional associations across fitted folds; some vary more than others.” | Independent replications or causal treatment effects. |

## A short explanation in your voice

“I started by comparing models, but the bigger issue became whether the data supported the claim I was making. My first part shows the analysis I can actually reproduce with the dataset I have. My second part explains what would need to change before I could study future diagnosis. I want the model results and the clinical conclusion to match.”

## Reading order for the meeting

1. [The paper Dr. Husari suggested](https://pmc.ncbi.nlm.nih.gov/articles/PMC12405719/): understand its diagnosis-relative DRE analysis and its limits.
2. [The PLCO/SELECT risk study](https://link.springer.com/article/10.1186/s12894-022-00986-w): an example of a future-risk question evaluated in another cohort.
3. [TRIPOD+AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/): use it to check what a model-development paper should report.
4. [The ACM leakage paper](https://dl.acm.org/doi/10.1145/2382577.2382579): the cited definition helps explain why information availability matters. Full text may require institutional access.
5. [Source audit](source_claim_audit.md): use this to locate supporting passages and see which sources were inspected only at abstract level.

## What to bring and decide

Bring the revised Word draft and the draft PR with code, outputs, figures and the source audit. Ask whether the completed methods chapter plus literature review/protocol meets the thesis scope. If he wants a new PLCO experiment, agree on a feasible access timeline and endpoint before extending the code. Also confirm final thesis formatting and submission requirements.
