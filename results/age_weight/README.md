# Age and prostate weight ridge analysis

This analysis keeps `age` and `lweight`, the stored log prostate weight, and predicts `lpsa` with the thesis ridge pipeline (`StandardScaler` and `Ridge(alpha=1.0)`). The other six predictors and the historical `train` marker are excluded from the reduced model.

The 97 records use the same five folds repeated ten times, with seed 42, as the main thesis comparison. Scaling is fitted on training records only. The folds are shuffled, not stratified. No hyperparameter search was performed.

| Model | Mean fold RMSE | SD of fold RMSE |
| --- | ---: | ---: |
| Ridge using age and log prostate weight | 1.058913 | 0.150679 |
| Ridge using all eight predictors | 0.731483 | 0.121869 |
| Training-mean baseline | 1.153646 | 0.151607 |

RMSE is in natural-log PSA units. These are unweighted averages across the 50 held-out folds. The SD describes split variability, not a confidence interval. The reduced model's mean RMSE is 8.2% lower than the baseline's and 44.8% higher than full ridge's.

## Saved files

- `fold_metrics.csv`: 150 fold RMSE values across the three comparisons.
- `oof_predictions.csv`: 2,910 held-out predictions, including 970 for the reduced model. These reuse the same 97 patients across ten repeats.
- `model_cv_summary.csv`: mean and sample SD of fold RMSE, with evaluation counts.
- `run_metadata.json`: predictors, exclusions, package versions, data and script hashes, and validation details.

The runner verifies that its partitions exactly match `results/thesis/fold_assignments.json` and that full ridge and the baseline reproduce the saved thesis RMSE summaries. An additional check recalculated all fold RMSE values from the saved predictions, confirmed no training/test overlap, and checked that baseline predictions used only the training-fold outcomes.

Removing the other measurements reduces prediction quality in this sample. Two predictors are not inherently inadequate, but these two leave much of the log-PSA variation unexplained. The source documentation calls `lweight` log prostate weight without establishing preoperative measurement timing. A prostate-size estimate from imaging cannot be assumed equivalent to the stored measurement. This result supports follow-up work with verified predictor timing and an appropriate screening cohort; it does not validate early cancer detection.
