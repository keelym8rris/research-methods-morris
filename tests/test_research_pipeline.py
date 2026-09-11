import pandas as pd

from src.data_loader import get_predictors_and_target, load_prostate_data, prepare_data
from src.research_analysis import build_models, evaluate_models_repeated_cv


def test_train_indicator_is_never_a_predictor():
    data = load_prostate_data()
    X, y = get_predictors_and_target(data)
    prepared = prepare_data(data)

    assert "train" in data.columns
    assert "train" not in X.columns
    assert "train" not in prepared["feature_names"]
    assert "lpsa" not in X.columns
    assert len(X) == len(y) == 97


def test_repeated_cv_compares_all_prespecified_models():
    data = load_prostate_data()
    X, y = get_predictors_and_target(data)
    small_models = {
        name: model
        for name, model in build_models().items()
        if name in {"Linear Regression", "Decision Tree"}
    }
    folds, summary = evaluate_models_repeated_cv(
        X,
        y,
        models=small_models,
        n_splits=3,
        n_repeats=2,
    )

    assert set(summary["model"]) == set(small_models)
    assert len(folds) == len(small_models) * 3 * 2
    assert pd.api.types.is_numeric_dtype(summary["rmse_mean"])
    assert summary["rank"].tolist() == [1, 2]
