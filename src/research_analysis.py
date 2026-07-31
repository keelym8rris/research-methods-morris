"""Leakage-safe, reproducible analysis for the 97-patient pilot dataset.

This module supports Part 1 of the thesis. It compares five regression
algorithms with repeated cross-validation and estimates held-out permutation
importance for the best-performing model. The outcome is log PSA (``lpsa``),
not cancer diagnosis or screening risk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import get_predictors_and_target, load_prostate_data


RANDOM_STATE = 42


def build_models(random_state=RANDOM_STATE):
    """Return pre-specified models with preprocessing kept inside each fold."""
    return {
        "Linear Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=5,
            random_state=random_state,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=5,
            random_state=random_state,
        ),
        "Neural Network": TransformedTargetRegressor(
            regressor=Pipeline([
                ("scale", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=(16,),
                    activation="relu",
                    alpha=1.0,
                    early_stopping=True,
                    validation_fraction=0.2,
                    max_iter=2000,
                    random_state=random_state,
                )),
            ]),
            transformer=StandardScaler(),
        ),
    }


def _fold_metrics(model_name, repeat, fold, y_true, y_pred):
    return {
        "model": model_name,
        "repeat": repeat,
        "fold": fold,
        "n_test": len(y_true),
        "r2": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
    }


def evaluate_models_repeated_cv(
    X,
    y,
    models=None,
    n_splits=5,
    n_repeats=10,
    random_state=RANDOM_STATE,
):
    """Evaluate every model using the same repeated K-fold partitions."""
    models = models or build_models(random_state)
    splitter = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    splits = list(splitter.split(X, y))
    records = []

    for model_name, estimator in models.items():
        for split_index, (train_idx, test_idx) in enumerate(splits):
            repeat = split_index // n_splits + 1
            fold = split_index % n_splits + 1
            model = clone(estimator)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions = model.predict(X.iloc[test_idx])
            records.append(
                _fold_metrics(
                    model_name,
                    repeat,
                    fold,
                    y.iloc[test_idx],
                    predictions,
                )
            )

    fold_results = pd.DataFrame(records)
    summary = (
        fold_results.groupby("model", sort=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            evaluations=("r2", "size"),
        )
        .reset_index()
        .sort_values(["rmse_mean", "mae_mean"], ascending=True)
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return fold_results, summary


def held_out_permutation_importance(
    X,
    y,
    estimator,
    n_splits=5,
    n_repeats=30,
    random_state=RANDOM_STATE,
):
    """Aggregate permutation importance measured only on held-out folds."""
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        importance = permutation_importance(
            model,
            X.iloc[test_idx],
            y.iloc[test_idx],
            scoring="neg_root_mean_squared_error",
            n_repeats=n_repeats,
            random_state=random_state + fold,
            n_jobs=-1,
        )
        for feature, values in zip(X.columns, importance.importances):
            for iteration, value in enumerate(values, start=1):
                rows.append({
                    "fold": fold,
                    "iteration": iteration,
                    "feature": feature,
                    "importance": value,
                })

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("feature")
        .agg(
            importance_mean=("importance", "mean"),
            importance_sd=("importance", "std"),
            positive_fraction=("importance", lambda values: float((values > 0).mean())),
        )
        .reset_index()
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return raw, summary


def run_analysis(data_path="data/prostate.csv", output_dir="results/research"):
    """Run the thesis pilot analysis and save reproducible tabular outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = load_prostate_data(data_path)
    X, y = get_predictors_and_target(data)
    models = build_models()

    folds, model_summary = evaluate_models_repeated_cv(X, y, models=models)
    best_model_name = model_summary.iloc[0]["model"]
    importance_raw, importance_summary = held_out_permutation_importance(
        X,
        y,
        models[best_model_name],
    )

    # Limit serialized precision so identical seeded runs do not create noisy
    # diffs from platform-level floating-point summation order.
    csv_options = {"index": False, "float_format": "%.12g"}
    folds.to_csv(output_path / "fold_metrics.csv", **csv_options)
    model_summary.to_csv(output_path / "model_cv_summary.csv", **csv_options)
    importance_raw.to_csv(output_path / "permutation_importance_raw.csv", **csv_options)
    importance_summary.to_csv(output_path / "permutation_importance_summary.csv", **csv_options)

    metadata = {
        "study_role": "post-diagnostic pilot; not a screening or diagnostic model",
        "outcome": "lpsa (log PSA)",
        "patients": int(len(data)),
        "predictors": list(X.columns),
        "excluded_metadata": [column for column in ["train"] if column in data.columns],
        "validation": "5-fold cross-validation repeated 10 times",
        "selection_metric": "mean held-out RMSE",
        "best_model": best_model_name,
        "random_state": RANDOM_STATE,
    }
    (output_path / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return model_summary, importance_summary, metadata
