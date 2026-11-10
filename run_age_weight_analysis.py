"""Ridge comparison using only age and log prostate weight."""

from pathlib import Path
from datetime import date
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error

from run_thesis_analysis import FEATURES, SEED, thesis_models
from src.data_loader import get_predictors_and_target

ROOT = Path(__file__).resolve().parent
REDUCED_FEATURES = ['age', 'lweight']


def run(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = ROOT / 'data/prostate.csv'
    data = pd.read_csv(data_path)
    X, y = get_predictors_and_target(data)
    if len(data) != 97 or list(X.columns) != FEATURES:
        raise ValueError('Expected the original 97-record thesis dataset')
    if data.isna().any().any() or not np.isfinite(data.select_dtypes('number')).all().all():
        raise ValueError('Missing or nonfinite values require review')
    if data.duplicated().any():
        raise ValueError('Duplicate records require review')

    splits = list(RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED).split(X))
    saved = json.loads((ROOT / 'results/thesis/fold_assignments.json').read_text())
    assignments = [
        {'repeat': k // 5 + 1, 'fold': k % 5 + 1,
         'train_rows': (train + 1).tolist(), 'test_rows': (test + 1).tolist()}
        for k, (train, test) in enumerate(splits)
    ]
    if assignments != saved:
        raise ValueError('Folds must match the original thesis comparison')

    ridge = thesis_models()['Ridge']
    specifications = {
        'Ridge age and weight': (ridge, REDUCED_FEATURES),
        'Ridge all predictors': (ridge, FEATURES),
        'Mean baseline': (thesis_models()['Mean baseline'], REDUCED_FEATURES),
    }
    fold_rows, prediction_rows = [], []
    for k, (train, test) in enumerate(splits):
        repeat, fold = k // 5 + 1, k % 5 + 1
        for name, (estimator, columns) in specifications.items():
            model = clone(estimator)
            model.fit(X.iloc[train][columns], y.iloc[train])
            prediction = model.predict(X.iloc[test][columns])
            fold_rows.append({
                'model': name, 'repeat': repeat, 'fold': fold, 'n_test': len(test),
                'rmse': float(mean_squared_error(y.iloc[test], prediction) ** 0.5),
            })
            for row, value in zip(test, prediction):
                prediction_rows.append({
                    'model': name, 'repeat': repeat, 'fold': fold,
                    'row_id': int(row + 1), 'observed': float(y.iloc[row]),
                    'predicted': float(value),
                })

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = folds.groupby('model', sort=False).agg(
        rmse_mean=('rmse', 'mean'), rmse_sd=('rmse', 'std'),
        evaluations=('rmse', 'size'),
    ).reset_index()
    original = pd.read_csv(ROOT / 'results/thesis/model_cv_summary.csv').set_index('model')
    comparison = summary.set_index('model')
    for name, old_name in [('Ridge all predictors', 'Ridge'), ('Mean baseline', 'Mean baseline')]:
        for metric in ['rmse_mean', 'rmse_sd']:
            if not np.isclose(comparison.loc[name, metric], original.loc[old_name, metric], atol=1e-9, rtol=0):
                raise ValueError('Original ridge and baseline results must be reproduced')

    for (_, _), group in predictions.groupby(['model', 'repeat']):
        if len(group) != 97 or group.row_id.nunique() != 97:
            raise ValueError('Each record needs one held-out prediction per repeat')
    options = {'index': False, 'float_format': '%.12g'}
    folds.to_csv(output / 'fold_metrics.csv', **options)
    predictions.to_csv(output / 'oof_predictions.csv', **options)
    summary.to_csv(output / 'model_cv_summary.csv', **options)
    metadata = {
        'analysis_date': date.today().isoformat(),
        'dataset_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'patients': len(data), 'predictors': REDUCED_FEATURES,
        'excluded_predictors': [c for c in FEATURES if c not in REDUCED_FEATURES],
        'target': 'natural-log PSA (lpsa)',
        'model': 'StandardScaler followed by Ridge(alpha=1.0)',
        'scaling': 'fitted separately on each training fold',
        'validation': 'same 5 folds x 10 repeats as the main thesis comparison',
        'stratified': False, 'seed': SEED,
        'summary': 'unweighted mean and sample SD of 50 held-out fold RMSE values',
        'original_results_reproduced': True, 'hyperparameter_search': False,
        'interpretation': 'Exploratory reduced-predictor log-PSA regression; not early cancer detection. Preoperative availability of lweight has not been established.',
        'python': platform.python_version(), 'numpy': np.__version__,
        'pandas': pd.__version__, 'sklearn': sklearn.__version__,
    }
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(summary.to_string(index=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/age_weight')
    run(parser.parse_args().output_dir)
