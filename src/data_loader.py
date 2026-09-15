"""Load the prostate data and separate predictors from the outcome."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / 'data/prostate.csv'
TARGET_COLUMN = 'lpsa'
METADATA_COLUMNS = ('train',)


def load_prostate_data(filepath=DATA_PATH):
    """Read the repository CSV or a supplied path."""
    data = pd.read_csv(filepath)
    predictors, _ = get_predictors_and_target(data)
    print(f'Dataset loaded: {len(data)} records, {predictors.shape[1]} predictors')
    return data


def get_predictors_and_target(data, target_column=TARGET_COLUMN):
    """Exclude the source publication's split marker from patient predictors."""
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset")

    excluded = [target_column, *METADATA_COLUMNS]
    X = data.drop(columns=excluded, errors='ignore').copy()
    y = data[target_column].copy()
    if X.empty:
        raise ValueError('No predictor columns remain after removing target and metadata')
    return X, y
