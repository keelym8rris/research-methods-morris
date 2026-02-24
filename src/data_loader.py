"""
Data Loading and Preprocessing Module
======================================

This module handles all data loading, cleaning, and preprocessing operations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_prostate_data(filepath='data/prostate.csv'):
    """
    Load the prostate cancer dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing prostate cancer data
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with prostate cancer patient data
    """
    data = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {len(data)} patients, {data.shape[1]} features")
    return data


def prepare_data(data, test_size=0.2, random_state=42):
    """
    Prepare data for machine learning by splitting into train/test sets
    and scaling features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The prostate cancer dataset
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - X_train_scaled: Scaled training features
        - X_test_scaled: Scaled testing features
        - y_train: Training target values
        - y_test: Testing target values
        - scaler: Fitted StandardScaler object
        - feature_names: List of feature names
    """
    # Separate features from target
    X = data.drop(columns=['lpsa'])
    y = data['lpsa']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Training set: {len(X_train)} patients ({len(X_train)/len(X)*100:.1f}%)")
    print(f"🧪 Testing set: {len(X_test)} patients ({len(X_test)/len(X)*100:.1f}%)")
    
    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }


def get_feature_descriptions():
    """
    Return descriptions of all features in the dataset.
    
    Returns:
    --------
    dict
        Dictionary mapping feature names to their descriptions
    """
    return {
        'lcavol': 'Log of cancer volume',
        'lweight': 'Log of prostate weight',
        'age': 'Patient age (years)',
        'lbph': 'Log of benign prostatic hyperplasia amount',
        'svi': 'Seminal vesicle invasion (0=no, 1=yes)',
        'lcp': 'Log of capsular penetration',
        'gleason': 'Gleason score (cancer grading)',
        'pgg45': 'Percentage Gleason scores 4 or 5',
        'train': 'Training set indicator',
        'lpsa': 'Log PSA level (TARGET VARIABLE)'
    }
