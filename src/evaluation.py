"""
Model Evaluation Module
========================

This module provides functions for evaluating model performance and
calculating various metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance using multiple metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model being evaluated
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2
    }
    
    return metrics


def print_evaluation(metrics):
    """
    Print evaluation metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from evaluate_model()
    """
    print("\n" + "="*70)
    print(f"📊 {metrics['model_name'].upper()} PERFORMANCE")
    print("="*70)
    print(f"\n📉 Mean Squared Error (MSE):  {metrics['mse']:.4f}")
    print(f"📉 Root Mean Squared Error:   {metrics['rmse']:.4f}")
    print(f"📉 Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"📈 R² Score:                  {metrics['r2_score']:.4f} ({metrics['r2_score']*100:.1f}%)")
    
    if metrics['r2_score'] > 0.7:
        print("\n🌟 Excellent! This is a strong model.")
    elif metrics['r2_score'] > 0.5:
        print("\n✅ Good! This model captures meaningful patterns.")
    else:
        print("\n📈 Room for improvement.")
    
    print("="*70)


def compare_models(results_list):
    """
    Compare multiple models and rank them by performance.
    
    Parameters:
    -----------
    results_list : list of dict
        List of metric dictionaries from evaluate_model()
        
    Returns:
    --------
    pd.DataFrame
        Comparison table sorted by R² score
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values('r2_score', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    df = df[['rank', 'model_name', 'mse', 'rmse', 'mae', 'r2_score']]
    
    return df


def get_prediction_summary(y_true, y_pred, n_samples=10):
    """
    Create a summary dataframe comparing actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    n_samples : int
        Number of samples to show
        
    Returns:
    --------
    pd.DataFrame
        Comparison of actual vs predicted values
    """
    comparison = pd.DataFrame({
        'Actual PSA': y_true[:n_samples] if hasattr(y_true, '__getitem__') else y_true.values[:n_samples],
        'Predicted PSA': y_pred[:n_samples],
        'Error': (y_true[:n_samples] if hasattr(y_true, '__getitem__') else y_true.values[:n_samples]) - y_pred[:n_samples]
    })
    
    return comparison
