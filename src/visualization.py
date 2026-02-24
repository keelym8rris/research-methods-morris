"""
Visualization Module
====================

This module contains all visualization functions for exploratory data analysis
and model evaluation. All graphs can be saved for use in research papers.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Set default style
sns.set_theme(style='whitegrid', font='serif')
plt.rcParams['figure.figsize'] = (10, 6)


def save_figure(filename, dpi=300):
    """
    Save the current figure to the results/figures directory.
    
    Parameters:
    -----------
    filename : str
        Name of the file (without path)
    dpi : int
        Resolution for saving (default: 300 for publication quality)
    """
    os.makedirs('results/figures', exist_ok=True)
    filepath = os.path.join('results', 'figures', filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"💾 Figure saved: {filepath}")


def plot_data_distribution(data, save=True):
    """
    Plot the distribution of the target variable (PSA levels).
    
    Parameters:
    -----------
    data : pd.DataFrame
        The prostate cancer dataset
    save : bool
        Whether to save the figure
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(data['lpsa'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Log PSA Level', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.title('Distribution of PSA Levels\n(Target Variable)', fontsize=13, weight='bold')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(data['lpsa'], vert=True)
    plt.ylabel('Log PSA Level', fontsize=12)
    plt.title('PSA Levels - Box Plot\n(Shows median and outliers)', fontsize=13, weight='bold')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        save_figure('psa_distribution.png')
    
    plt.show()


def plot_correlation_heatmap(data, save=True):
    """
    Create a correlation heatmap showing relationships between all variables.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The prostate cancer dataset
    save : bool
        Whether to save the figure
    """
    plt.figure(figsize=(12, 10))
    
    # Remove 'train' column if it exists
    plot_data = data.drop(columns=['train']) if 'train' in data.columns else data
    correlation_matrix = plot_data.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap\n(Red=Positive Correlation, Blue=Negative)', 
              fontsize=14, pad=20, weight='bold')
    plt.tight_layout()
    
    if save:
        save_figure('correlation_heatmap.png')
    
    plt.show()


def plot_feature_relationships(data, save=True):
    """
    Plot relationships between key features and PSA levels.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The prostate cancer dataset
    save : bool
        Whether to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    features_to_plot = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'gleason']
    feature_names = ['Cancer Volume', 'Prostate Weight', 'Age', 
                     'BPH Amount', 'Seminal Vesicle Invasion', 'Gleason Score']
    
    for i, (feature, name) in enumerate(zip(features_to_plot, feature_names)):
        row, col = i // 3, i % 3
        axes[row, col].scatter(data[feature], data['lpsa'], alpha=0.6, 
                              color='coral', edgecolors='black', linewidth=0.5)
        axes[row, col].set_xlabel(name, fontsize=11)
        axes[row, col].set_ylabel('PSA Level', fontsize=11)
        axes[row, col].set_title(f'{name} vs PSA', fontsize=11, weight='bold')
        axes[row, col].grid(alpha=0.3)
    
    plt.suptitle('Feature Relationships with PSA Levels', fontsize=16, y=0.995, weight='bold')
    plt.tight_layout()
    
    if save:
        save_figure('feature_relationships.png')
    
    plt.show()


def plot_predictions(y_true, y_pred, model_name, r2_score, save=True):
    """
    Plot predicted vs actual values for a model.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
    r2_score : float
        R² score of the model
    save : bool
        Whether to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=100, edgecolors='black', linewidth=1)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', lw=2, label='Perfect Prediction Line')
    plt.xlabel('Actual PSA Levels', fontsize=12)
    plt.ylabel('Predicted PSA Levels', fontsize=12)
    plt.title(f'{model_name}: Predicted vs Actual PSA\nR² = {r2_score:.4f}', 
              fontsize=14, pad=20, weight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save:
        filename = f'{model_name.lower().replace(" ", "_")}_predictions.png'
        save_figure(filename)
    
    plt.show()


def plot_residuals(y_true, y_pred, model_name, save=True):
    """
    Plot residual analysis for a model.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
    save : bool
        Whether to save the figure
    """
    residuals = y_true - y_pred if hasattr(y_true, '__sub__') else y_true.values - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=80)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted PSA', fontsize=11)
    axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=11)
    axes[0].set_title(f'{model_name}: Residual Plot\n(Should be randomly scattered)', 
                     fontsize=12, weight='bold')
    axes[0].grid(alpha=0.3)
    
    # Distribution of errors
    axes[1].hist(residuals, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual (Error)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'{model_name}: Error Distribution\n(Should be centered at 0)', 
                     fontsize=12, weight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = f'{model_name.lower().replace(" ", "_")}_residuals.png'
        save_figure(filename)
    
    plt.show()


def plot_feature_importance(feature_names, importances, model_name, save=True):
    """
    Plot feature importance for tree-based or linear models.
    
    Parameters:
    -----------
    feature_names : list
        Names of features
    importances : array-like
        Importance scores for each feature
    model_name : str
        Name of the model
    save : bool
        Whether to save the figure
    """
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', key=abs, ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in feature_df['Importance']]
    plt.barh(feature_df['Feature'], feature_df['Importance'], 
             color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'{model_name}: Feature Importance\n(Green=Increases PSA, Red=Decreases PSA)', 
              fontsize=14, pad=20, weight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save:
        filename = f'{model_name.lower().replace(" ", "_")}_importance.png'
        save_figure(filename)
    
    plt.show()


def plot_model_comparison(comparison_df, save=True):
    """
    Create comparison visualizations for all models.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame with model comparison results
    save : bool
        Whether to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # MSE Comparison
    model_order_mse = comparison_df.sort_values('mse')
    colors_mse = ['gold' if i == 0 else 'silver' if i == 1 else 'chocolate' if i == 2 else 'lightgray' 
                  for i in range(len(model_order_mse))]
    bars1 = axes[0].barh(model_order_mse['model_name'], model_order_mse['mse'], 
                         color=colors_mse, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Mean Squared Error (Lower is Better)', fontsize=12)
    axes[0].set_ylabel('Model', fontsize=12)
    axes[0].set_title('Model Comparison: Mean Squared Error', fontsize=13, 
                     pad=15, weight='bold')
    axes[0].grid(alpha=0.3, axis='x')
    
    # R² Comparison
    model_order_r2 = comparison_df.sort_values('r2_score', ascending=True)
    colors_r2 = ['gold' if x == model_order_r2['r2_score'].max() 
                 else 'silver' if x == model_order_r2['r2_score'].iloc[-2] 
                 else 'chocolate' if x == model_order_r2['r2_score'].iloc[-3] 
                 else 'lightgray' 
                 for x in model_order_r2['r2_score']]
    bars2 = axes[1].barh(model_order_r2['model_name'], model_order_r2['r2_score'], 
                         color=colors_r2, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('R² Score (Higher is Better)', fontsize=12)
    axes[1].set_ylabel('Model', fontsize=12)
    axes[1].set_title('Model Comparison: R-squared', fontsize=13, 
                     pad=15, weight='bold')
    axes[1].grid(alpha=0.3, axis='x')
    axes[1].set_xlim(0, 1)
    
    plt.suptitle('Model Performance Comparison\n(Gold = 1st, Silver = 2nd, Bronze = 3rd)', 
                 fontsize=15, y=1.02, weight='bold')
    plt.tight_layout()
    
    if save:
        save_figure('model_comparison.png')
    
    plt.show()


def plot_all_predictions_grid(y_true, predictions_dict, save=True):
    """
    Create a grid of prediction plots for all models.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    predictions_dict : dict
        Dictionary with model names as keys and (predictions, r2_score) tuples as values
    save : bool
        Whether to save the figure
    """
    n_models = len(predictions_dict)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    colors = ['blue', 'green', 'darkgreen', 'orange', 'purple']
    
    for idx, ((model_name, (y_pred, r2)), color) in enumerate(zip(predictions_dict.items(), colors)):
        if idx < len(axes):
            axes[idx].scatter(y_true, y_pred, alpha=0.6, s=60, 
                            color=color, edgecolors='black', linewidth=0.5)
            axes[idx].plot([y_true.min(), y_true.max()], 
                          [y_true.min(), y_true.max()], 
                          'r--', lw=2, alpha=0.8)
            axes[idx].set_xlabel('Actual PSA', fontsize=10)
            axes[idx].set_ylabel('Predicted PSA', fontsize=10)
            axes[idx].set_title(f'{model_name}\nR² = {r2:.4f}', 
                              fontsize=11, weight='bold')
            axes[idx].grid(alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('All Models: Actual vs Predicted PSA Levels', 
                 fontsize=16, y=0.995, weight='bold')
    plt.tight_layout()
    
    if save:
        save_figure('all_models_grid.png')
    
    plt.show()
