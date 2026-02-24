"""
Linear Regression Model
=======================

Linear Regression is a simple but powerful baseline model that finds the best 
linear relationship between features and the target variable.

Advantages:
- Fast to train
- Easy to interpret
- Works well when relationships are roughly linear
- Shows which features are most important

When to use:
- As a baseline to compare other models against
- When model interpretability is important
- When you have limited data
"""

import sys
import numpy as np
from sklearn.linear_model import LinearRegression
sys.path.append('.')

from src.data_loader import load_prostate_data, prepare_data
from src.evaluation import evaluate_model, print_evaluation
from src.visualization import plot_predictions, plot_residuals, plot_feature_importance


def train_linear_regression(data_dict, verbose=True):
    """
    Train a Linear Regression model for PSA prediction.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary from prepare_data() containing training and test sets
    verbose : bool
        Whether to print detailed output
        
    Returns:
    --------
    tuple
        (model, predictions, metrics)
    """
    if verbose:
        print("\n" + "="*70)
        print("🎨 LINEAR REGRESSION MODEL")
        print("="*70)
        print("\nTraining linear regression model...")
    
    # Create and train the model
    model = LinearRegression()
    model.fit(data_dict['X_train_scaled'], data_dict['y_train'])
    
    if verbose:
        print("✅ Model trained successfully!")
    
    # Make predictions
    y_pred = model.predict(data_dict['X_test_scaled'])
    
    # Evaluate the model
    metrics = evaluate_model(data_dict['y_test'], y_pred, "Linear Regression")
    
    if verbose:
        print_evaluation(metrics)
    
    return model, y_pred, metrics


def analyze_coefficients(model, feature_names):
    """
    Analyze and display the coefficients learned by the linear regression model.
    
    Parameters:
    -----------
    model : LinearRegression
        Trained linear regression model
    feature_names : list
        Names of the features
    """
    import pandas as pd
    
    print("\n📊 Feature Coefficients (How each feature affects PSA):")
    print("="*70)
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print(coef_df.to_string(index=False))
    print("\n💡 Interpretation:")
    print("   • Positive coefficient → feature increases PSA prediction")
    print("   • Negative coefficient → feature decreases PSA prediction")
    print("   • Larger absolute value → stronger influence")
    print("="*70)
    
    return coef_df


if __name__ == "__main__":
    print("\n🚀 Running Linear Regression Analysis")
    print("="*70)
    
    # Load and prepare data
    data = load_prostate_data()
    data_dict = prepare_data(data)
    
    # Train model
    model, predictions, metrics = train_linear_regression(data_dict)
    
    # Analyze coefficients
    coef_df = analyze_coefficients(model, data_dict['feature_names'])
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    plot_feature_importance(
        data_dict['feature_names'], 
        model.coef_,
        "Linear Regression",
        save=True
    )
    
    plot_predictions(
        data_dict['y_test'],
        predictions,
        "Linear Regression",
        metrics['r2_score'],
        save=True
    )
    
    plot_residuals(
        data_dict['y_test'],
        predictions,
        "Linear Regression",
        save=True
    )
    
    print("\n✅ Linear Regression analysis complete!")
    print(f"🏆 Final R² Score: {metrics['r2_score']:.4f}")
