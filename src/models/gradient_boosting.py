"""
Gradient Boosting Model
=======================

Gradient Boosting is a powerful ensemble method that builds trees sequentially,
where each new tree corrects the errors of previous trees.

How it works:
1. Build tree #1 → Make predictions → Calculate errors
2. Build tree #2 to predict those errors
3. Combine tree #1 and tree #2
4. Build tree #3 to predict remaining errors
5. Repeat until errors are minimized

Advantages:
- Often very accurate (wins many ML competitions!)
- Good at learning complex patterns
- Provides feature importance
- Can handle different types of features

Disadvantages:
- Slower to train (trees built sequentially)
- More prone to overfitting than Random Forest
- Requires careful tuning of hyperparameters
- More sensitive to noisy data

When to use:
- When you want maximum accuracy
- When you have time to tune hyperparameters
- For structured/tabular data (often outperforms neural networks)
"""

import sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
sys.path.append('.')

from src.data_loader import load_prostate_data, prepare_data
from src.evaluation import evaluate_model, print_evaluation
from src.visualization import plot_predictions, plot_residuals, plot_feature_importance


def train_gradient_boosting(data_dict, n_estimators=100, learning_rate=0.1, 
                           max_depth=3, random_state=42, verbose=True):
    """
    Train a Gradient Boosting model for PSA prediction.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary from prepare_data() containing training and test sets
    n_estimators : int
        Number of boosting stages (trees)
    learning_rate : float
        Learning rate shrinks contribution of each tree
    max_depth : int
        Maximum depth of individual trees
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print detailed output
        
    Returns:
    --------
    tuple
        (model, predictions, metrics)
    """
    if verbose:
        print("\n" + "="*70)
        print("🚀 GRADIENT BOOSTING MODEL")
        print("="*70)
        print(f"\nTraining gradient boosting with {n_estimators} stages...")
        print("(This learns sequentially, so may take a moment...)")
    
    # Create and train the model
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(data_dict['X_train_scaled'], data_dict['y_train'])
    
    if verbose:
        print("✅ Gradient boosting trained successfully!")
    
    # Make predictions
    y_pred = model.predict(data_dict['X_test_scaled'])
    
    # Evaluate the model
    metrics = evaluate_model(data_dict['y_test'], y_pred, "Gradient Boosting")
    
    if verbose:
        print_evaluation(metrics)
    
    return model, y_pred, metrics


def analyze_boosting(model, feature_names):
    """
    Analyze and display information about the gradient boosting model.
    
    Parameters:
    -----------
    model : GradientBoostingRegressor
        Trained gradient boosting model
    feature_names : list
        Names of the features
    """
    import pandas as pd
    
    print("\n🚀 Gradient Boosting Details:")
    print("="*70)
    print(f"   Number of boosting stages: {model.n_estimators}")
    print(f"   Learning rate: {model.learning_rate}")
    print(f"   Max depth per tree: {model.max_depth}")
    print(f"   Training score: {model.train_score_[-1]:.4f}")
    print("="*70)
    
    print("\n📊 Feature Importance:")
    print("="*70)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(importance_df.to_string(index=False))
    print("\n💡 Features that contribute most to reducing prediction error")
    print("="*70)
    
    return importance_df


if __name__ == "__main__":
    print("\n🚀 Running Gradient Boosting Analysis")
    print("="*70)
    
    # Load and prepare data
    data = load_prostate_data()
    data_dict = prepare_data(data)
    
    # Train model
    model, predictions, metrics = train_gradient_boosting(data_dict)
    
    # Analyze boosting
    importance_df = analyze_boosting(model, data_dict['feature_names'])
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    plot_feature_importance(
        data_dict['feature_names'], 
        model.feature_importances_,
        "Gradient Boosting",
        save=True
    )
    
    plot_predictions(
        data_dict['y_test'],
        predictions,
        "Gradient Boosting",
        metrics['r2_score'],
        save=True
    )
    
    plot_residuals(
        data_dict['y_test'],
        predictions,
        "Gradient Boosting",
        save=True
    )
    
    print("\n✅ Gradient Boosting analysis complete!")
    print(f"🏆 Final R² Score: {metrics['r2_score']:.4f}")
    
    # Experiment suggestions
    print("\n💡 Want to experiment? Try tuning hyperparameters:")
    print("   from src.models.gradient_boosting import train_gradient_boosting")
    print("   # Try slower learning rate with more estimators:")
    print("   train_gradient_boosting(data_dict, n_estimators=200, learning_rate=0.05)")
