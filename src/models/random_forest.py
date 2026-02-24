"""
Random Forest Model
===================

Random Forest is an ensemble method that combines many decision trees to make
more robust and accurate predictions. Think of it as "wisdom of the crowd".

How it works:
1. Creates many decision trees (typically 100+)
2. Each tree is trained on a random subset of data
3. Each tree makes a prediction
4. Final prediction = average of all trees

Advantages:
- Usually more accurate than a single decision tree
- Less prone to overfitting
- Handles non-linear relationships well
- Works well "out of the box"
- Provides feature importance rankings

Disadvantages:
- Slower to train (many trees)
- Less interpretable than a single tree
- Can be memory-intensive

When to use:
- When you want high accuracy
- When you have sufficient data
- As a strong baseline for many problems
"""

import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
sys.path.append('.')

from src.data_loader import load_prostate_data, prepare_data
from src.evaluation import evaluate_model, print_evaluation
from src.visualization import plot_predictions, plot_residuals, plot_feature_importance


def train_random_forest(data_dict, n_estimators=100, max_depth=10, 
                       random_state=42, verbose=True):
    """
    Train a Random Forest model for PSA prediction.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary from prepare_data() containing training and test sets
    n_estimators : int
        Number of trees in the forest
    max_depth : int
        Maximum depth of each tree
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
        print("🌲 RANDOM FOREST MODEL")
        print("="*70)
        print(f"\nTraining random forest with {n_estimators} trees...")
    
    # Create and train the model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    model.fit(data_dict['X_train_scaled'], data_dict['y_train'])
    
    if verbose:
        print(f"✅ Successfully trained {n_estimators} trees!")
    
    # Make predictions
    y_pred = model.predict(data_dict['X_test_scaled'])
    
    # Evaluate the model
    metrics = evaluate_model(data_dict['y_test'], y_pred, "Random Forest")
    
    if verbose:
        print_evaluation(metrics)
    
    return model, y_pred, metrics


def analyze_forest(model, feature_names):
    """
    Analyze and display information about the random forest.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained random forest model
    feature_names : list
        Names of the features
    """
    import pandas as pd
    
    print("\n🌲 Random Forest Details:")
    print("="*70)
    print(f"   Number of trees: {model.n_estimators}")
    print(f"   Max depth per tree: {model.max_depth}")
    print(f"   Number of features considered per split: {model.max_features}")
    print("="*70)
    
    print("\n📊 Feature Importance (averaged across all trees):")
    print("="*70)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(importance_df.to_string(index=False))
    print("\n💡 Higher values = feature is more important for predictions")
    print("="*70)
    
    return importance_df


if __name__ == "__main__":
    print("\n🚀 Running Random Forest Analysis")
    print("="*70)
    
    # Load and prepare data
    data = load_prostate_data()
    data_dict = prepare_data(data)
    
    # Train model
    model, predictions, metrics = train_random_forest(data_dict)
    
    # Analyze forest
    importance_df = analyze_forest(model, data_dict['feature_names'])
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    plot_feature_importance(
        data_dict['feature_names'], 
        model.feature_importances_,
        "Random Forest",
        save=True
    )
    
    plot_predictions(
        data_dict['y_test'],
        predictions,
        "Random Forest",
        metrics['r2_score'],
        save=True
    )
    
    plot_residuals(
        data_dict['y_test'],
        predictions,
        "Random Forest",
        save=True
    )
    
    print("\n✅ Random Forest analysis complete!")
    print(f"🏆 Final R² Score: {metrics['r2_score']:.4f}")
    
    # Experiment suggestions
    print("\n💡 Want to experiment? Try different parameters:")
    print("   from src.models.random_forest import train_random_forest")
    print("   train_random_forest(data_dict, n_estimators=200, max_depth=15)")
