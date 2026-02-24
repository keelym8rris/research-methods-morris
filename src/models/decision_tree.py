"""
Decision Tree Model
===================

Decision Trees make predictions by learning a series of yes/no questions
about the features. They're easy to understand and visualize.

Advantages:
- Very interpretable (can visualize the decision process)
- Can capture non-linear relationships
- No need for feature scaling
- Handles both numerical and categorical data

Disadvantages:
- Can easily overfit (memorize training data)
- Can be unstable (small changes in data → different tree)

When to use:
- When you need an easily explainable model
- When relationships might be non-linear
- As a building block for ensemble methods
"""

import sys
import numpy as np
from sklearn.tree import DecisionTreeRegressor
sys.path.append('.')

from src.data_loader import load_prostate_data, prepare_data
from src.evaluation import evaluate_model, print_evaluation
from src.visualization import plot_predictions, plot_residuals, plot_feature_importance


def train_decision_tree(data_dict, max_depth=5, random_state=42, verbose=True):
    """
    Train a Decision Tree model for PSA prediction.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary from prepare_data() containing training and test sets
    max_depth : int
        Maximum depth of the tree (prevents overfitting)
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
        print("🌳 DECISION TREE MODEL")
        print("="*70)
        print(f"\nTraining decision tree (max_depth={max_depth})...")
    
    # Create and train the model
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(data_dict['X_train_scaled'], data_dict['y_train'])
    
    if verbose:
        print("✅ Model trained successfully!")
    
    # Make predictions
    y_pred = model.predict(data_dict['X_test_scaled'])
    
    # Evaluate the model
    metrics = evaluate_model(data_dict['y_test'], y_pred, "Decision Tree")
    
    if verbose:
        print_evaluation(metrics)
    
    return model, y_pred, metrics


def analyze_tree_structure(model, feature_names):
    """
    Analyze and display information about the decision tree structure.
    
    Parameters:
    -----------
    model : DecisionTreeRegressor
        Trained decision tree model
    feature_names : list
        Names of the features
    """
    import pandas as pd
    
    print("\n🌳 Decision Tree Structure:")
    print("="*70)
    print(f"   Tree depth: {model.get_depth()}")
    print(f"   Number of leaves: {model.get_n_leaves()}")
    print(f"   Number of nodes: {model.tree_.node_count}")
    print("="*70)
    
    print("\n📊 Feature Importance (from tree splits):")
    print("="*70)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(importance_df.to_string(index=False))
    print("\n💡 Higher values = feature is used more in decision splits")
    print("="*70)
    
    return importance_df


if __name__ == "__main__":
    print("\n🚀 Running Decision Tree Analysis")
    print("="*70)
    
    # Load and prepare data
    data = load_prostate_data()
    data_dict = prepare_data(data)
    
    # Train model
    model, predictions, metrics = train_decision_tree(data_dict)
    
    # Analyze tree structure
    importance_df = analyze_tree_structure(model, data_dict['feature_names'])
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    plot_feature_importance(
        data_dict['feature_names'], 
        model.feature_importances_,
        "Decision Tree",
        save=True
    )
    
    plot_predictions(
        data_dict['y_test'],
        predictions,
        "Decision Tree",
        metrics['r2_score'],
        save=True
    )
    
    plot_residuals(
        data_dict['y_test'],
        predictions,
        "Decision Tree",
        save=True
    )
    
    print("\n✅ Decision Tree analysis complete!")
    print(f"🏆 Final R² Score: {metrics['r2_score']:.4f}")
    
    # Experiment suggestion
    print("\n💡 Want to experiment? Try different max_depth values:")
    print("   from src.models.decision_tree import train_decision_tree")
    print("   train_decision_tree(data_dict, max_depth=10)")
