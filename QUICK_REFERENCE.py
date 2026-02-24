"""
QUICK REFERENCE GUIDE
=====================

Quick commands for running different analyses:
"""

# ============================================================================
# BASIC USAGE
# ============================================================================

# 1. COMPLETE ANALYSIS (Do this first!)
# Runs everything: EDA + All Models + Comparisons
python run_all_models.py

# 2. EXPLORATORY DATA ANALYSIS ONLY
# Look at your data before building models
python explore_data.py

# 3. COMPARE ALL MODELS
# Train and compare all 5 models
python compare_models.py


# ============================================================================
# RUN INDIVIDUAL MODELS
# ============================================================================

# Linear Regression
python src/models/linear_regression.py

# Decision Tree
python src/models/decision_tree.py

# Random Forest (often the best!)
python src/models/random_forest.py

# Gradient Boosting (often wins competitions)
python src/models/gradient_boosting.py

# Neural Network (Deep Learning)
python src/models/neural_network.py


# ============================================================================
# PROGRAMMATIC USAGE (In Python scripts or Jupyter)
# ============================================================================

# Example: Train multiple models with custom parameters

from src.data_loader import load_prostate_data, prepare_data
from src.models.random_forest import train_random_forest
from src.models.gradient_boosting import train_gradient_boosting

# Load data once
data = load_prostate_data()
data_dict = prepare_data(data)

# Experiment with Random Forest
print("Testing Random Forest with different parameters...")
for n_trees in [50, 100, 200]:
    model, pred, metrics = train_random_forest(
        data_dict, 
        n_estimators=n_trees,
        verbose=False
    )
    print(f"Trees: {n_trees}, R² = {metrics['r2_score']:.4f}")

# Experiment with Gradient Boosting
print("\nTesting Gradient Boosting with different learning rates...")
for lr in [0.01, 0.05, 0.1, 0.2]:
    model, pred, metrics = train_gradient_boosting(
        data_dict,
        learning_rate=lr,
        verbose=False
    )
    print(f"LR: {lr}, R² = {metrics['r2_score']:.4f}")


# ============================================================================
# CUSTOM VISUALIZATIONS
# ============================================================================

from src.visualization import *
from src.data_loader import load_prostate_data

data = load_prostate_data()

# Create custom plots
plot_data_distribution(data, save=True)
plot_correlation_heatmap(data, save=True)
plot_feature_relationships(data, save=True)


# ============================================================================
# WHERE TO FIND RESULTS
# ============================================================================

# All visualizations (publication-quality 300 DPI):
# → results/figures/

# Individual model scripts:
# → src/models/

# Shared utilities:
# → src/data_loader.py
# → src/evaluation.py
# → src/visualization.py


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# If you get import errors:
# Make sure you're running from the project root directory

# If visualizations don't show:
# They're still saved to results/figures/ - check there!

# If you want to modify model parameters:
# Open the individual model scripts in src/models/
# Each has detailed documentation and parameter descriptions
