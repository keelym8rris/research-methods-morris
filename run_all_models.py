"""Run the original exploratory analysis and save its figures."""

import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append('.')

from src.data_loader import get_predictors_and_target, load_prostate_data, prepare_data
from src.visualization import (
    plot_data_distribution,
    plot_correlation_heatmap,
    plot_feature_relationships
)
from src.evaluation import compare_models
from src.visualization import plot_model_comparison, plot_all_predictions_grid

# Import all model training functions
from src.models.linear_regression import train_linear_regression
from src.models.decision_tree import train_decision_tree
from src.models.random_forest import train_random_forest
from src.models.gradient_boosting import train_gradient_boosting
from src.models.neural_network import train_neural_network, plot_training_history


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70 + "\n")


def main():
    print("\nNOTE: This is the legacy educational single-split workflow.")
    print("Use `python run_research_analysis.py` for thesis results.\n")
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█  PROSTATE CANCER PSA PREDICTION - COMPLETE ML ANALYSIS  ".center(70 - 2) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print("\n This script will:")
    print("   1. Perform exploratory data analysis")
    print("   2. Train 5 different machine learning models")
    print("   3. Compare their performance")
    print("   4. Save all visualizations and results")
    print("\n⏱  Estimated time: 2-3 minutes")
    print("\nStarting analysis...\n")
    
    # ========================================================================
    # PART 1: EXPLORATORY DATA ANALYSIS
    # ========================================================================
    print_header(" PART 1: EXPLORATORY DATA ANALYSIS")
    
    print("Loading dataset...")
    data = load_prostate_data()
    
    print("\nDataset Overview:")
    print(f"   • {data.shape[0]} patients")
    print(f"   • {data.shape[1]} features")
    print(f"   • Target variable: PSA levels")
    
    print("\nCreating exploratory visualizations...")
    plot_data_distribution(data, save=True)
    plot_correlation_heatmap(data, save=True)
    plot_feature_relationships(data, save=True)
    print(" EDA complete!\n")
    
    # ========================================================================
    # PART 2: DATA PREPARATION
    # ========================================================================
    print_header(" PART 2: DATA PREPARATION")
    
    data_dict = prepare_data(data)
    print(" Data prepared and split!\n")
    
    # ========================================================================
    # PART 3: MODEL TRAINING
    # ========================================================================
    print_header(" PART 3: TRAINING MACHINE LEARNING MODELS")
    
    models_results = []
    predictions_dict = {}
    trained_models = {}
    
    # 1. Linear Regression
    print("1⃣ Linear Regression...")
    lr_model, lr_pred, lr_metrics = train_linear_regression(data_dict, verbose=False)
    models_results.append(lr_metrics)
    predictions_dict['Linear Regression'] = (lr_pred, lr_metrics['r2_score'])
    trained_models['Linear Regression'] = lr_model
    print(f"    R² = {lr_metrics['r2_score']:.4f}\n")
    
    # 2. Decision Tree
    print("2⃣ Decision Tree...")
    dt_model, dt_pred, dt_metrics = train_decision_tree(data_dict, verbose=False)
    models_results.append(dt_metrics)
    predictions_dict['Decision Tree'] = (dt_pred, dt_metrics['r2_score'])
    trained_models['Decision Tree'] = dt_model
    print(f"    R² = {dt_metrics['r2_score']:.4f}\n")
    
    # 3. Random Forest
    print("3⃣ Random Forest (100 trees)...")
    rf_model, rf_pred, rf_metrics = train_random_forest(data_dict, verbose=False)
    models_results.append(rf_metrics)
    predictions_dict['Random Forest'] = (rf_pred, rf_metrics['r2_score'])
    trained_models['Random Forest'] = rf_model
    print(f"    R² = {rf_metrics['r2_score']:.4f}\n")
    
    # 4. Gradient Boosting
    print("4⃣ Gradient Boosting...")
    gb_model, gb_pred, gb_metrics = train_gradient_boosting(data_dict, verbose=False)
    models_results.append(gb_metrics)
    predictions_dict['Gradient Boosting'] = (gb_pred, gb_metrics['r2_score'])
    trained_models['Gradient Boosting'] = gb_model
    print(f"    R² = {gb_metrics['r2_score']:.4f}\n")
    
    # 5. Neural Network
    print("5⃣ Neural Network (Deep Learning)...")
    nn_model, nn_pred, nn_metrics, nn_history = train_neural_network(data_dict, verbose=False)
    models_results.append(nn_metrics)
    predictions_dict['Neural Network'] = (nn_pred, nn_metrics['r2_score'])
    trained_models['Neural Network'] = (nn_model, nn_history)
    print(f"    R² = {nn_metrics['r2_score']:.4f}\n")
    
    # ========================================================================
    # PART 4: MODEL COMPARISON
    # ========================================================================
    print_header(" PART 4: MODEL COMPARISON & RESULTS")
    
    comparison_df = compare_models(models_results)
    
    print(" MODEL PERFORMANCE RANKINGS:")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70)
    
    # Identify winner
    best_model = comparison_df.iloc[0]
    print(f"\n Highest R² on this split: {best_model['model_name']}")
    print(f"   R² Score: {best_model['r2_score']:.4f} ({best_model['r2_score']*100:.1f}%)")
    print(f"   Test-set R² is {best_model['r2_score']:.4f}; this is not diagnostic accuracy.")
    
    # Create comparison visualizations
    print("\n Creating comparison visualizations...")
    plot_model_comparison(comparison_df, save=True)
    plot_all_predictions_grid(data_dict['y_test'], predictions_dict, save=True)
    
    # Plot neural network training history
    if 'Neural Network' in trained_models:
        _, history = trained_models['Neural Network']
        plot_training_history(history, save=True)
    
    # ========================================================================
    # PART 5: FINAL SUMMARY
    # ========================================================================
    print_header(" ANALYSIS COMPLETE - FINAL SUMMARY")
    
    print(" RESULTS OVERVIEW:")
    print(f"   • Models trained: {len(models_results)}")
    print(f"   • Best performing model: {best_model['model_name']}")
    print(f"   • Best R² score: {best_model['r2_score']:.4f}")
    print(f"   • Best MSE: {best_model['mse']:.4f}")
    
    print("\n MODEL PERFORMANCE RANGE:")
    print(f"   • R² scores: {comparison_df['r2_score'].min():.4f} to {comparison_df['r2_score'].max():.4f}")
    print(f"   • Average R²: {comparison_df['r2_score'].mean():.4f}")
    
    print("\n OUTPUT FILES:")
    print("   • All visualizations saved to: results/figures/")
    print("   • Individual model scripts available in: src/models/")
    
    print("\n KEY FINDINGS:")
    predictors, target = get_predictors_and_target(data)
    correlations = predictors.corrwith(target).abs().sort_values(ascending=False)
    print(f"   • Top predictor: {correlations.index[0]} (correlation: {correlations.iloc[0]:.3f})")
    print(f"   • Dataset size: {len(data)} patients")
    print(f"   • Features used: {len(data_dict['feature_names'])}")
    
    print("\n NEXT STEPS:")
    print("   • Review visualizations in results/figures/")
    print("   • Run individual model scripts for detailed analysis")
    print("   • Experiment with different hyperparameters")
    print("   • Compare with the repeated-validation thesis results")
    
    print("\n" + "="*70)
    print(" SUCCESS! All analysis complete.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
