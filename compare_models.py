"""Compare the exploratory single-split models. Thesis results use repeated cross-validation."""

import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append('.')

from src.data_loader import load_prostate_data, prepare_data
from src.evaluation import compare_models
from src.visualization import plot_model_comparison, plot_all_predictions_grid

# Import all model training functions
from src.models.linear_regression import train_linear_regression
from src.models.decision_tree import train_decision_tree
from src.models.random_forest import train_random_forest
from src.models.gradient_boosting import train_gradient_boosting
from src.models.neural_network import train_neural_network


def main():
    print("\nNOTE: This is the legacy educational single-split comparison.")
    print("Use `python run_research_analysis.py` for thesis results.\n")
    print("\n" + "="*70)
    print(" COMPREHENSIVE MODEL COMPARISON")
    print("="*70)
    print("\nThis will train and evaluate 5 different ML models.")
    print("Grab a coffee, this may take a minute... \n")
    
    # Load and prepare data
    print(" Step 1: Loading and preparing data...")
    print("-"*70)
    data = load_prostate_data()
    data_dict = prepare_data(data)
    
    # Train all models
    print("\n Step 2: Training all models...")
    print("-"*70)
    
    models_results = []
    predictions_dict = {}
    
    # 1. Linear Regression
    print("\n1⃣ Training Linear Regression...")
    lr_model, lr_pred, lr_metrics = train_linear_regression(data_dict, verbose=False)
    models_results.append(lr_metrics)
    predictions_dict['Linear Regression'] = (lr_pred, lr_metrics['r2_score'])
    print(f"    Linear Regression: R² = {lr_metrics['r2_score']:.4f}")
    
    # 2. Decision Tree
    print("\n2⃣ Training Decision Tree...")
    dt_model, dt_pred, dt_metrics = train_decision_tree(data_dict, verbose=False)
    models_results.append(dt_metrics)
    predictions_dict['Decision Tree'] = (dt_pred, dt_metrics['r2_score'])
    print(f"    Decision Tree: R² = {dt_metrics['r2_score']:.4f}")
    
    # 3. Random Forest
    print("\n3⃣ Training Random Forest...")
    rf_model, rf_pred, rf_metrics = train_random_forest(data_dict, verbose=False)
    models_results.append(rf_metrics)
    predictions_dict['Random Forest'] = (rf_pred, rf_metrics['r2_score'])
    print(f"    Random Forest: R² = {rf_metrics['r2_score']:.4f}")
    
    # 4. Gradient Boosting
    print("\n4⃣ Training Gradient Boosting...")
    gb_model, gb_pred, gb_metrics = train_gradient_boosting(data_dict, verbose=False)
    models_results.append(gb_metrics)
    predictions_dict['Gradient Boosting'] = (gb_pred, gb_metrics['r2_score'])
    print(f"    Gradient Boosting: R² = {gb_metrics['r2_score']:.4f}")
    
    # 5. Neural Network
    print("\n5⃣ Training Neural Network...")
    nn_model, nn_pred, nn_metrics, nn_history = train_neural_network(data_dict, verbose=False)
    models_results.append(nn_metrics)
    predictions_dict['Neural Network'] = (nn_pred, nn_metrics['r2_score'])
    print(f"    Neural Network: R² = {nn_metrics['r2_score']:.4f}")
    
    # Compare all models
    print("\n" + "="*70)
    print(" Step 3: Comparing Model Performance")
    print("="*70)
    
    comparison_df = compare_models(models_results)
    print("\n MODEL RANKINGS:")
    print(comparison_df.to_string(index=False))
    
    # Identify best model
    best_model = comparison_df.iloc[0]
    print("\n" + "="*70)
    print(f" Highest R² on this split: {best_model['model_name']}")
    print("="*70)
    print(f"\n   R² Score:  {best_model['r2_score']:.4f} ({best_model['r2_score']*100:.1f}%)")
    print(f"   MSE:       {best_model['mse']:.4f}")
    print(f"   RMSE:      {best_model['rmse']:.4f}")
    print(f"   MAE:       {best_model['mae']:.4f}")
    print(f"\n   Test-set R² is {best_model['r2_score']:.4f}; this is not diagnostic accuracy.")
    print("="*70)
    
    # Create comparison visualizations
    print("\n Step 4: Creating comparison visualizations...")
    print("-"*70)
    
    print("\n1⃣ Creating model comparison charts...")
    plot_model_comparison(comparison_df, save=True)
    
    print("\n2⃣ Creating prediction comparison grid...")
    plot_all_predictions_grid(data_dict['y_test'], predictions_dict, save=True)
    
    # Summary
    print("\n" + "="*70)
    print(" MODEL COMPARISON COMPLETE!")
    print("="*70)
    
    print("\n Summary Statistics:")
    print(f"   • Best R² Score:    {comparison_df['r2_score'].max():.4f}")
    print(f"   • Worst R² Score:   {comparison_df['r2_score'].min():.4f}")
    print(f"   • Average R² Score: {comparison_df['r2_score'].mean():.4f}")
    print(f"   • Best MSE:         {comparison_df['mse'].min():.4f}")
    
    print("\n Recommendations:")
    if best_model['r2_score'] > 0.7:
        print(f"   • {best_model['model_name']} ranks first on this one split")
        print("   • Confirm with repeated validation before drawing conclusions")
    elif best_model['r2_score'] > 0.5:
        print(f"   • {best_model['model_name']} ranks first on this one split")
        print("   • Compare uncertainty with the thesis-grade workflow")
    else:
        print("   • All models show room for improvement")
        print("   • Consider feature engineering or more data")
    
    print("\n All visualizations saved to: results/figures/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
