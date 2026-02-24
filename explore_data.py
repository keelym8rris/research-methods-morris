"""
Exploratory Data Analysis (EDA)
================================

This script performs comprehensive exploratory data analysis on the 
prostate cancer dataset. Run this first to understand your data before 
building models.

What this script does:
1. Loads and displays basic dataset information
2. Shows statistical summaries
3. Creates visualizations of data distributions
4. Analyzes correlations between features
5. Saves all plots for use in papers/presentations
"""

import sys
sys.path.append('.')

from src.data_loader import load_prostate_data, get_feature_descriptions
from src.visualization import (
    plot_data_distribution, 
    plot_correlation_heatmap, 
    plot_feature_relationships
)


def main():
    print("\n" + "="*70)
    print("🔍 EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    print("\n📊 Step 1: Loading Dataset")
    print("-"*70)
    data = load_prostate_data()
    
    print("\n📋 Step 2: Dataset Overview")
    print("-"*70)
    print(f"\nDataset shape: {data.shape[0]} patients × {data.shape[1]} features")
    print("\nFirst 5 patients:")
    print(data.head())
    
    print("\n📊 Step 3: Statistical Summary")
    print("-"*70)
    print(data.describe())
    
    print("\n📖 Step 4: Feature Descriptions")
    print("-"*70)
    descriptions = get_feature_descriptions()
    for feature, description in descriptions.items():
        if feature in data.columns:
            print(f"   • {feature}: {description}")
    
    print("\n📊 Step 5: Checking for Missing Values")
    print("-"*70)
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values detected!")
    else:
        print("Missing values:")
        print(missing[missing > 0])
    
    print("\n📈 Step 6: Creating Visualizations")
    print("-"*70)
    print("\nGenerating plots (these will be saved to results/figures/)...")
    
    # Plot 1: Distribution of target variable
    print("\n1️⃣ Creating PSA distribution plots...")
    plot_data_distribution(data, save=True)
    
    # Plot 2: Correlation heatmap
    print("\n2️⃣ Creating correlation heatmap...")
    plot_correlation_heatmap(data, save=True)
    
    # Plot 3: Feature relationships
    print("\n3️⃣ Creating feature relationship plots...")
    plot_feature_relationships(data, save=True)
    
    print("\n" + "="*70)
    print("✅ EXPLORATORY DATA ANALYSIS COMPLETE!")
    print("="*70)
    print("\n📊 Key Findings:")
    
    # Calculate some interesting statistics
    print(f"\n   Target Variable (PSA) Statistics:")
    print(f"   • Mean PSA: {data['lpsa'].mean():.3f}")
    print(f"   • Median PSA: {data['lpsa'].median():.3f}")
    print(f"   • Std Dev: {data['lpsa'].std():.3f}")
    print(f"   • Range: {data['lpsa'].min():.3f} to {data['lpsa'].max():.3f}")
    
    # Find strongest correlations with PSA
    correlations = data.corr()['lpsa'].abs().sort_values(ascending=False)
    print(f"\n   Top 3 Features Correlated with PSA:")
    for i, (feature, corr) in enumerate(list(correlations.items())[1:4], 1):
        print(f"   {i}. {feature}: {corr:.3f}")
    
    print("\n💾 All visualizations saved to: results/figures/")
    print("\n🚀 Next Step: Run individual models or compare_models.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
