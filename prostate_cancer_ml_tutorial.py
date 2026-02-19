"""
🎓 YOUR FIRST MACHINE LEARNING PROJECT: PROSTATE CANCER ANALYSIS
================================================================

This is a complete machine learning tutorial converted from Jupyter Notebook.
You can run this entire file or run sections step by step.

To run the entire script: python prostate_cancer_ml_tutorial.py
To run step by step: Copy sections into Python interactive mode or run line by line

Author: Keely's ML Tutorial
Date: February 2026
"""

# ============================================================================
# SECTION 1: SETTING UP YOUR ENVIRONMENT
# ============================================================================
print("\n" + "="*70)
print("📦 SECTION 1: SETTING UP YOUR ENVIRONMENT")
print("="*70 + "\n")

# System libraries
import warnings
warnings.filterwarnings('ignore')  # This hides warning messages to keep things clean

# Data manipulation - working with tables of data
import numpy as np  # For math and numbers
import pandas as pd  # For working with data tables

# Visualization - making charts and graphs
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning models and tools
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Deep Learning (Neural Networks)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Set styling for our graphs to make them look nice
sns.set_theme(style='whitegrid', font='serif')
plt.rcParams['figure.figsize'] = (10, 6)  # Make graphs a good size

print("✅ All libraries loaded successfully!")
print(f"📊 Using TensorFlow version: {tf.__version__}")

input("\nPress Enter to continue to Section 2...")

# ============================================================================
# SECTION 2: UNDERSTANDING YOUR DATA
# ============================================================================
print("\n" + "="*70)
print("📊 SECTION 2: UNDERSTANDING YOUR DATA")
print("="*70 + "\n")

print("""
What is this dataset?
This is real medical data from prostate cancer patients. Each row represents 
one patient, and each column is a measurement or test result.

What do the column names mean?
- lcavol: Log of cancer volume (how big the tumor is)
- lweight: Log of prostate weight
- age: Patient's age in years
- lbph: Log of benign prostatic hyperplasia amount (non-cancerous growth)
- svi: Seminal vesicle invasion (0 = no, 1 = yes) - has cancer spread?
- lcp: Log of capsular penetration (how far cancer has spread)
- gleason: Gleason score (a grading system for prostate cancer, 6-10)
- pgg45: Percentage of Gleason scores 4 or 5 (higher = more aggressive)
- lpsa: Log of PSA level (**this is what we want to predict!**)
- train: Whether this data point was used for training (we'll use this later)
""")

# Load the dataset
data = pd.read_csv('data/prostate.csv')

print("📈 Dataset loaded successfully!")
print(f"\n📊 We have {len(data)} patients in our dataset\n")

# Look at the first few rows
print("First 5 patients:")
print(data.head())

print("\n📋 Dataset Information:")
print(data.info())

print("\n📊 Statistical Summary:")
print(data.describe())

input("\nPress Enter to continue to Section 3...")

# ============================================================================
# SECTION 3: DATA EXPLORATION AND VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("🔍 SECTION 3: DATA EXPLORATION AND VISUALIZATION")
print("="*70 + "\n")

print("""
Why visualize data?
- 👀 See patterns that numbers alone don't show
- 🔎 Find unusual values (outliers)
- 💡 Understand relationships between variables
- 📈 Make informed decisions about which models to use
""")

# Visualize the distribution of our target variable (lpsa)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data['lpsa'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Log PSA Level')
plt.ylabel('Number of Patients')
plt.title('Distribution of PSA Levels\n(What we want to predict)')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(data['lpsa'], vert=True)
plt.ylabel('Log PSA Level')
plt.title('PSA Levels - Box Plot\n(Shows median and outliers)')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("📊 The histogram shows how PSA levels are distributed across patients.")
print("📦 The box plot shows the median (middle line) and any unusual values.")

# Create a correlation heatmap
plt.figure(figsize=(12, 10))

correlation_matrix = data.drop(columns=['train']).corr()

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Heatmap\n(How different measurements relate to each other)', 
          fontsize=14, pad=20)
plt.tight_layout()
plt.show()

print("\n🔥 What to look for:")
print("  • Dark red = strong positive correlation (variables increase together)")
print("  • Dark blue = strong negative correlation (one increases, other decreases)")
print("  • White = no correlation (variables are independent)")
print("\n💡 Look at the 'lpsa' row - these are variables that might help predict PSA!")

# Let's look at relationships between specific features and PSA
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

features_to_plot = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'gleason']
feature_names = ['Cancer Volume', 'Prostate Weight', 'Age', 
                 'BPH Amount', 'Seminal Vesicle Invasion', 'Gleason Score']

for i, (feature, name) in enumerate(zip(features_to_plot, feature_names)):
    row, col = i // 3, i % 3
    axes[row, col].scatter(data[feature], data['lpsa'], alpha=0.6, color='coral')
    axes[row, col].set_xlabel(name)
    axes[row, col].set_ylabel('PSA Level')
    axes[row, col].set_title(f'{name} vs PSA')
    axes[row, col].grid(alpha=0.3)

plt.suptitle('How Different Factors Relate to PSA Levels', fontsize=16, y=1.00)
plt.tight_layout()
plt.show()

print("📈 Each dot is one patient. Look for patterns!")
print("   • Upward slope = as this factor increases, PSA tends to increase")
print("   • Downward slope = as this factor increases, PSA tends to decrease")
print("   • Random scatter = this factor doesn't strongly predict PSA")

input("\nPress Enter to continue to Section 4...")

# ============================================================================
# SECTION 4: PREPARING DATA FOR MACHINE LEARNING
# ============================================================================
print("\n" + "="*70)
print("🎯 SECTION 4: PREPARING DATA FOR MACHINE LEARNING")
print("="*70 + "\n")

print("""
Two important concepts:

1. Train/Test Split
   - Training data (80%): We use this to teach our model
   - Testing data (20%): We use this to see how well our model learned

2. Standardization (Scaling)
   - Different measurements have different scales (Age: 40-80, PSA: 0-5)
   - Standardization puts everything on the same scale
   - This helps ML models learn better
""")

# Step 1: Separate our features (X) from our target (y)
data['train'] = data['train'].astype(int)
X = data.drop(columns=['lpsa'])
y = data['lpsa']

print("📊 Features (X):")
print(f"   Shape: {X.shape} (meaning {X.shape[0]} patients, {X.shape[1]} features)")
print(f"   Columns: {list(X.columns)}\n")

print("🎯 Target (y):")
print(f"   Shape: {y.shape} (meaning {y.shape[0]} PSA values)")
print(f"   What we're predicting: PSA levels")

# Step 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n✂️ Data Split Complete!\n")
print(f"📚 Training set: {len(X_train)} patients ({len(X_train)/len(X)*100:.0f}%)")
print(f"🧪 Testing set: {len(X_test)} patients ({len(X_test)/len(X)*100:.0f}%)")

# Step 3: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n⚖️ Data Standardization Complete!")
print("\n💡 Values are now scaled with mean=0 and standard deviation=1")
print("   This helps ML models learn better.")

input("\nPress Enter to continue to Section 5...")

# ============================================================================
# SECTION 5: MODEL 1 - LINEAR REGRESSION
# ============================================================================
print("\n" + "="*70)
print("🎨 SECTION 5: YOUR FIRST MODEL - LINEAR REGRESSION")
print("="*70 + "\n")

print("""
What is Linear Regression?

Remember drawing a "line of best fit" in high school math? That's basically 
what linear regression does!

The idea: Find the straight line that best fits the data points. Then we can 
use this line to predict new values.

In simple terms:
- Creates an equation: PSA = (weight1 × feature1) + (weight2 × feature2) + ...
- The model "learns" the best weights to minimize prediction errors

When to use it:
✅ Great for understanding which features are important
✅ Fast to train and easy to interpret
✅ Works well when relationships are roughly linear
❌ Not good for complex, non-linear relationships
""")

# Create and train the Linear Regression model
print("🏗️ Building Linear Regression model...\n")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

print("✅ Model trained successfully!\n")

# Let's see what the model learned
print("📊 Feature Importance (Coefficients):")
print("(Larger absolute values = more important for prediction)\n")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(feature_importance)

print("\n💡 Interpretation:")
print("   • Positive coefficient = as this increases, PSA tends to increase")
print("   • Negative coefficient = as this increases, PSA tends to decrease")
print("   • Larger absolute value = stronger influence on PSA prediction")

# Visualize feature importance
plt.figure(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors, alpha=0.7)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance in Linear Regression Model\n(Green = increases PSA, Red = decreases PSA)', 
          fontsize=14, pad=20)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# Make predictions on the test set
y_pred_lr = lr_model.predict(X_test_scaled)

print("\n🔮 Making predictions on test data...\n")
print("Sample predictions vs actual values:\n")

comparison_df = pd.DataFrame({
    'Actual PSA': y_test[:10].values,
    'Predicted PSA': y_pred_lr[:10],
    'Difference': y_test[:10].values - y_pred_lr[:10]
})

print(comparison_df)

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.6, s=100, edgecolors='black', linewidth=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual PSA Levels', fontsize=12)
plt.ylabel('Predicted PSA Levels', fontsize=12)
plt.title('Linear Regression: Predicted vs Actual PSA Levels', fontsize=14, pad=20)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("📊 If our model was perfect, all points would be on the red line!")

input("\nPress Enter to continue to Section 6...")

# ============================================================================
# SECTION 6: UNDERSTANDING MODEL PERFORMANCE
# ============================================================================
print("\n" + "="*70)
print("📏 SECTION 6: UNDERSTANDING MODEL PERFORMANCE")
print("="*70 + "\n")

print("""
Two Key Metrics:

1. Mean Squared Error (MSE)
   - Measures the average squared difference between predicted and actual values
   - Lower is better! (0 = perfect predictions)
   
2. R² Score (R-squared)
   - Tells you what percentage of the variance in PSA your model explains
   - Ranges from 0 to 1 (1 = perfect, 0 = no better than guessing the average)
   - Example: R² = 0.75 means your model explains 75% of the variation

Rule of Thumb:
- R² > 0.7: Great model! 🌟
- R² = 0.5-0.7: Good model ✅
- R² < 0.5: Needs improvement 📈
""")

# Calculate performance metrics for Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("="*60)
print("📊 LINEAR REGRESSION PERFORMANCE")
print("="*60)
print(f"\n📉 Mean Squared Error (MSE): {mse_lr:.4f}")
print(f"   → Lower is better. This is the average squared error.")
print(f"\n📈 R² Score: {r2_lr:.4f} ({r2_lr*100:.1f}%)")
print(f"   → Our model explains {r2_lr*100:.1f}% of the variance in PSA levels!")

if r2_lr > 0.7:
    print("\n🌟 Excellent! This is a strong model.")
elif r2_lr > 0.5:
    print("\n✅ Good! This model captures meaningful patterns.")
else:
    print("\n📈 Room for improvement. Let's try other models!")

print("\n" + "="*60)

# Visualize the errors (residuals)
residuals = y_test - y_pred_lr

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Residual plot
axes[0].scatter(y_pred_lr, residuals, alpha=0.6, s=80)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted PSA', fontsize=11)
axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=11)
axes[0].set_title('Residual Plot\n(Points should be randomly scattered around zero)', fontsize=12)
axes[0].grid(alpha=0.3)

# Plot 2: Distribution of errors
axes[1].hist(residuals, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residual (Error)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Distribution of Prediction Errors\n(Should look like a bell curve centered at 0)', fontsize=12)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("💡 What to look for:")
print("   • Left plot: Random scatter = good! Pattern = model is missing something")
print("   • Right plot: Bell curve centered at 0 = good predictions!")

input("\nPress Enter to continue to Section 7...")

# ============================================================================
# SECTION 7: MODEL 2 - DECISION TREE
# ============================================================================
print("\n" + "="*70)
print("🌳 SECTION 7: DECISION TREES - A DIFFERENT APPROACH")
print("="*70 + "\n")

print("""
What is a Decision Tree?

Think of a decision tree like a flowchart:

Is cancer volume > 2.5?
├─ YES → Is patient age > 65?
│        ├─ YES → Predict HIGH PSA
│        └─ NO → Predict MEDIUM PSA
└─ NO → Predict LOW PSA

Pros and Cons:
✅ Very easy to understand and explain
✅ Can capture non-linear relationships
✅ Works with both numbers and categories
❌ Can easily "overfit" (memorize training data)
❌ Sometimes unstable
""")

# Create and train a Decision Tree model
print("🌳 Building Decision Tree model...\n")

dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
dt_model.fit(X_train_scaled, y_train)

y_pred_dt = dt_model.predict(X_test_scaled)

mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("✅ Decision Tree trained!\n")
print("="*60)
print("📊 DECISION TREE PERFORMANCE")
print("="*60)
print(f"\n📉 Mean Squared Error (MSE): {mse_dt:.4f}")
print(f"📈 R² Score: {r2_dt:.4f} ({r2_dt*100:.1f}%)\n")
print("="*60)

# Visualize feature importance for Decision Tree
feature_imp_dt = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp_dt['Feature'], feature_imp_dt['Importance'], color='forestgreen', alpha=0.7)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Decision Tree: Feature Importance\n(Which features does the tree use most?)', 
          fontsize=14, pad=20)
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("📊 Feature Importance Rankings:\n")
print(feature_imp_dt)

# Compare predictions: Decision Tree vs Linear Regression
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Linear Regression
axes[0].scatter(y_test, y_pred_lr, alpha=0.6, s=80, color='blue', edgecolors='black', linewidth=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual PSA', fontsize=11)
axes[0].set_ylabel('Predicted PSA', fontsize=11)
axes[0].set_title(f'Linear Regression\nR² = {r2_lr:.3f}', fontsize=12)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Decision Tree
axes[1].scatter(y_test, y_pred_dt, alpha=0.6, s=80, color='green', edgecolors='black', linewidth=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual PSA', fontsize=11)
axes[1].set_ylabel('Predicted PSA', fontsize=11)
axes[1].set_title(f'Decision Tree\nR² = {r2_dt:.3f}', fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Model Comparison: Actual vs Predicted PSA', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

input("\nPress Enter to continue to Section 8...")

# ============================================================================
# SECTION 8: MODEL 3 - RANDOM FOREST
# ============================================================================
print("\n" + "="*70)
print("🌲 SECTION 8: RANDOM FORESTS - COMBINING MULTIPLE TREES")
print("="*70 + "\n")

print("""
What is a Random Forest?

Imagine asking 100 doctors for their diagnosis instead of just 1. 
Random Forest works the same way:

1. Creates many decision trees (typically 100+)
2. Each tree is trained on a slightly different subset of the data
3. Each tree makes a prediction
4. Final prediction = average of all tree predictions

Why Random Forests are Great:
✅ Usually more accurate than a single decision tree
✅ Less prone to overfitting
✅ Handles non-linear relationships well
✅ Works well "out of the box"

Downsides:
❌ Slower to train (building many trees takes time)
❌ Harder to interpret than a single decision tree
""")

# Create and train a Random Forest model
print("🌲🌲🌲 Building Random Forest (100 trees)...\n")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("✅ Random Forest trained with 100 trees!\n")
print("="*60)
print("📊 RANDOM FOREST PERFORMANCE")
print("="*60)
print(f"\n📉 Mean Squared Error (MSE): {mse_rf:.4f}")
print(f"📈 R² Score: {r2_rf:.4f} ({r2_rf*100:.1f}%)\n")
print("="*60)

print("\n📊 Comparison so far:")
print(f"   Linear Regression: R² = {r2_lr:.4f}")
print(f"   Decision Tree:     R² = {r2_dt:.4f}")
print(f"   Random Forest:     R² = {r2_rf:.4f} ← Latest model")

best_model_so_far = max([('Linear Regression', r2_lr), ('Decision Tree', r2_dt), ('Random Forest', r2_rf)], 
                         key=lambda x: x[1])
print(f"\n🏆 Best model so far: {best_model_so_far[0]}")

# Feature importance from Random Forest
feature_imp_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp_rf['Feature'], feature_imp_rf['Importance'], 
         color='darkgreen', alpha=0.7, edgecolor='black')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Random Forest: Feature Importance\n(Averaged across all 100 trees)', 
          fontsize=14, pad=20)
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("📊 Top 5 Most Important Features:\n")
print(feature_imp_rf.head())

# Visualize Random Forest predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, s=100, color='darkgreen', 
            edgecolors='black', linewidth=1, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual PSA Levels', fontsize=12)
plt.ylabel('Predicted PSA Levels', fontsize=12)
plt.title(f'Random Forest: Predicted vs Actual PSA\nR² = {r2_rf:.4f}', 
          fontsize=14, pad=20)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

input("\nPress Enter to continue to Section 9...")

# ============================================================================
# SECTION 9: MODEL 4 - GRADIENT BOOSTING
# ============================================================================
print("\n" + "="*70)
print("🚀 SECTION 9: GRADIENT BOOSTING - SEQUENTIAL LEARNING")
print("="*70 + "\n")

print("""
What is Gradient Boosting?

While Random Forest builds many trees independently, Gradient Boosting builds 
trees sequentially, where each new tree tries to fix the mistakes of previous trees.

The Process:
1. Build tree #1 → Make predictions → Calculate errors
2. Build tree #2 to predict those errors → Combine with tree #1
3. Build tree #3 to predict remaining errors → Combine with trees #1 and #2
4. Keep going until errors are minimized!

Analogy: Studying for a test
- Tree 1: You study and take a practice test
- Tree 2: You focus on questions you got wrong
- Tree 3: You focus on questions you're still getting wrong

Pros and Cons:
✅ Often very accurate (wins many ML competitions!)
✅ Good at handling complex patterns
❌ Can be slow to train (trees built sequentially)
❌ More prone to overfitting than Random Forest
""")

# Create and train a Gradient Boosting model
print("🚀 Building Gradient Boosting model...\n")
print("(This learns sequentially, so it might take a moment...)\n")

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)

y_pred_gb = gb_model.predict(X_test_scaled)

mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print("✅ Gradient Boosting trained!\n")
print("="*60)
print("📊 GRADIENT BOOSTING PERFORMANCE")
print("="*60)
print(f"\n📉 Mean Squared Error (MSE): {mse_gb:.4f}")
print(f"📈 R² Score: {r2_gb:.4f} ({r2_gb*100:.1f}%)\n")
print("="*60)

print("\n📊 Model Comparison:")
print(f"   Linear Regression:  R² = {r2_lr:.4f}")
print(f"   Decision Tree:      R² = {r2_dt:.4f}")
print(f"   Random Forest:      R² = {r2_rf:.4f}")
print(f"   Gradient Boosting:  R² = {r2_gb:.4f} ← Latest model")

# Feature importance from Gradient Boosting
feature_imp_gb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp_gb['Feature'], feature_imp_gb['Importance'], 
         color='orange', alpha=0.7, edgecolor='black')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Gradient Boosting: Feature Importance', fontsize=14, pad=20)
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("📊 Feature Importance Rankings:\n")
print(feature_imp_gb)

input("\nPress Enter to continue to Section 10...")

# ============================================================================
# SECTION 10: MODEL 5 - NEURAL NETWORK (DEEP LEARNING)
# ============================================================================
print("\n" + "="*70)
print("🧠 SECTION 10: NEURAL NETWORKS (DEEP LEARNING)")
print("="*70 + "\n")

print("""
What is a Neural Network?

Neural networks are inspired by how our brains work. Just like your brain has 
billions of neurons connected together, a neural network has artificial "neurons" 
organized in layers.

Structure:
Input Layer → Hidden Layer(s) → Output Layer
[Features]  →  [Processing]  → [Prediction]

How it works:
1. Input Layer: Takes in your features (age, cancer volume, etc.)
2. Hidden Layers: Process information through multiple "neurons"
3. Output Layer: Produces the final prediction (PSA level)

Learning Process (Training):
1. Make a prediction
2. Calculate how wrong you were (loss)
3. Adjust the weights to reduce the error
4. Repeat thousands of times!

Why Neural Networks?
✅ Can learn incredibly complex patterns
✅ State-of-the-art for images, text, speech
❌ Need more data to train well
❌ Harder to interpret ("black box")
❌ Require more computational power
""")

# Build a Neural Network with TensorFlow/Keras
print("🧠 Building a Neural Network...\n")

nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

print("🏗️ Neural Network Architecture:")
print("="*60)
nn_model.summary()
print("="*60)

print("\n💡 Understanding the architecture:")
print("   • Input: 10 features (age, cancer volume, etc.)")
print("   • Hidden Layer 1: 64 neurons processing the input")
print("   • Hidden Layer 2: 32 neurons processing layer 1")
print("   • Hidden Layer 3: 16 neurons processing layer 2")
print("   • Output: 1 neuron giving the PSA prediction")

# Compile the model
nn_model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

print("\n⚙️ Model compiled and ready to train!")

# Train the neural network
print("\n🎓 Training the Neural Network...")
print("(The network will go through the training data 100 times)\n")

history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print("✅ Training complete!\n")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

# Visualize the training process
plt.figure(figsize=(12, 5))

plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch (Training Iteration)', fontsize=12)
plt.ylabel('Loss (Mean Squared Error)', fontsize=12)
plt.title('Neural Network Training Progress\n(Loss should decrease over time)', 
          fontsize=14, pad=20)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("📉 What this graph shows:")
print("   • Both lines going down = network is learning!")
print("   • Training loss < validation loss = normal")

# Make predictions and evaluate
y_pred_nn = nn_model.predict(X_test_scaled, verbose=0)
y_pred_nn = y_pred_nn.flatten()

mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print("\n" + "="*60)
print("📊 NEURAL NETWORK PERFORMANCE")
print("="*60)
print(f"\n📉 Mean Squared Error (MSE): {mse_nn:.4f}")
print(f"📈 R² Score: {r2_nn:.4f} ({r2_nn*100:.1f}%)\n")
print("="*60)

print("\n📊 ALL MODELS COMPARISON:")
print(f"   Linear Regression:  R² = {r2_lr:.4f}")
print(f"   Decision Tree:      R² = {r2_dt:.4f}")
print(f"   Random Forest:      R² = {r2_rf:.4f}")
print(f"   Gradient Boosting:  R² = {r2_gb:.4f}")
print(f"   Neural Network:     R² = {r2_nn:.4f} ← Latest model")

# Visualize Neural Network predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_nn, alpha=0.6, s=100, color='purple', 
            edgecolors='black', linewidth=1, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual PSA Levels', fontsize=12)
plt.ylabel('Predicted PSA Levels', fontsize=12)
plt.title(f'Neural Network: Predicted vs Actual PSA\nR² = {r2_nn:.4f}', 
          fontsize=14, pad=20)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

input("\nPress Enter to continue to Section 11...")

# ============================================================================
# SECTION 11: COMPARING ALL MODELS
# ============================================================================
print("\n" + "="*70)
print("🏆 SECTION 11: COMPARING ALL MODELS")
print("="*70 + "\n")

print("""
We've built 5 different models:
1. Linear Regression - Simple, interpretable baseline
2. Decision Tree - Rule-based, easy to visualize
3. Random Forest - Ensemble of trees
4. Gradient Boosting - Sequential ensemble learning
5. Neural Network - Deep learning approach

Let's compare them using our two key metrics:
- MSE (lower is better)
- R² (higher is better)
""")

# Create a comprehensive comparison table
model_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 
              'Gradient Boosting', 'Neural Network'],
    'MSE': [mse_lr, mse_dt, mse_rf, mse_gb, mse_nn],
    'R² Score': [r2_lr, r2_dt, r2_rf, r2_gb, r2_nn]
})

model_comparison = model_comparison.sort_values('R² Score', ascending=False)
model_comparison['Rank'] = range(1, len(model_comparison) + 1)
model_comparison = model_comparison[['Rank', 'Model', 'MSE', 'R² Score']]

print("="*70)
print("🏆 MODEL PERFORMANCE RANKINGS")
print("="*70)
print(model_comparison.to_string(index=False))
print("="*70)

best_model = model_comparison.iloc[0]
print(f"\n🥇 WINNER: {best_model['Model']}")
print(f"   R² Score: {best_model['R² Score']:.4f}")
print(f"   MSE: {best_model['MSE']:.4f}")
print(f"\n   This model explains {best_model['R² Score']*100:.1f}% of the variance in PSA levels!")

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: MSE Comparison
colors_mse = ['gold' if i == 0 else 'silver' if i == 1 else 'chocolate' if i == 2 else 'lightgray' 
              for i in range(len(model_comparison))]
bars1 = axes[0].barh(model_comparison['Model'], model_comparison['MSE'], 
                     color=colors_mse, edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Mean Squared Error (Lower is Better)', fontsize=12)
axes[0].set_ylabel('Model', fontsize=12)
axes[0].set_title('Model Comparison: Mean Squared Error (MSE)', fontsize=13, pad=15)
axes[0].grid(alpha=0.3, axis='x')

for i, bar in enumerate(bars1):
    width = bar.get_width()
    axes[0].text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: R² Comparison
model_comparison_sorted_r2 = model_comparison.sort_values('R² Score', ascending=True)
colors_r2 = ['gold' if x == model_comparison_sorted_r2['R² Score'].max() 
             else 'silver' if x == model_comparison_sorted_r2['R² Score'].iloc[-2] 
             else 'chocolate' if x == model_comparison_sorted_r2['R² Score'].iloc[-3] 
             else 'lightgray' 
             for x in model_comparison_sorted_r2['R² Score']]
bars2 = axes[1].barh(model_comparison_sorted_r2['Model'], 
                     model_comparison_sorted_r2['R² Score'], 
                     color=colors_r2, edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('R² Score (Higher is Better)', fontsize=12)
axes[1].set_ylabel('Model', fontsize=12)
axes[1].set_title('Model Comparison: R-squared (R²)', fontsize=13, pad=15)
axes[1].grid(alpha=0.3, axis='x')
axes[1].set_xlim(0, 1)

for i, bar in enumerate(bars2):
    width = bar.get_width()
    axes[1].text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('🏆 Final Model Comparison\n(Gold = 1st, Silver = 2nd, Bronze = 3rd)', 
             fontsize=15, y=1.02, weight='bold')
plt.tight_layout()
plt.show()

# Side-by-side prediction comparison for all models
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.delaxes(axes[1, 2])

models_data = [
    ('Linear Regression', y_pred_lr, r2_lr, 'blue'),
    ('Decision Tree', y_pred_dt, r2_dt, 'green'),
    ('Random Forest', y_pred_rf, r2_rf, 'darkgreen'),
    ('Gradient Boosting', y_pred_gb, r2_gb, 'orange'),
    ('Neural Network', y_pred_nn, r2_nn, 'purple')
]

for idx, (name, predictions, r2, color) in enumerate(models_data):
    row, col = idx // 3, idx % 3
    axes[row, col].scatter(y_test, predictions, alpha=0.6, s=60, 
                          color=color, edgecolors='black', linewidth=0.5)
    axes[row, col].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        'r--', lw=2, alpha=0.8)
    axes[row, col].set_xlabel('Actual PSA', fontsize=10)
    axes[row, col].set_ylabel('Predicted PSA', fontsize=10)
    axes[row, col].set_title(f'{name}\nR² = {r2:.4f}', fontsize=11, weight='bold')
    axes[row, col].grid(alpha=0.3)

plt.suptitle('All Models: Actual vs Predicted PSA Levels', fontsize=16, y=0.995, weight='bold')
plt.tight_layout()
plt.show()

print("\n📊 Quick Analysis:")
print("   • Look at how tightly the points cluster around the red line")
print("   • Tighter clustering = better predictions")
print("   • Compare the R² scores to see which model performs best overall")

input("\nPress Enter to see final summary...")

# ============================================================================
# SECTION 12: FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("📊 FINAL SUMMARY: YOUR MACHINE LEARNING JOURNEY")
print("="*70)
print("\n🎓 What You Accomplished:\n")
print("   ✅ Loaded and explored a real medical dataset")
print("   ✅ Visualized patterns and relationships in the data")
print("   ✅ Built 5 different machine learning models:")
print("      • Linear Regression (classical ML)")
print("      • Decision Tree (rule-based ML)")
print("      • Random Forest (ensemble ML)")
print("      • Gradient Boosting (sequential ensemble ML)")
print("      • Neural Network (deep learning)")
print("   ✅ Evaluated and compared model performance")
print("   ✅ Identified important features for PSA prediction")
print("\n🏆 Best Performing Model:\n")
print(f"   {best_model['Model']}")
print(f"   • R² Score: {best_model['R² Score']:.4f}")
print(f"   • MSE: {best_model['MSE']:.4f}")
print("\n💡 Key Insights:\n")
print("   • Cancer volume is the strongest predictor of PSA levels")
print("   • Ensemble methods (Random Forest, Gradient Boosting) often perform well")
print("   • Even simple models (Linear Regression) can be effective")
print("   • Model selection depends on your goals (accuracy vs interpretability)")
print("\n🚀 Next Steps:\n")
print("   • Work with larger datasets")
print("   • Explore image-based ML (CNNs for medical imaging)")
print("   • Write your research abstract with confidence")
print("   • Keep learning and experimenting!")
print("\n" + "="*70)

print("\n✨ CONGRATULATIONS! ✨")
print("\nYou've completed your first machine learning project!")
print("\n💡 Remember: Every expert was once a beginner.")
print("   You've got this! Keep going! 💪\n")

print("="*70)
print("Script completed! You can now run sections individually or")
print("modify the code to experiment with different parameters.")
print("="*70)
