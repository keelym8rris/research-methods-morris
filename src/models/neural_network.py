"""
Neural Network Model
====================

Neural Networks (also called Deep Learning) are inspired by the human brain.
They consist of layers of artificial neurons that can learn complex patterns.

Structure:
    Input Layer → Hidden Layers → Output Layer
    [Features]  →  [Processing]  → [Prediction]

How it works:
1. Input layer receives features (age, cancer volume, etc.)
2. Hidden layers process information through multiple neurons
3. Output layer produces final prediction (PSA level)
4. During training, weights are adjusted to minimize error

Advantages:
- Can learn incredibly complex non-linear patterns
- State-of-the-art for images, text, and speech
- Flexible architecture (can be customized)
- Can improve with more data

Disadvantages:
- Needs more data to train effectively
- Harder to interpret ("black box")
- Requires more computational power
- Can overfit on small datasets
- Many hyperparameters to tune

When to use:
- When you have large amounts of data
- For complex pattern recognition
- When other methods plateau in performance
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
sys.path.append('.')

from src.data_loader import load_prostate_data, prepare_data
from src.evaluation import evaluate_model, print_evaluation
from src.visualization import plot_predictions, plot_residuals, save_figure


def train_neural_network(data_dict, epochs=100, batch_size=32, verbose=True):
    """
    Train a Neural Network model for PSA prediction.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary from prepare_data() containing training and test sets
    epochs : int
        Number of times to train on the full dataset
    batch_size : int
        Number of samples per gradient update
    verbose : bool
        Whether to print detailed output
        
    Returns:
    --------
    tuple
        (model, predictions, metrics, history)
    """
    if verbose:
        print("\n" + "="*70)
        print("🧠 NEURAL NETWORK MODEL")
        print("="*70)
        print(f"\nBuilding neural network architecture...")
    
    # Build the neural network
    model = Sequential([
        Dense(64, activation='relu', input_shape=(data_dict['X_train_scaled'].shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    if verbose:
        print("\n🏗️ Neural Network Architecture:")
        print("="*70)
        model.summary()
        print("="*70)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    
    if verbose:
        print(f"\n🎓 Training neural network for {epochs} epochs...")
        print("(Network will learn from the data multiple times)")
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        data_dict['X_train_scaled'], 
        data_dict['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0  # Suppress epoch-by-epoch output
    )
    
    if verbose:
        print("✅ Training complete!")
        print(f"\nFinal training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Stopped at epoch: {len(history.history['loss'])}/{epochs}")
    
    # Make predictions
    y_pred = model.predict(data_dict['X_test_scaled'], verbose=0)
    y_pred = y_pred.flatten()
    
    # Evaluate the model
    metrics = evaluate_model(data_dict['y_test'], y_pred, "Neural Network")
    
    if verbose:
        print_evaluation(metrics)
    
    return model, y_pred, metrics, history


def plot_training_history(history, save=True):
    """
    Plot the training history showing how loss decreased over epochs.
    
    Parameters:
    -----------
    history : History object
        Training history returned by model.fit()
    save : bool
        Whether to save the figure
    """
    plt.figure(figsize=(12, 5))
    
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch (Training Iteration)', fontsize=12)
    plt.ylabel('Loss (Mean Squared Error)', fontsize=12)
    plt.title('Neural Network Training Progress\n(Loss should decrease over time)', 
              fontsize=14, pad=20, weight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save:
        save_figure('neural_network_training_history.png')
    
    plt.show()
    
    print("\n📉 Training History:")
    print("   • Both lines decreasing = network is learning!")
    print("   • Training loss < validation loss = normal")
    print("   • If validation loss increases while training decreases = overfitting")


def analyze_network(model, history):
    """
    Analyze and display information about the neural network.
    
    Parameters:
    -----------
    model : Sequential
        Trained neural network model
    history : History object
        Training history
    """
    print("\n🧠 Neural Network Details:")
    print("="*70)
    
    # Count parameters
    total_params = model.count_params()
    print(f"   Total trainable parameters: {total_params:,}")
    print(f"   Number of layers: {len(model.layers)}")
    print(f"   Training epochs completed: {len(history.history['loss'])}")
    print(f"   Best validation loss: {min(history.history['val_loss']):.4f}")
    
    print("="*70)


if __name__ == "__main__":
    print("\n🚀 Running Neural Network Analysis")
    print("="*70)
    
    # Load and prepare data
    data = load_prostate_data()
    data_dict = prepare_data(data)
    
    # Train model
    model, predictions, metrics, history = train_neural_network(data_dict)
    
    # Analyze network
    analyze_network(model, history)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    plot_training_history(history, save=True)
    
    plot_predictions(
        data_dict['y_test'],
        predictions,
        "Neural Network",
        metrics['r2_score'],
        save=True
    )
    
    plot_residuals(
        data_dict['y_test'],
        predictions,
        "Neural Network",
        save=True
    )
    
    print("\n✅ Neural Network analysis complete!")
    print(f"🏆 Final R² Score: {metrics['r2_score']:.4f}")
    
    # Experiment suggestions
    print("\n💡 Want to experiment? Try different architectures:")
    print("   # Modify the layers in train_neural_network()")
    print("   # Try: Dense(128), Dense(64), Dense(32), Dense(1)")
    print("   # Or add dropout: Dense(64), Dropout(0.2), Dense(32)...")
