"""
Fraud Detection Neural Network Model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import List
import numpy as np

from config import MODEL_CONFIG


def create_fraud_detection_model(input_dim: int) -> keras.Model:
    """
    Create a neural network for fraud detection
    
    Args:
        input_dim: Number of input features
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
    ])
    
    # Hidden layers
    for units in MODEL_CONFIG['hidden_layers']:
        model.add(layers.Dense(
            units,
            activation=MODEL_CONFIG['activation'],
            kernel_initializer='he_normal'
        ))
        model.add(layers.Dropout(MODEL_CONFIG['dropout_rate']))
    
    # Output layer
    model.add(layers.Dense(
        1,
        activation=MODEL_CONFIG['output_activation']
    ))
    
    return model


def compile_model(model: keras.Model, learning_rate: float = None) -> keras.Model:
    """
    Compile the model with optimizer and loss
    
    Args:
        model: Keras model
        learning_rate: Learning rate (uses config default if None)
        
    Returns:
        Compiled model
    """
    lr = learning_rate or MODEL_CONFIG['learning_rate']
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


class FraudDetectionModel:
    """Wrapper class for fraud detection model"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = create_fraud_detection_model(input_dim)
        self.model = compile_model(self.model)
        
    def get_weights(self) -> List[np.ndarray]:
        """Get model weights"""
        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]):
        """Set model weights"""
        self.model.set_weights(weights)
        
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 5, batch_size: int = 32,
              class_weight: dict = None, verbose: int = 0) -> dict:
        """
        Train the model
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of epochs
            batch_size: Batch size
            class_weight: Class weights for imbalanced data
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=verbose,
            validation_split=0.1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions (probabilities)
        """
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary of metrics
        """
        results = self.model.evaluate(X, y, verbose=0)
        metric_names = ['loss', 'accuracy', 'precision', 'recall', 'auc']
        
        return dict(zip(metric_names, results))
    
    def save(self, filepath: str):
        """Save model"""
        self.model.save(filepath)
        
    def load(self, filepath: str):
        """Load model"""
        self.model = keras.models.load_model(filepath)


if __name__ == "__main__":
    # Test model creation
    print("Creating fraud detection model...")
    model = FraudDetectionModel(input_dim=30)
    print(model.model.summary())
    
    # Test with random data
    X_test = np.random.randn(100, 30)
    y_test = np.random.randint(0, 2, 100)
    
    print("\nTesting prediction...")
    predictions = model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    
    print("\nModel created successfully!")
