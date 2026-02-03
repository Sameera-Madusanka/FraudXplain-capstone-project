"""
Federated Learning Client
Represents a financial institution training locally on private data
"""

import numpy as np
from typing import List, Tuple
import copy

from models.fraud_detector import FraudDetectionModel
from config import FL_CONFIG, TRAINING_CONFIG


class FederatedClient:
    """
    Federated Learning Client
    Trains model locally on private data without sharing raw data
    """
    
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray, 
                 input_dim: int):
        """
        Initialize federated client
        
        Args:
            client_id: Unique client identifier
            X_train: Local training data (features)
            y_train: Local training labels
            input_dim: Input dimension for model
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.input_dim = input_dim
        
        # Create local model
        self.model = FraudDetectionModel(input_dim)
        
        # Training statistics
        self.num_samples = len(X_train)
        self.fraud_rate = y_train.mean()
        
        print(f"Client {client_id} initialized with {self.num_samples} samples "
              f"(fraud rate: {self.fraud_rate:.4f})")
    
    def get_weights(self) -> List[np.ndarray]:
        """Get current model weights"""
        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]):
        """Set model weights (from server)"""
        self.model.set_weights(weights)
    
    def train(self, epochs: int = None, batch_size: int = None, 
              verbose: int = 0) -> Tuple[List[np.ndarray], int, dict]:
        """
        Train model locally on private data
        
        Args:
            epochs: Number of local training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Tuple of (updated_weights, num_samples, training_metrics)
        """
        epochs = epochs or FL_CONFIG['local_epochs']
        batch_size = batch_size or FL_CONFIG['batch_size']
        
        # Train on local data
        history = self.model.train(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=TRAINING_CONFIG['class_weight'],
            verbose=verbose
        )
        
        # Get updated weights
        updated_weights = self.get_weights()
        
        # Extract final metrics
        metrics = {
            'loss': history['loss'][-1],
            'accuracy': history['accuracy'][-1],
            'precision': history['precision'][-1],
            'recall': history['recall'][-1],
            'auc': history['auc'][-1]
        }
        
        if verbose > 0:
            print(f"Client {self.client_id} training complete - "
                  f"Loss: {metrics['loss']:.4f}, AUC: {metrics['auc']:.4f}")
        
        return updated_weights, self.num_samples, metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        return self.model.evaluate(X_test, y_test)
    
    def add_privacy_noise(self, weights: List[np.ndarray], 
                         noise_multiplier: float = 0.1) -> List[np.ndarray]:
        """
        Add differential privacy noise to weights
        
        Args:
            weights: Model weights
            noise_multiplier: Scale of noise to add
            
        Returns:
            Noisy weights
        """
        noisy_weights = []
        for w in weights:
            noise = np.random.normal(0, noise_multiplier, w.shape)
            noisy_weights.append(w + noise)
        
        return noisy_weights


if __name__ == "__main__":
    # Test client creation
    print("Testing Federated Client...")
    
    # Create dummy data
    X_train = np.random.randn(1000, 30)
    y_train = np.random.randint(0, 2, 1000)
    
    # Create client
    client = FederatedClient(
        client_id=1,
        X_train=X_train,
        y_train=y_train,
        input_dim=30
    )
    
    # Test training
    print("\nTraining client model...")
    weights, num_samples, metrics = client.train(epochs=2, verbose=1)
    
    print(f"\nClient training successful!")
    print(f"Number of weight arrays: {len(weights)}")
    print(f"Metrics: {metrics}")
