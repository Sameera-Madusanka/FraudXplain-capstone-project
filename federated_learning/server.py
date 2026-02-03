"""
Federated Learning Server
Orchestrates federated training and aggregates client updates
"""

import numpy as np
from typing import List, Dict, Tuple
import copy

from models.fraud_detector import FraudDetectionModel
from federated_learning.aggregation import federated_averaging
from config import FL_CONFIG, PRIVACY_CONFIG


class FederatedServer:
    """
    Federated Learning Server
    Coordinates training across clients and aggregates model updates
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize federated server
        
        Args:
            input_dim: Input dimension for model
        """
        self.input_dim = input_dim
        
        # Create global model
        self.global_model = FraudDetectionModel(input_dim)
        
        # Training history
        self.history = {
            'round': [],
            'train_loss': [],
            'train_accuracy': [],
            'train_auc': [],
            'test_loss': [],
            'test_accuracy': [],
            'test_auc': [],
            'num_clients': []
        }
        
        print(f"Federated Server initialized with global model")
    
    def get_global_weights(self) -> List[np.ndarray]:
        """Get global model weights"""
        return self.global_model.get_weights()
    
    def set_global_weights(self, weights: List[np.ndarray]):
        """Set global model weights"""
        self.global_model.set_weights(weights)
    
    def aggregate_weights(self, client_weights: List[List[np.ndarray]], 
                         client_samples: List[int],
                         aggregation_method: str = 'fedavg') -> List[np.ndarray]:
        """
        Aggregate client weights
        
        Args:
            client_weights: List of weight updates from clients
            client_samples: Number of samples per client
            aggregation_method: Aggregation strategy ('fedavg', 'simple', etc.)
            
        Returns:
            Aggregated weights
        """
        if aggregation_method == 'fedavg':
            return federated_averaging(client_weights, client_samples)
        else:
            # Default to FedAvg
            return federated_averaging(client_weights, client_samples)
    
    def train_round(self, clients: List, round_num: int, 
                   X_test: np.ndarray = None, y_test: np.ndarray = None,
                   verbose: int = 1) -> Dict:
        """
        Execute one round of federated training
        
        Args:
            clients: List of FederatedClient objects
            round_num: Current round number
            X_test: Test features for evaluation
            y_test: Test labels for evaluation
            verbose: Verbosity level
            
        Returns:
            Dictionary of round metrics
        """
        if verbose > 0:
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{FL_CONFIG['num_rounds']}")
            print(f"{'='*60}")
        
        # Select clients for this round
        num_clients_per_round = max(
            int(len(clients) * FL_CONFIG['client_fraction']),
            FL_CONFIG['min_clients']
        )
        selected_clients = np.random.choice(
            clients, 
            size=min(num_clients_per_round, len(clients)), 
            replace=False
        )
        
        if verbose > 0:
            print(f"Selected {len(selected_clients)} clients for training")
        
        # Distribute global model to clients
        global_weights = self.get_global_weights()
        for client in selected_clients:
            client.set_weights(global_weights)
        
        # Train clients locally
        client_weights = []
        client_samples = []
        client_metrics = []
        
        for i, client in enumerate(selected_clients):
            if verbose > 0:
                print(f"\nTraining Client {client.client_id}...")
            
            # Local training
            weights, num_samples, metrics = client.train(
                epochs=FL_CONFIG['local_epochs'],
                batch_size=FL_CONFIG['batch_size'],
                verbose=max(0, verbose-1)
            )
            
            # Apply differential privacy if enabled
            if PRIVACY_CONFIG['use_differential_privacy']:
                weights = self._add_differential_privacy(weights)
            
            client_weights.append(weights)
            client_samples.append(num_samples)
            client_metrics.append(metrics)
        
        # Aggregate client updates
        if verbose > 0:
            print(f"\nAggregating updates from {len(client_weights)} clients...")
        
        aggregated_weights = self.aggregate_weights(
            client_weights, 
            client_samples,
            aggregation_method='fedavg'
        )
        
        # Update global model
        self.set_global_weights(aggregated_weights)
        
        # Compute average training metrics
        avg_metrics = self._average_client_metrics(client_metrics, client_samples)
        
        # Evaluate global model on test set
        test_metrics = {}
        if X_test is not None and y_test is not None:
            test_metrics = self.global_model.evaluate(X_test, y_test)
            if verbose > 0:
                print(f"\nGlobal Model Test Performance:")
                print(f"  Loss: {test_metrics['loss']:.4f}")
                print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"  Precision: {test_metrics['precision']:.4f}")
                print(f"  Recall: {test_metrics['recall']:.4f}")
                print(f"  AUC: {test_metrics['auc']:.4f}")
        
        # Update history
        self.history['round'].append(round_num)
        self.history['train_loss'].append(avg_metrics['loss'])
        self.history['train_accuracy'].append(avg_metrics['accuracy'])
        self.history['train_auc'].append(avg_metrics['auc'])
        self.history['test_loss'].append(test_metrics.get('loss', 0))
        self.history['test_accuracy'].append(test_metrics.get('accuracy', 0))
        self.history['test_auc'].append(test_metrics.get('auc', 0))
        self.history['num_clients'].append(len(selected_clients))
        
        return {
            'round': round_num,
            'train_metrics': avg_metrics,
            'test_metrics': test_metrics,
            'num_clients': len(selected_clients)
        }
    
    def _average_client_metrics(self, client_metrics: List[Dict], 
                                client_samples: List[int]) -> Dict:
        """
        Compute weighted average of client metrics
        
        Args:
            client_metrics: List of metric dictionaries from clients
            client_samples: Number of samples per client
            
        Returns:
            Averaged metrics
        """
        total_samples = sum(client_samples)
        avg_metrics = {}
        
        # Get metric names from first client
        metric_names = client_metrics[0].keys()
        
        for metric_name in metric_names:
            weighted_sum = sum(
                metrics[metric_name] * num_samples 
                for metrics, num_samples in zip(client_metrics, client_samples)
            )
            avg_metrics[metric_name] = weighted_sum / total_samples
        
        return avg_metrics
    
    def _add_differential_privacy(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add differential privacy noise to weights
        
        Args:
            weights: Model weights
            
        Returns:
            Noisy weights
        """
        noisy_weights = []
        noise_multiplier = PRIVACY_CONFIG['noise_multiplier']
        
        for w in weights:
            # Clip gradients (L2 norm clipping)
            l2_norm = np.linalg.norm(w)
            if l2_norm > PRIVACY_CONFIG['l2_norm_clip']:
                w = w * PRIVACY_CONFIG['l2_norm_clip'] / l2_norm
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_multiplier, w.shape)
            noisy_weights.append(w + noise)
        
        return noisy_weights
    
    def get_history(self) -> Dict:
        """Get training history"""
        return self.history
    
    def save_model(self, filepath: str):
        """Save global model"""
        self.global_model.save(filepath)
        print(f"Global model saved to {filepath}")


if __name__ == "__main__":
    # Test server creation
    print("Testing Federated Server...")
    
    # Create server
    server = FederatedServer(input_dim=30)
    
    # Get global weights
    weights = server.get_global_weights()
    print(f"\nGlobal model has {len(weights)} weight arrays")
    
    print("\nServer created successfully!")
