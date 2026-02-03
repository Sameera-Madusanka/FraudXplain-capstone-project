"""
Federated Aggregation Strategies
Implements FedAvg and other aggregation methods
"""

import numpy as np
from typing import List, Tuple


def federated_averaging(client_weights: List[List[np.ndarray]], 
                       client_samples: List[int]) -> List[np.ndarray]:
    """
    Federated Averaging (FedAvg) algorithm
    Aggregates client model weights weighted by number of samples
    
    Args:
        client_weights: List of weight lists from each client
        client_samples: List of number of samples per client
        
    Returns:
        Aggregated weights
    """
    total_samples = sum(client_samples)
    
    # Initialize aggregated weights with zeros
    num_layers = len(client_weights[0])
    aggregated_weights = [np.zeros_like(w) for w in client_weights[0]]
    
    # Weighted average
    for client_w, num_samples in zip(client_weights, client_samples):
        weight = num_samples / total_samples
        for i in range(num_layers):
            aggregated_weights[i] += client_w[i] * weight
    
    return aggregated_weights


def simple_averaging(client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Simple averaging (unweighted)
    Each client contributes equally regardless of data size
    
    Args:
        client_weights: List of weight lists from each client
        
    Returns:
        Aggregated weights
    """
    num_clients = len(client_weights)
    num_layers = len(client_weights[0])
    
    aggregated_weights = [np.zeros_like(w) for w in client_weights[0]]
    
    for client_w in client_weights:
        for i in range(num_layers):
            aggregated_weights[i] += client_w[i] / num_clients
    
    return aggregated_weights


def median_aggregation(client_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Median aggregation (robust to outliers)
    Takes median of weights across clients
    
    Args:
        client_weights: List of weight lists from each client
        
    Returns:
        Aggregated weights
    """
    num_layers = len(client_weights[0])
    aggregated_weights = []
    
    for layer_idx in range(num_layers):
        # Stack weights from all clients for this layer
        layer_weights = np.stack([client_w[layer_idx] for client_w in client_weights])
        # Take median across clients
        median_weights = np.median(layer_weights, axis=0)
        aggregated_weights.append(median_weights)
    
    return aggregated_weights


def trimmed_mean_aggregation(client_weights: List[List[np.ndarray]], 
                             trim_ratio: float = 0.1) -> List[np.ndarray]:
    """
    Trimmed mean aggregation (robust to outliers)
    Removes extreme values before averaging
    
    Args:
        client_weights: List of weight lists from each client
        trim_ratio: Fraction of extreme values to remove from each end
        
    Returns:
        Aggregated weights
    """
    num_layers = len(client_weights[0])
    aggregated_weights = []
    
    for layer_idx in range(num_layers):
        # Stack weights from all clients for this layer
        layer_weights = np.stack([client_w[layer_idx] for client_w in client_weights])
        
        # Sort and trim
        sorted_weights = np.sort(layer_weights, axis=0)
        n_clients = len(client_weights)
        trim_count = int(n_clients * trim_ratio)
        
        if trim_count > 0:
            trimmed_weights = sorted_weights[trim_count:-trim_count]
        else:
            trimmed_weights = sorted_weights
        
        # Average remaining weights
        mean_weights = np.mean(trimmed_weights, axis=0)
        aggregated_weights.append(mean_weights)
    
    return aggregated_weights


if __name__ == "__main__":
    # Test aggregation
    print("Testing aggregation strategies...")
    
    # Create dummy client weights
    num_clients = 3
    client_weights = []
    client_samples = []
    
    for i in range(num_clients):
        # Simulate weights for a small model
        weights = [
            np.random.randn(10, 5),  # Layer 1
            np.random.randn(5),       # Bias 1
            np.random.randn(5, 1),    # Layer 2
            np.random.randn(1)        # Bias 2
        ]
        client_weights.append(weights)
        client_samples.append(100 * (i + 1))  # Different sample sizes
    
    # Test FedAvg
    print("\nTesting FedAvg...")
    aggregated = federated_averaging(client_weights, client_samples)
    print(f"Aggregated weights: {len(aggregated)} layers")
    print(f"First layer shape: {aggregated[0].shape}")
    
    # Test simple averaging
    print("\nTesting Simple Averaging...")
    aggregated_simple = simple_averaging(client_weights)
    print(f"Aggregated weights: {len(aggregated_simple)} layers")
    
    print("\nAggregation tests successful!")
