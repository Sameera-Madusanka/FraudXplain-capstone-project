"""
Data Loading and Preprocessing Module
Handles fraud detection dataset loading, preprocessing, and distribution among federated clients
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from typing import List, Tuple, Dict
import os

from config import DATA_CONFIG


class FraudDataLoader:
    """Loads and preprocesses fraud detection dataset"""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or DATA_CONFIG['dataset_path']
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load fraud detection dataset
        If dataset doesn't exist, generate synthetic data
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if os.path.exists(self.dataset_path):
            print(f"Loading dataset from {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
        else:
            print("Dataset not found. Generating synthetic fraud data...")
            df = self._generate_synthetic_data()
            
        # Separate features and target
        if 'Class' in df.columns:
            X = df.drop('Class', axis=1).values
            y = df['Class'].values
        else:
            # Assume last column is target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=DATA_CONFIG['test_size'],
            random_state=DATA_CONFIG['random_state'],
            stratify=y
        )
        
        # Normalize features
        if DATA_CONFIG['normalize']:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"Fraud rate in training: {y_train.mean():.4f}")
        print(f"Fraud rate in test: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def _generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic fraud detection data
        Simulates credit card transaction features
        """
        np.random.seed(DATA_CONFIG['random_state'])
        
        # Generate features similar to credit card fraud dataset
        n_features = 30
        
        # Normal transactions (99%)
        n_normal = int(n_samples * 0.99)
        X_normal = np.random.randn(n_normal, n_features)
        y_normal = np.zeros(n_normal)
        
        # Fraudulent transactions (1%)
        n_fraud = n_samples - n_normal
        # Fraudulent transactions have different distribution
        X_fraud = np.random.randn(n_fraud, n_features) * 2 + 1
        y_fraud = np.ones(n_fraud)
        
        # Combine
        X = np.vstack([X_normal, X_fraud])
        y = np.hstack([y_normal, y_fraud])
        
        # Create DataFrame
        feature_names = [f'V{i}' for i in range(1, n_features)]
        feature_names.append('Amount')
        df = pd.DataFrame(X, columns=feature_names)
        df['Class'] = y
        
        # Shuffle
        df = df.sample(frac=1, random_state=DATA_CONFIG['random_state']).reset_index(drop=True)
        
        print(f"Generated {n_samples} synthetic transactions")
        return df
    
    def handle_imbalance(self, X: np.ndarray, y: np.ndarray, method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance in dataset
        
        Args:
            X: Features
            y: Labels
            method: 'smote', 'undersample', or 'none'
            
        Returns:
            Balanced X, y
        """
        if method == 'smote':
            print("Applying SMOTE to balance dataset...")
            smote = SMOTE(random_state=DATA_CONFIG['random_state'])
            X_balanced, y_balanced = smote.fit_resample(X, y)
            print(f"After SMOTE: {X_balanced.shape[0]} samples")
            return X_balanced, y_balanced
        elif method == 'undersample':
            # Simple random undersampling of majority class
            fraud_idx = np.where(y == 1)[0]
            normal_idx = np.where(y == 0)[0]
            
            # Sample same number of normal as fraud
            normal_idx_sampled = np.random.choice(normal_idx, len(fraud_idx), replace=False)
            
            balanced_idx = np.hstack([fraud_idx, normal_idx_sampled])
            np.random.shuffle(balanced_idx)
            
            return X[balanced_idx], y[balanced_idx]
        else:
            return X, y


class FederatedDataDistributor:
    """Distributes data among federated clients"""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        
    def distribute_iid(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Distribute data in IID (Independent and Identically Distributed) manner
        Each client gets random subset of data
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            List of (X_client, y_client) tuples
        """
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        # Split indices among clients
        client_indices = np.array_split(indices, self.num_clients)
        
        client_data = []
        for i, idx in enumerate(client_indices):
            X_client = X[idx]
            y_client = y[idx]
            fraud_rate = y_client.mean()
            print(f"Client {i+1}: {len(X_client)} samples, fraud rate: {fraud_rate:.4f}")
            client_data.append((X_client, y_client))
            
        return client_data
    
    def distribute_non_iid(self, X: np.ndarray, y: np.ndarray, 
                          concentration: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Distribute data in non-IID manner
        Some clients may have different fraud rates (more realistic)
        
        Args:
            X: Features
            y: Labels
            concentration: Controls how non-IID the distribution is (0-1)
            
        Returns:
            List of (X_client, y_client) tuples
        """
        # Separate fraud and normal transactions
        fraud_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
        
        # Shuffle
        np.random.shuffle(fraud_idx)
        np.random.shuffle(normal_idx)
        
        # Distribute with varying fraud rates
        client_data = []
        fraud_splits = np.array_split(fraud_idx, self.num_clients)
        normal_splits = np.array_split(normal_idx, self.num_clients)
        
        for i in range(self.num_clients):
            # Combine fraud and normal for this client
            client_idx = np.hstack([fraud_splits[i], normal_splits[i]])
            np.random.shuffle(client_idx)
            
            X_client = X[client_idx]
            y_client = y[client_idx]
            fraud_rate = y_client.mean()
            print(f"Client {i+1}: {len(X_client)} samples, fraud rate: {fraud_rate:.4f}")
            client_data.append((X_client, y_client))
            
        return client_data


if __name__ == "__main__":
    # Test data loading
    loader = FraudDataLoader()
    X_train, X_test, y_train, y_test = loader.load_data()
    
    # Test federated distribution
    distributor = FederatedDataDistributor(num_clients=5)
    client_data = distributor.distribute_iid(X_train, y_train)
    
    print(f"\nSuccessfully distributed data to {len(client_data)} clients")
