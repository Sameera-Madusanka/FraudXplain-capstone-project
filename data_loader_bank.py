"""
Bank Account Fraud Dataset Loader
Loads and preprocesses the Bank Account Fraud Dataset (NeurIPS 2022)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from typing import Tuple, List, Dict
import os


class BankAccountFraudLoader:
    """
    Loads and preprocesses Bank Account Fraud Dataset
    """
    
    # Bank Account Fraud Dataset features (31 features + 1 target)
    FEATURE_COLUMNS = [
        'income', 'name_email_similarity', 'prev_address_months_count',
        'current_address_months_count', 'customer_age', 'days_since_request',
        'intended_balcon_amount', 'payment_type', 'zip_count_4w',
        'velocity_6h', 'velocity_24h', 'velocity_4w', 'bank_branch_count_8w',
        'date_of_birth_distinct_emails_4w', 'employment_status',
        'credit_risk_score', 'email_is_free', 'housing_status',
        'phone_home_valid', 'phone_mobile_valid', 'bank_months_count',
        'has_other_cards', 'proposed_credit_limit', 'foreign_request',
        'source', 'session_length_in_minutes', 'device_os',
        'keep_alive_session', 'device_distinct_emails_8w',
        'device_fraud_count', 'month'
    ]
    
    TARGET_COLUMN = 'fraud_bool'
    
    def __init__(self, dataset_path: str = 'data/Base.csv', 
                 variant: str = 'Base',
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize dataset loader
        
        Args:
            dataset_path: Path to dataset CSV file
            variant: Dataset variant ('Base', 'Variant I', 'Variant II', etc.)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.variant = variant
        self.test_size = test_size
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        print(f"Initializing Bank Account Fraud Loader for {variant}")
    
    def load_data(self, sample_size: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load dataset from CSV
        
        Args:
            sample_size: Optional sample size (for testing with subset)
            
        Returns:
            (features_df, target_series)
        """
        print(f"Loading dataset from {self.dataset_path}...")
        
        if sample_size:
            df = pd.read_csv(self.dataset_path, nrows=sample_size)
            print(f"  Loaded {len(df):,} samples (sample mode)")
        else:
            df = pd.read_csv(self.dataset_path)
            print(f"  Loaded {len(df):,} samples")
        
        # Separate features and target
        X = df[self.FEATURE_COLUMNS]
        y = df[self.TARGET_COLUMN]
        
        print(f"  Features: {X.shape[1]}")
        print(f"  Fraud rate: {y.mean():.4%}")
        
        return X, y
    
    def preprocess(self, X: pd.DataFrame, y: pd.Series = None,
                   fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features
        
        Args:
            X: Features dataframe
            y: Target series (optional)
            fit: Whether to fit scalers/encoders
            
        Returns:
            (X_processed, y_processed)
        """
        X_processed = X.copy()
        
        # Handle categorical features
        categorical_features = [
            'payment_type', 'employment_status', 'housing_status',
            'source', 'device_os'
        ]
        
        for col in categorical_features:
            if col in X_processed.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    X_processed[col] = self.label_encoders[col].fit_transform(
                        X_processed[col].astype(str)
                    )
                else:
                    if col in self.label_encoders:
                        X_processed[col] = self.label_encoders[col].transform(
                            X_processed[col].astype(str)
                        )
        
        # Handle boolean features
        boolean_features = [
            'email_is_free', 'phone_home_valid', 'phone_mobile_valid',
            'has_other_cards', 'foreign_request', 'keep_alive_session'
        ]
        
        for col in boolean_features:
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].astype(int)
        
        # Fill missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = self.scaler.transform(X_processed)
        
        y_processed = y.values if y is not None else None
        
        return X_scaled, y_processed
    
    def load_and_split(self, sample_size: int = None,
                      balance_classes: bool = True) -> Tuple:
        """
        Load, preprocess, and split dataset
        
        Args:
            sample_size: Optional sample size
            balance_classes: Whether to use SMOTE for class balancing
            
        Returns:
            (X_train, X_test, y_train, y_test, feature_names)
        """
        # Load data
        X, y = self.load_data(sample_size)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y
        )
        
        print(f"\nTrain/Test Split:")
        print(f"  Train: {len(X_train):,} samples")
        print(f"  Test: {len(X_test):,} samples")
        
        # Preprocess
        X_train_processed, y_train_processed = self.preprocess(X_train, y_train, fit=True)
        X_test_processed, y_test_processed = self.preprocess(X_test, y_test, fit=False)
        
        # Balance classes if requested
        if balance_classes:
            print(f"\nBalancing classes with SMOTE...")
            print(f"  Before: {np.sum(y_train_processed)} frauds / {len(y_train_processed)} total")
            
            # Use sampling_strategy=0.1 for 10% fraud rate (vs 1% original)
            # This is more realistic than 50-50 balance and reduces false positives
            smote = SMOTE(random_state=self.random_state, sampling_strategy=0.1)
            X_train_processed, y_train_processed = smote.fit_resample(
                X_train_processed, y_train_processed
            )
            
            print(f"  After: {np.sum(y_train_processed)} frauds / {len(y_train_processed)} total")
        
        feature_names = self.FEATURE_COLUMNS
        
        return (X_train_processed, X_test_processed, 
                y_train_processed, y_test_processed, 
                feature_names)


class FederatedBankAccountDistributor:
    """
    Distributes Bank Account Fraud data across federated clients
    """
    
    def __init__(self, num_clients: int = 5, distribution: str = 'iid'):
        """
        Initialize federated distributor
        
        Args:
            num_clients: Number of federated clients
            distribution: 'iid' or 'non-iid'
        """
        self.num_clients = num_clients
        self.distribution = distribution
    
    def distribute_data(self, X: np.ndarray, y: np.ndarray) -> List[Tuple]:
        """
        Distribute data across clients
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            List of (X_client, y_client) tuples
        """
        n_samples = len(X)
        
        if self.distribution == 'iid':
            # IID distribution: random shuffle and split
            indices = np.random.permutation(n_samples)
            split_indices = np.array_split(indices, self.num_clients)
            
            client_data = [
                (X[idx], y[idx]) for idx in split_indices
            ]
        
        elif self.distribution == 'balanced':
            # Balanced distribution: each client gets equal fraud and legitimate
            # This is critical for federated learning with imbalanced data
            fraud_indices = np.where(y == 1)[0]
            legit_indices = np.where(y == 0)[0]
            
            np.random.shuffle(fraud_indices)
            np.random.shuffle(legit_indices)
            
            # Split fraud samples across clients
            fraud_splits = np.array_split(fraud_indices, self.num_clients)
            
            # Each client gets same number of legitimate as fraud samples
            client_data = []
            legit_start = 0
            for i in range(self.num_clients):
                n_fraud = len(fraud_splits[i])
                n_legit = n_fraud  # Equal number of legitimate samples
                
                legit_end = min(legit_start + n_legit, len(legit_indices))
                legit_split = legit_indices[legit_start:legit_end]
                legit_start = legit_end
                
                # Combine fraud and legitimate, shuffle
                combined_idx = np.concatenate([fraud_splits[i], legit_split])
                np.random.shuffle(combined_idx)
                
                client_data.append((X[combined_idx], y[combined_idx]))
        
        elif self.distribution == 'non-iid':
            # Non-IID: sort by a feature and split
            sorted_indices = np.argsort(X[:, 0])
            split_indices = np.array_split(sorted_indices, self.num_clients)
            
            client_data = [
                (X[idx], y[idx]) for idx in split_indices
            ]
        
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        
        print(f"\nDistributed data across {self.num_clients} clients ({self.distribution}):")
        for i, (X_client, y_client) in enumerate(client_data):
            fraud_rate = np.mean(y_client)
            print(f"  Client {i+1}: {len(X_client):,} samples, {fraud_rate:.2%} fraud rate")
        
        return client_data


if __name__ == "__main__":
    print("Bank Account Fraud Dataset Loader")
    print("=" * 70)
    
    # Test loading
    loader = BankAccountFraudLoader(
        dataset_path='data/Base.csv',
        variant='Base'
    )
    
    # Load small sample for testing
    X_train, X_test, y_train, y_test, feature_names = loader.load_and_split(
        sample_size=10000,
        balance_classes=True
    )
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Features: {len(feature_names)}")
    
    # Test federated distribution
    distributor = FederatedBankAccountDistributor(num_clients=5, distribution='iid')
    client_data = distributor.distribute_data(X_train, y_train)
    
    print(f"\n✅ Federated distribution complete!")
