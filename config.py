"""
Configuration file for Federated Learning Fraud Detection System
Based on: "Transparency and Privacy: The Role of Explainable AI and 
Federated Learning in Financial Fraud Detection"
"""

# Federated Learning Configuration
FL_CONFIG = {
    'num_clients': 5,  # Number of simulated financial institutions
    'num_rounds': 20,  # Number of federated learning rounds
    'local_epochs': 5,  # Local training epochs per round
    'batch_size': 32,
    'client_fraction': 1.0,  # Fraction of clients to use per round
    'min_clients': 3,  # Minimum clients required per round
}

# Model Architecture Configuration
MODEL_CONFIG = {
    'input_dim': 30,  # Will be set based on dataset
    'hidden_layers': [128, 64, 32],  # Hidden layer sizes
    'dropout_rate': 0.3,
    'activation': 'relu',
    'output_activation': 'sigmoid',
    'learning_rate': 0.001,
}

# Training Configuration
TRAINING_CONFIG = {
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy', 'precision', 'recall', 'auc'],
    # NO SMOTE - use class weights to handle ~1% fraud rate
    # Weight fraud errors 50x more than legitimate errors
    # This compensates for the 99:1 imbalance without synthetic data
    'class_weight': {0: 1, 1: 50},  # 50x weight for fraud class
    'validation_split': 0.2,
}

# Privacy Configuration
PRIVACY_CONFIG = {
    'use_differential_privacy': True,
    'noise_multiplier': 0.1,  # Noise scale for differential privacy
    'l2_norm_clip': 1.0,  # Gradient clipping threshold
    'delta': 1e-5,  # Privacy parameter
}

# Counterfactual Explanation Configuration
EXPLANATION_CONFIG = {
    'num_counterfactuals': 3,  # Number of counterfactuals to generate
    'desired_class': 0,  # Target class (0 = legitimate)
    'proximity_weight': 0.5,
    'diversity_weight': 1.0,
    'categorical_features': [],  # Will be set based on dataset
    'continuous_features': [],  # Will be set based on dataset
}

# Data Configuration
DATA_CONFIG = {
    'dataset_path': 'data/creditcard.csv',  # Path to fraud dataset
    'test_size': 0.3,
    'random_state': 42,
    'normalize': True,
    'handle_imbalance': 'class_weight',  # Options: 'smote', 'class_weight', 'undersample'
}

# Logging and Output
OUTPUT_CONFIG = {
    'model_save_path': 'models/saved_models/',
    'results_path': 'results/',
    'log_path': 'logs/',
    'verbose': 1,
    'save_frequency': 5,  # Save model every N rounds
}
