"""
Configuration file for Federated Learning Fraud Detection System
Based on: "Transparency and Privacy: The Role of Explainable AI and 
Federated Learning in Financial Fraud Detection"
"""

# Federated Learning Configuration
FL_CONFIG = {
    'num_clients': 5,  # Number of simulated financial institutions
    'num_rounds': 30,  # More rounds for better convergence
    'local_epochs': 10,  # More epochs since balanced distribution gives small client datasets
    'batch_size': 64,  # Larger batch for more stable gradients
    'client_fraction': 1.0,  # Fraction of clients to use per round
    'min_clients': 3,  # Minimum clients required per round
}

# Model Architecture Configuration
MODEL_CONFIG = {
    'input_dim': 30,  # Will be set based on dataset
    'hidden_layers': [256, 128, 64, 32],  # Deeper network for better separation
    'dropout_rate': 0.2,  # Reduced dropout (balanced data doesn't need heavy regularization)
    'activation': 'relu',
    'output_activation': 'sigmoid',
    'learning_rate': 0.0005,  # Lower LR for finer convergence in FL
    'l2_reg': 0.001,  # L2 regularization to prevent overfitting
}

# Training Configuration
TRAINING_CONFIG = {
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',  # BCE works best with federated learning
    'metrics': ['accuracy', 'precision', 'recall', 'auc'],
    # Focal Loss parameters (available but not used - incompatible with FL)
    'focal_loss_gamma': 2.0,
    'focal_loss_alpha': 0.75,
    # No class weights needed - balanced distribution gives 50:50 per client
    'class_weight': {0: 1, 1: 1},
    'validation_split': 0.2,
}

# Privacy Configuration
PRIVACY_CONFIG = {
    'use_differential_privacy': False,  # DISABLED - was destroying model weights
    # When enabled, noise is added to weights EVERY round (noise_multiplier * random)
    # With 20 rounds x 5 clients = 100 noise injections, model can't learn
    # Enable ONLY after confirming model works without it
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
