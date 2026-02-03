"""
Visualization utilities for fraud detection and explanations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot federated learning training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    rounds = history['round']
    
    # Loss
    axes[0, 0].plot(rounds, history['train_loss'], 'b-', label='Train Loss', marker='o')
    axes[0, 0].plot(rounds, history['test_loss'], 'r-', label='Test Loss', marker='s')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(rounds, history['train_accuracy'], 'b-', label='Train Accuracy', marker='o')
    axes[0, 1].plot(rounds, history['test_accuracy'], 'r-', label='Test Accuracy', marker='s')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Test Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(rounds, history['train_auc'], 'b-', label='Train AUC', marker='o')
    axes[1, 0].plot(rounds, history['test_auc'], 'r-', label='Test AUC', marker='s')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].set_title('Training and Test AUC-ROC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Number of clients
    axes[1, 1].plot(rounds, history['num_clients'], 'g-', marker='o')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Number of Clients')
    axes[1, 1].set_title('Clients per Round')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         save_path: str = None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Fraud Detection')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                  save_path: str = None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_counterfactual_comparison(original: np.ndarray, 
                                   counterfactual: np.ndarray,
                                   feature_names: List[str],
                                   changes: List[Dict],
                                   save_path: str = None):
    """
    Visualize counterfactual explanation
    
    Args:
        original: Original instance
        counterfactual: Counterfactual instance
        feature_names: Feature names
        changes: List of feature changes
        save_path: Path to save figure
    """
    # Show top changed features
    top_changes = changes[:10]  # Top 10 changes
    
    if not top_changes:
        print("No significant changes to visualize")
        return
    
    feature_indices = [feature_names.index(c['feature']) for c in top_changes]
    feature_labels = [c['feature'] for c in top_changes]
    
    original_values = [original[i] for i in feature_indices]
    cf_values = [counterfactual[i] for i in feature_indices]
    
    x = np.arange(len(feature_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.barh(x - width/2, original_values, width, label='Original', alpha=0.8)
    bars2 = ax.barh(x + width/2, cf_values, width, label='Counterfactual', alpha=0.8)
    
    ax.set_ylabel('Features')
    ax.set_xlabel('Feature Values')
    ax.set_title('Counterfactual Explanation: Feature Changes')
    ax.set_yticks(x)
    ax.set_yticklabels(feature_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Counterfactual comparison saved to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_importance: Dict, top_n: int = 15,
                           save_path: str = None):
    """
    Plot feature importance
    
    Args:
        feature_importance: Dictionary of feature names and importance scores
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    features = [f[0] for f in sorted_features]
    importance = [f[1] for f in sorted_features]
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if i < 0 else 'green' for i in importance]
    plt.barh(features, importance, color=colors, alpha=0.7)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title(f'Top {top_n} Feature Importance')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def plot_client_data_distribution(client_data: List, save_path: str = None):
    """
    Visualize data distribution across federated clients
    
    Args:
        client_data: List of (X, y) tuples for each client
        save_path: Path to save figure
    """
    num_clients = len(client_data)
    
    client_sizes = [len(X) for X, y in client_data]
    fraud_rates = [y.mean() for X, y in client_data]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sample sizes
    axes[0].bar(range(1, num_clients + 1), client_sizes, alpha=0.7, color='skyblue')
    axes[0].set_xlabel('Client ID')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Data Distribution Across Clients')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Fraud rates
    axes[1].bar(range(1, num_clients + 1), fraud_rates, alpha=0.7, color='coral')
    axes[1].set_xlabel('Client ID')
    axes[1].set_ylabel('Fraud Rate')
    axes[1].set_title('Fraud Rate Across Clients')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Client distribution plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
