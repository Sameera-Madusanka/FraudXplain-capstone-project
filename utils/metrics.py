"""
Utility functions for metrics and evaluation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from typing import Dict, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: np.ndarray = None) -> Dict:
    """
    Calculate comprehensive evaluation metrics for fraud detection
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['auc_roc'] = 0.0
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    # Specificity (True Negative Rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # False Positive Rate
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Evaluation Metrics"):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Main metrics
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}")
    
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # Confusion matrix
    if all(k in metrics for k in ['true_positives', 'true_negatives', 
                                   'false_positives', 'false_negatives']):
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
    
    print(f"{'='*60}\n")


def evaluate_fraud_detection(model, X_test: np.ndarray, y_test: np.ndarray,
                            threshold: float = 0.5) -> Dict:
    """
    Comprehensive evaluation for fraud detection model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    return metrics


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing metrics utilities...")
    
    # Create dummy predictions
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.9, 0.4, 0.3, 0.8, 0.6, 0.7, 0.85, 0.15])
    
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    print_metrics(metrics, "Test Metrics")
    
    print("Metrics utilities working correctly!")
