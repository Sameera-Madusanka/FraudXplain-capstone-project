"""
Main Federated Learning Fraud Detection System
Orchestrates the entire training pipeline
"""

import numpy as np
import os
import sys
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import FL_CONFIG, DATA_CONFIG, OUTPUT_CONFIG
from data_loader import FraudDataLoader, FederatedDataDistributor
from federated_learning.client import FederatedClient
from federated_learning.server import FederatedServer
from explainability.counterfactual_generator import CounterfactualExplainer, SimpleCounterfactualGenerator
from explainability.visualization import (
    plot_training_history, plot_confusion_matrix, 
    plot_roc_curve, plot_client_data_distribution
)
from utils.metrics import evaluate_fraud_detection, print_metrics


def main():
    """Main training pipeline"""
    
    print("="*80)
    print("FEDERATED LEARNING FRAUD DETECTION WITH COUNTERFACTUAL EXPLANATIONS")
    print("Based on: 'Transparency and Privacy: The Role of Explainable AI and")
    print("          Federated Learning in Financial Fraud Detection'")
    print("="*80)
    
    # ========== 1. Load and Distribute Data ==========
    print("\n[1/5] Loading and distributing data...")
    
    loader = FraudDataLoader()
    X_train, X_test, y_train, y_test = loader.load_data()
    
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Distribute data among clients
    distributor = FederatedDataDistributor(num_clients=FL_CONFIG['num_clients'])
    client_data = distributor.distribute_iid(X_train, y_train)
    
    # Visualize data distribution
    print("\nVisualizing client data distribution...")
    plot_client_data_distribution(
        client_data, 
        save_path=os.path.join(OUTPUT_CONFIG['results_path'], 'client_distribution.png')
    )
    
    # ========== 2. Initialize Federated System ==========
    print("\n[2/5] Initializing federated learning system...")
    
    # Create server
    server = FederatedServer(input_dim=input_dim)
    
    # Create clients
    clients = []
    for i, (X_client, y_client) in enumerate(client_data):
        client = FederatedClient(
            client_id=i+1,
            X_train=X_client,
            y_train=y_client,
            input_dim=input_dim
        )
        clients.append(client)
    
    print(f"Created {len(clients)} federated clients")
    
    # ========== 3. Federated Training ==========
    print(f"\n[3/5] Starting federated training for {FL_CONFIG['num_rounds']} rounds...")
    
    for round_num in range(1, FL_CONFIG['num_rounds'] + 1):
        round_results = server.train_round(
            clients=clients,
            round_num=round_num,
            X_test=X_test,
            y_test=y_test,
            verbose=1
        )
        
        # Save model periodically
        if round_num % OUTPUT_CONFIG['save_frequency'] == 0:
            model_path = os.path.join(
                OUTPUT_CONFIG['model_save_path'], 
                f'global_model_round_{round_num}.h5'
            )
            os.makedirs(OUTPUT_CONFIG['model_save_path'], exist_ok=True)
            server.save_model(model_path)
    
    print("\n✓ Federated training complete!")
    
    # ========== 4. Evaluate Global Model ==========
    print("\n[4/5] Evaluating global model...")
    
    metrics = evaluate_fraud_detection(
        server.global_model,
        X_test,
        y_test,
        threshold=0.5
    )
    
    print_metrics(metrics, "Global Model Performance on Test Set")
    
    # Plot training history
    print("\nPlotting training history...")
    history = server.get_history()
    plot_training_history(
        history,
        save_path=os.path.join(OUTPUT_CONFIG['results_path'], 'training_history.png')
    )
    
    # Plot confusion matrix
    y_pred_proba = server.global_model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(OUTPUT_CONFIG['results_path'], 'confusion_matrix.png')
    )
    
    # Plot ROC curve
    plot_roc_curve(
        y_test, y_pred_proba,
        save_path=os.path.join(OUTPUT_CONFIG['results_path'], 'roc_curve.png')
    )
    
    # ========== 5. Generate Counterfactual Explanations ==========
    print("\n[5/5] Generating counterfactual explanations...")
    
    # Find some fraud predictions to explain
    fraud_indices = np.where((y_pred == 1) & (y_test == 1))[0][:5]
    
    if len(fraud_indices) > 0:
        print(f"\nGenerating explanations for {len(fraud_indices)} fraud predictions...")
        
        # Use simple counterfactual generator (DiCE can be complex to set up)
        explainer = SimpleCounterfactualGenerator(
            model=server.global_model,
            feature_names=[f'feature_{i}' for i in range(input_dim)]
        )
        
        for idx in fraud_indices:
            instance = X_test[idx]
            true_label = y_test[idx]
            pred_proba = y_pred_proba[idx]
            
            print(f"\n--- Transaction {idx} ---")
            print(f"True Label: {'FRAUD' if true_label == 1 else 'LEGITIMATE'}")
            print(f"Predicted: FRAUD (probability: {pred_proba:.2%})")
            
            # Generate counterfactual
            cf_result = explainer.generate_counterfactual(
                instance,
                target_class=0,  # Make it legitimate
                max_iterations=50
            )
            
            print(f"Counterfactual prediction: {cf_result['counterfactual_prediction']:.2%}")
            print(f"Explanation: {cf_result['explanation']}")
    else:
        print("No fraud predictions found to explain")
    
    # ========== Summary ==========
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Test AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Test Precision: {metrics['precision']:.4f}")
    print(f"Final Test Recall: {metrics['recall']:.4f}")
    print(f"\nResults saved to: {OUTPUT_CONFIG['results_path']}")
    print(f"Models saved to: {OUTPUT_CONFIG['model_save_path']}")
    print("="*80)


if __name__ == "__main__":
    # Create output directories
    os.makedirs(OUTPUT_CONFIG['results_path'], exist_ok=True)
    os.makedirs(OUTPUT_CONFIG['model_save_path'], exist_ok=True)
    os.makedirs(OUTPUT_CONFIG['log_path'], exist_ok=True)
    
    # Run main pipeline
    main()
