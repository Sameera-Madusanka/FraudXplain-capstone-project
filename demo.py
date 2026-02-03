"""
Interactive Demo for Federated Learning Fraud Detection
Demonstrates the system with visualizations and explanations
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import FL_CONFIG, DATA_CONFIG
from data_loader import FraudDataLoader, FederatedDataDistributor
from federated_learning.client import FederatedClient
from federated_learning.server import FederatedServer
from explainability.counterfactual_generator import SimpleCounterfactualGenerator
from utils.metrics import evaluate_fraud_detection, print_metrics


def run_quick_demo(num_rounds: int = 10, num_clients: int = 3):
    """
    Run a quick demonstration of the system
    
    Args:
        num_rounds: Number of federated learning rounds
        num_clients: Number of simulated clients
    """
    print("="*80)
    print("FEDERATED LEARNING FRAUD DETECTION - QUICK DEMO")
    print("="*80)
    
    # Load data
    print("\n📊 Loading fraud detection dataset...")
    loader = FraudDataLoader()
    X_train, X_test, y_train, y_test = loader.load_data()
    input_dim = X_train.shape[1]
    
    # Distribute to clients
    print(f"\n🌐 Distributing data to {num_clients} financial institutions...")
    distributor = FederatedDataDistributor(num_clients=num_clients)
    client_data = distributor.distribute_iid(X_train, y_train)
    
    # Initialize federated system
    print("\n🔧 Initializing federated learning system...")
    server = FederatedServer(input_dim=input_dim)
    
    clients = []
    for i, (X_client, y_client) in enumerate(client_data):
        client = FederatedClient(
            client_id=i+1,
            X_train=X_client,
            y_train=y_client,
            input_dim=input_dim
        )
        clients.append(client)
    
    # Federated training
    print(f"\n🚀 Starting federated training ({num_rounds} rounds)...")
    print("Each round: Clients train locally → Server aggregates → Repeat")
    print("-" * 80)
    
    for round_num in range(1, num_rounds + 1):
        round_results = server.train_round(
            clients=clients,
            round_num=round_num,
            X_test=X_test,
            y_test=y_test,
            verbose=1
        )
    
    # Evaluate
    print("\n📈 Evaluating global model...")
    metrics = evaluate_fraud_detection(server.global_model, X_test, y_test)
    print_metrics(metrics, "Final Model Performance")
    
    # Demonstrate counterfactual explanations
    print("\n🔍 Generating Counterfactual Explanations...")
    print("-" * 80)
    
    # Get predictions
    y_pred_proba = server.global_model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Find fraud predictions
    fraud_indices = np.where(y_pred == 1)[0][:3]
    
    if len(fraud_indices) > 0:
        explainer = SimpleCounterfactualGenerator(
            model=server.global_model,
            feature_names=[f'V{i}' for i in range(1, input_dim+1)]
        )
        
        for i, idx in enumerate(fraud_indices, 1):
            instance = X_test[idx]
            true_label = "FRAUD" if y_test[idx] == 1 else "LEGITIMATE"
            pred_proba = y_pred_proba[idx]
            
            print(f"\n🔴 Example {i}: Transaction flagged as FRAUD")
            print(f"   True Label: {true_label}")
            print(f"   Fraud Probability: {pred_proba:.2%}")
            
            # Generate counterfactual
            cf_result = explainer.generate_counterfactual(
                instance,
                target_class=0,
                max_iterations=30
            )
            
            cf_prob = cf_result['counterfactual_prediction']
            print(f"   ✓ Counterfactual generated!")
            print(f"   → Modified transaction probability: {cf_prob:.2%}")
            
            if cf_prob < 0.5:
                print(f"   → Successfully changed to LEGITIMATE")
            else:
                print(f"   → Still classified as FRAUD (needs more iterations)")
    
    # Summary
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("="*80)
    print("\n📊 Key Results:")
    print(f"   • Trained with {num_clients} federated clients")
    print(f"   • Completed {num_rounds} training rounds")
    print(f"   • Test AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"   • Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"   • Precision: {metrics['precision']:.4f}")
    print(f"   • Recall: {metrics['recall']:.4f}")
    
    print("\n🔒 Privacy Features:")
    print("   • No raw data shared between institutions")
    print("   • Only model updates aggregated")
    print("   • Differential privacy can be enabled")
    
    print("\n💡 Explainability:")
    print("   • Counterfactual explanations generated")
    print("   • Shows minimal changes to flip predictions")
    print("   • Enhances transparency and trust")
    
    print("\n" + "="*80)


def interactive_prediction():
    """Interactive prediction demo"""
    print("\n🎯 Interactive Prediction Demo")
    print("Enter transaction features to get fraud prediction and explanation")
    print("(Type 'quit' to exit)")
    
    # This would require a trained model
    # Placeholder for interactive functionality
    print("\n[Feature not yet implemented - requires trained model]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Learning Fraud Detection Demo')
    parser.add_argument('--rounds', type=int, default=10, help='Number of training rounds')
    parser.add_argument('--clients', type=int, default=3, help='Number of federated clients')
    parser.add_argument('--interactive', action='store_true', help='Run interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_prediction()
    else:
        run_quick_demo(num_rounds=args.rounds, num_clients=args.clients)
