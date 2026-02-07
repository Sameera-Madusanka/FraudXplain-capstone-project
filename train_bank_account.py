"""
End-to-End Training: Federated Learning Fraud Detection with Constrained Counterfactuals
Bank Account Fraud Dataset (NeurIPS 2022)
"""

import numpy as np
import argparse
import os
from datetime import datetime

# Import our modules
from data_loader_bank import BankAccountFraudLoader, FederatedBankAccountDistributor
from models.fraud_detector import FraudDetectionModel
from federated_learning.client import FederatedClient
from federated_learning.server import FederatedServer
from explainability.constrained_counterfactuals import ConstrainedCounterfactualGenerator
from explainability.privacy_validator import PrivacyValidator, FeasibilityChecker
from explainability.actionable_recourse import ActionableRecourseGenerator
from explainability.visualization import (
    plot_training_history, plot_confusion_matrix, plot_roc_curve
)
from utils.metrics import evaluate_fraud_detection, print_metrics
from config import FL_CONFIG, MODEL_CONFIG


def main(args):
    """
    Main training pipeline
    """
    print("\n" + "="*80)
    print("FEDERATED LEARNING FRAUD DETECTION")
    print("Bank Account Fraud Dataset with Constrained Counterfactual Explanations")
    print("="*80 + "\n")
    
    # Set random seed
    np.random.seed(42)
    
    # ========================================================================
    # STEP 1: Load and Prepare Data
    # ========================================================================
    print("📊 STEP 1: Loading Bank Account Fraud Dataset...")
    print("-" * 80)
    
    loader = BankAccountFraudLoader(
        dataset_path='data/Base.csv',
        variant='Base',
        test_size=0.2,
        random_state=42
    )
    
    # Load data (use sample for quick testing)
    sample_size = args.sample_size if args.sample_size else None
    X_train, X_test, y_train, y_test, feature_names = loader.load_and_split(
        sample_size=sample_size,
        balance_classes=True
    )
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Features: {len(feature_names)}")
    
    # ========================================================================
    # STEP 2: Distribute Data Across Federated Clients
    # ========================================================================
    print(f"\n📡 STEP 2: Distributing Data to {args.num_clients} Federated Clients...")
    print("-" * 80)
    
    distributor = FederatedBankAccountDistributor(
        num_clients=args.num_clients,
        distribution='iid'
    )
    
    client_data = distributor.distribute_data(X_train, y_train)
    
    # ========================================================================
    # STEP 3: Initialize Federated Learning
    # ========================================================================
    print(f"\n🔧 STEP 3: Initializing Federated Learning...")
    print("-" * 80)
    
    # Create global model and federated server
    input_dim = X_train.shape[1]
    
    # Create federated server (it creates its own global model)
    server = FederatedServer(input_dim=input_dim)
    
    # Create federated clients
    clients = []
    for i, (X_client, y_client) in enumerate(client_data):
        client = FederatedClient(
            client_id=i+1,
            X_train=X_client,
            y_train=y_client,
            input_dim=input_dim
        )
        clients.append(client)
    
    print(f"✅ Initialized {len(clients)} federated clients")
    
    # ========================================================================
    # STEP 4: Federated Training
    # ========================================================================
    print(f"\n🚀 STEP 4: Starting Federated Training ({args.rounds} rounds)...")
    print("-" * 80)
    
    for round_num in range(1, args.rounds + 1):
        print(f"\n📍 Round {round_num}/{args.rounds}")
        
        # Train round
        history = server.train_round(
            clients=clients,
            round_num=round_num,
            X_test=X_test,
            y_test=y_test,
            verbose=1
        )
        
        # Print metrics
        if history['test_metrics']:
            print(f"   Global Model - AUC: {history['test_metrics']['auc']:.4f}, "
                  f"Precision: {history['test_metrics']['precision']:.4f}, "
                  f"Recall: {history['test_metrics']['recall']:.4f}")
    
    print(f"\n✅ Federated training complete!")
    
    # ========================================================================
    # STEP 5: Final Evaluation
    # ========================================================================
    print(f"\n📈 STEP 5: Final Model Evaluation...")
    print("-" * 80)
    
    final_metrics = evaluate_fraud_detection(
        server.global_model.model,
        X_test,
        y_test
    )
    
    print_metrics(final_metrics)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Plot training history
    if server.history:
        plot_training_history(
            server.history,
            save_path='results/training_history.png'
        )
        print("   ✓ Saved training history plot")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_test,
        (server.global_model.predict(X_test) > 0.5).astype(int),
        save_path='results/confusion_matrix.png'
    )
    print("   ✓ Saved confusion matrix")
    
    # Plot ROC curve
    plot_roc_curve(
        y_test,
        server.global_model.predict(X_test),
        save_path='results/roc_curve.png'
    )
    print("   ✓ Saved ROC curve")
    
    # ========================================================================
    # STEP 6: Constrained Counterfactual Explanations
    # ========================================================================
    print(f"\n🔒 STEP 6: Generating Constrained Counterfactual Explanations...")
    print("-" * 80)
    
    # Initialize constrained CF generator
    cf_generator = ConstrainedCounterfactualGenerator(
        model=server.global_model,
        feature_names=feature_names,
        protected_attrs_config='data/protected_attributes.json'
    )
    
    # Find a fraudulent example from test set
    fraud_indices = np.where(y_test == 1)[0]
    if len(fraud_indices) > 0:
        fraud_idx = fraud_indices[0]
        fraud_instance = X_test[fraud_idx]
        
        print(f"\n📋 Generating explanation for fraudulent transaction #{fraud_idx}...")
        
        # Generate constrained counterfactuals
        counterfactuals = cf_generator.generate_constrained_counterfactual(
            instance=fraud_instance,
            target_class=0,  # Legitimate
            max_iterations=100,
            num_counterfactuals=3
        )
        
        # Generate privacy-guaranteed explanation
        explanation = cf_generator.generate_privacy_guaranteed_explanation(
            instance=fraud_instance,
            counterfactuals=counterfactuals
        )
        
        print(explanation)
        
        # Validate privacy
        validator = PrivacyValidator(
            protected_attributes=cf_generator.config['protected_attributes']['attributes'],
            feature_names=feature_names
        )
        
        for i, cf in enumerate(counterfactuals):
            is_valid, violations = validator.validate_counterfactual(
                fraud_instance,
                cf['counterfactual']
            )
            
            if is_valid:
                print(f"   ✅ Counterfactual {i+1}: Privacy validated")
            else:
                print(f"   ❌ Counterfactual {i+1}: Privacy violations detected")
                for violation in violations:
                    print(f"      - {violation}")
        
        # Generate actionable recourse
        recourse_gen = ActionableRecourseGenerator(feature_names)
        recourse = recourse_gen.generate_recourse(
            original_instance=fraud_instance,
            counterfactuals=counterfactuals,
            original_pred=server.global_model.predict(fraud_instance.reshape(1, -1))[0][0]
        )
        
        print(recourse)
        
        # Save explanation to file
        with open('results/example_explanation.txt', 'w', encoding='utf-8') as f:
            f.write(explanation)
            f.write("\n\n")
            f.write(recourse)
        
        print("   ✓ Saved example explanation to results/example_explanation.txt")
    
    # ========================================================================
    # STEP 7: Save Model
    # ========================================================================
    print(f"\n💾 STEP 7: Saving Model...")
    print("-" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'results/fraud_model_{timestamp}.h5'
    server.global_model.save(model_path)
    print(f"   ✓ Model saved to {model_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"   AUC-ROC: {final_metrics['auc_roc']:.4f}")
    print(f"   Precision: {final_metrics['precision']:.4f}")
    print(f"   Recall: {final_metrics['recall']:.4f}")
    print(f"   F1-Score: {final_metrics['f1_score']:.4f}")
    print(f"\n📁 Results saved to: results/")
    print(f"   - Training history plot")
    print(f"   - Confusion matrix")
    print(f"   - ROC curve")
    print(f"   - Example explanation")
    print(f"   - Trained model")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Federated Fraud Detection with Constrained Counterfactuals'
    )
    
    parser.add_argument(
        '--num-clients',
        type=int,
        default=5,
        help='Number of federated clients (default: 5)'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=10,
        help='Number of federated training rounds (default: 10)'
    )
    
    parser.add_argument(
        '--local-epochs',
        type=int,
        default=5,
        help='Number of local training epochs per round (default: 5)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Sample size for quick testing (default: None, use full dataset)'
    )
    
    args = parser.parse_args()
    
    main(args)
