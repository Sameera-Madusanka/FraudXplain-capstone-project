"""
Compare federated model vs simple model performance
"""
import numpy as np
from models.fraud_detector import FraudDetectionModel
from data_loader_bank import BankAccountFraudLoader
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import glob

print("="*70)
print("MODEL COMPARISON: Federated vs Simple")
print("="*70)

# Load test data
loader = BankAccountFraudLoader('data/Base.csv')
X_train, X_test, y_train, y_test, features = loader.load_and_split(
    sample_size=10000,
    balance_classes=False
)

print(f"\nTest set: {len(X_test)} samples")
print(f"Fraud rate: {np.mean(y_test)*100:.2f}%")

# Test federated model
print("\n" + "="*70)
print("FEDERATED MODEL")
print("="*70)

fed_models = glob.glob('results/fraud_model_*.h5')
if fed_models:
    latest_fed = max(fed_models)
    print(f"Loading: {latest_fed}")
    
    fed_model = FraudDetectionModel(input_dim=31)
    fed_model.load(latest_fed)
    
    y_pred_fed = fed_model.predict(X_test).flatten()
    auc_fed = roc_auc_score(y_test, y_pred_fed)
    
    print(f"\nAUC-ROC: {auc_fed:.4f}")
    print(f"Predictions range: {y_pred_fed.min():.3f} to {y_pred_fed.max():.3f}")
    print(f"Mean prediction: {y_pred_fed.mean():.3f}")
    print(f"Std prediction: {y_pred_fed.std():.3f}")
    
    # Sample predictions
    print("\nSample predictions:")
    legit_idx = np.where(y_test == 0)[0][:5]
    fraud_idx = np.where(y_test == 1)[0][:5]
    
    print("Legitimate:")
    for i, idx in enumerate(legit_idx):
        print(f"  {i+1}. {y_pred_fed[idx]*100:.1f}%")
    
    if len(fraud_idx) > 0:
        print("Fraudulent:")
        for i, idx in enumerate(fraud_idx):
            print(f"  {i+1}. {y_pred_fed[idx]*100:.1f}%")
else:
    print("No federated model found!")

# Test simple model
print("\n" + "="*70)
print("SIMPLE MODEL (if exists)")
print("="*70)

simple_models = glob.glob('results/simple_model_*.h5')
if simple_models:
    latest_simple = max(simple_models)
    print(f"Loading: {latest_simple}")
    
    simple_model = FraudDetectionModel(input_dim=31)
    simple_model.load(latest_simple)
    
    y_pred_simple = simple_model.predict(X_test).flatten()
    auc_simple = roc_auc_score(y_test, y_pred_simple)
    
    print(f"\nAUC-ROC: {auc_simple:.4f}")
    print(f"Predictions range: {y_pred_simple.min():.3f} to {y_pred_simple.max():.3f}")
    print(f"Mean prediction: {y_pred_simple.mean():.3f}")
    print(f"Std prediction: {y_pred_simple.std():.3f}")
    
    # Sample predictions
    print("\nSample predictions:")
    print("Legitimate:")
    for i, idx in enumerate(legit_idx):
        print(f"  {i+1}. {y_pred_simple[idx]*100:.1f}%")
    
    if len(fraud_idx) > 0:
        print("Fraudulent:")
        for i, idx in enumerate(fraud_idx):
            print(f"  {i+1}. {y_pred_simple[idx]*100:.1f}%")
    
    # Comparison
    if fed_models:
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"Federated AUC: {auc_fed:.4f}")
        print(f"Simple AUC: {auc_simple:.4f}")
        print(f"Difference: {abs(auc_fed - auc_simple):.4f}")
        
        if auc_simple > auc_fed:
            print(f"\n✅ Simple model is better by {(auc_simple - auc_fed):.4f}")
        else:
            print(f"\n✅ Federated model is better by {(auc_fed - auc_simple):.4f}")
else:
    print("No simple model found!")
    print("Run: python test_simple_training.py")
