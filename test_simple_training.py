"""
Simple non-federated training to test if model can learn at all
"""
import numpy as np
from models.fraud_detector import FraudDetectionModel
from data_loader_bank import BankAccountFraudLoader
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

print("="*60)
print("SIMPLE NON-FEDERATED TRAINING TEST")
print("="*60)

# Load data WITHOUT SMOTE
print("\n1. Loading data WITHOUT SMOTE...")
loader = BankAccountFraudLoader('data/Base.csv')
X_train, X_test, y_train, y_test, features = loader.load_and_split(
    sample_size=50000,
    balance_classes=False  # NO SMOTE!
)

print(f"\nTraining set:")
print(f"  Samples: {len(X_train)}")
print(f"  Fraud rate: {np.mean(y_train)*100:.2f}%")
print(f"  Frauds: {np.sum(y_train)}")

print(f"\nTest set:")
print(f"  Samples: {len(X_test)}")
print(f"  Fraud rate: {np.mean(y_test)*100:.2f}%")
print(f"  Frauds: {np.sum(y_test)}")

# Train simple model with class weights
print("\n2. Training model with class weights (NO SMOTE)...")
model = FraudDetectionModel(input_dim=31)

# Calculate class weight based on real distribution
fraud_rate = np.mean(y_train)
class_weight = {
    0: 1.0,
    1: (1 - fraud_rate) / fraud_rate  # Inverse of fraud rate
}

print(f"  Class weights: {class_weight}")

history = model.train(
    X_train, y_train,
    epochs=20,
    batch_size=256,
    class_weight=class_weight,
    verbose=1
)

# Evaluate
print("\n3. Evaluating model...")
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC: {auc:.4f}")

if auc > 0.5:
    print("✅ Model is learning! (AUC > 0.5)")
else:
    print("❌ Model is NOT learning (AUC <= 0.5)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

print("\nConfusion Matrix:")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"  True Negatives: {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives: {tp}")

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f"\nFalse Positive Rate: {fpr*100:.1f}%")

# Test on sample transactions
print("\n4. Testing on sample transactions...")
legit_indices = np.where(y_test == 0)[0][:10]
fraud_indices = np.where(y_test == 1)[0][:10]

print("\nLegitimate transactions:")
for i, idx in enumerate(legit_indices):
    prob = y_pred_proba[idx]
    print(f"  {i+1}. {prob*100:.1f}% fraud")

if len(fraud_indices) > 0:
    print("\nFraudulent transactions:")
    for i, idx in enumerate(fraud_indices):
        prob = y_pred_proba[idx]
        print(f"  {i+1}. {prob*100:.1f}% fraud")

# Save model if it's good
if auc > 0.7:
    print("\n✅ Model is good! Saving...")
    model.save('results/simple_model_no_smote.h5')
else:
    print("\n❌ Model performance is poor. Not saving.")
