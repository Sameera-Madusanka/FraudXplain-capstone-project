"""
Quick script to check model predictions on real data
"""
import numpy as np
from models.fraud_detector import FraudDetectionModel
from data_loader_bank import BankAccountFraudLoader
import glob

# Load latest model
model_files = glob.glob('results/fraud_model_*.h5')
if not model_files:
    print("No model found!")
    exit(1)

latest_model = max(model_files)
print(f"Loading: {latest_model}\n")

model = FraudDetectionModel(input_dim=31)
model.load(latest_model)

# Load real test data
loader = BankAccountFraudLoader('data/Base.csv')
X_train, X_test, y_train, y_test, features = loader.load_and_split(
    sample_size=1000,
    balance_classes=False  # Keep real distribution
)

# Find legitimate and fraud samples
legit_indices = np.where(y_test == 0)[0]
fraud_indices = np.where(y_test == 1)[0]

print(f"Test set: {len(X_test)} samples")
print(f"Legitimate: {len(legit_indices)} ({len(legit_indices)/len(X_test)*100:.1f}%)")
print(f"Fraudulent: {len(fraud_indices)} ({len(fraud_indices)/len(X_test)*100:.1f}%)")
print()

# Test on 10 legitimate transactions
print("="*60)
print("Testing 10 LEGITIMATE transactions:")
print("="*60)
for i in range(min(10, len(legit_indices))):
    idx = legit_indices[i]
    pred = model.predict(X_test[idx].reshape(1, -1))[0][0]
    print(f"Transaction {i+1}: {pred*100:.1f}% fraud probability")

avg_legit = np.mean([model.predict(X_test[idx].reshape(1, -1))[0][0] 
                     for idx in legit_indices[:100]])
print(f"\nAverage on 100 legitimate: {avg_legit*100:.1f}%")

# Test on 10 fraudulent transactions
print("\n" + "="*60)
print("Testing 10 FRAUDULENT transactions:")
print("="*60)
for i in range(min(10, len(fraud_indices))):
    idx = fraud_indices[i]
    pred = model.predict(X_test[idx].reshape(1, -1))[0][0]
    print(f"Transaction {i+1}: {pred*100:.1f}% fraud probability")

if len(fraud_indices) > 0:
    avg_fraud = np.mean([model.predict(X_test[idx].reshape(1, -1))[0][0] 
                        for idx in fraud_indices[:min(100, len(fraud_indices))]])
    print(f"\nAverage on {min(100, len(fraud_indices))} fraudulent: {avg_fraud*100:.1f}%")

# Overall test set performance
print("\n" + "="*60)
print("Overall Test Set Performance:")
print("="*60)
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"Accuracy: {accuracy*100:.1f}%")
print(f"Precision: {precision*100:.1f}%")
print(f"Recall: {recall*100:.1f}%")
print(f"False Positive Rate: {fpr*100:.1f}%")
print(f"\nConfusion Matrix:")
print(f"  True Negatives: {tn}")
print(f"  False Positives: {fp} ← Legitimate flagged as fraud")
print(f"  False Negatives: {fn}")
print(f"  True Positives: {tp}")
