"""
Quick script to check model predictions on real data
Automatically finds the optimal classification threshold
"""
import numpy as np
from models.fraud_detector import FraudDetectionModel
from data_loader_bank import BankAccountFraudLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
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

# Load test data
loader = BankAccountFraudLoader('data/Base.csv')
X_train, X_test, y_train, y_test, features = loader.load_and_split(
    sample_size=200000,
    balance_classes=False
)

print(f"Test set: {len(X_test)} samples")
legit_indices = np.where(y_test == 0)[0]
fraud_indices = np.where(y_test == 1)[0]
print(f"Legitimate: {len(legit_indices)} ({len(legit_indices)/len(X_test)*100:.1f}%)")
print(f"Fraudulent: {len(fraud_indices)} ({len(fraud_indices)/len(X_test)*100:.1f}%)")

# Get all predictions
y_pred_proba = model.predict(X_test).flatten()

# Show prediction distribution
legit_probs = y_pred_proba[y_test == 0]
fraud_probs = y_pred_proba[y_test == 1]

print(f"\n{'='*60}")
print("Prediction Distribution:")
print(f"{'='*60}")
print(f"Legitimate: mean={legit_probs.mean()*100:.1f}%, "
      f"min={legit_probs.min()*100:.1f}%, max={legit_probs.max()*100:.1f}%")
print(f"Fraudulent: mean={fraud_probs.mean()*100:.1f}%, "
      f"min={fraud_probs.min()*100:.1f}%, max={fraud_probs.max()*100:.1f}%")

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC: {auc:.4f}")

# Find optimal threshold using F1-score (balances precision AND recall)
print(f"\n{'='*60}")
print("Threshold Analysis:")
print(f"{'='*60}")
print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'FPR':>10} | {'TP':>6} | {'FP':>6}")
print("-" * 75)

best_f1 = 0
best_threshold = 0.5
thresholds_to_test = np.percentile(y_pred_proba, np.arange(5, 96, 5))
thresholds_to_test = np.unique(np.round(thresholds_to_test, 4))

for t in thresholds_to_test:
    y_pred = (y_pred_proba > t).astype(int)
    if y_pred.sum() == 0:
        continue
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    
    marker = " <-- BEST" if f1 > best_f1 else ""
    print(f"{t:>10.4f} | {p:>10.4f} | {r:>10.4f} | {f1:>10.4f} | {fpr:>10.4f} | {tp:>6} | {fp:>6}{marker}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\n{'='*60}")
print(f"OPTIMAL THRESHOLD: {best_threshold:.4f} ({best_threshold*100:.1f}%)")
print(f"(Maximizes F1-score = {best_f1:.4f})")
print(f"{'='*60}")

# Show results with optimal threshold
y_pred = (y_pred_proba > best_threshold).astype(int)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\nResults with Optimal threshold ({best_threshold:.4f}):")
print(f"  Accuracy:  {accuracy*100:.1f}%")
print(f"  Precision: {precision*100:.1f}%")
print(f"  Recall:    {recall*100:.1f}%")
print(f"  F1-Score:  {best_f1:.4f}")
print(f"  FPR:       {fpr*100:.1f}%")
print(f"\n  Confusion Matrix:")
print(f"    True Negatives:  {tn}")
print(f"    False Positives: {fp}")
print(f"    False Negatives: {fn}")
print(f"    True Positives:  {tp}")

# Save optimal threshold
with open('results/optimal_threshold.txt', 'w') as f:
    f.write(f"{best_threshold}")
print(f"\nOptimal threshold saved to results/optimal_threshold.txt")
