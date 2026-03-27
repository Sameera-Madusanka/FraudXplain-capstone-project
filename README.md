# Federated Learning Fraud Detection with Constrained Counterfactual Explanations

A privacy-preserving fraud detection system combining **Federated Learning** for distributed training with **Constrained Counterfactual Explanations** for actionable, privacy-guaranteed insights.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

This system addresses three critical challenges in financial fraud detection:

1. **Privacy in Training**: Federated Learning trains models across distributed institutions without sharing raw customer data
2. **Privacy in Explanations**: Constrained Counterfactuals ensure sensitive attributes are never revealed or suggested for change
3. **Actionability**: Provides clear, user-friendly recourse options that are actually achievable

### Novel Contribution

**First system to combine Federated Learning with Constrained Counterfactual Explanations**, providing:
- Privacy-preserving distributed training (FedAvg)
- Formal privacy guarantees in explanations
- Actionable recourse that protects sensitive information
- Multi-variant federated training across heterogeneous datasets

---

## 🔒 Privacy Guarantees

### Training Privacy (Federated Learning)
- ✅ Raw data never leaves client institutions
- ✅ Only model weights shared
- ✅ Secure aggregation (FedAvg)

### Explanation Privacy (Constrained Counterfactuals)
- ✅ **Protected Attributes** (NEVER changed):
  - Income, Age, Employment Status, Housing Status
  - Birth date metrics, Foreign request flag
- ✅ **Actionable Attributes** (Can suggest):
  - Credit score, Payment type, Device settings
  - Session behavior, Phone validation
- ✅ **Formal Validation**: Automated privacy compliance checking

---

## 📊 Dataset

**Bank Account Fraud Dataset Suite** (NeurIPS 2022) — Jesus et al.

| Variant | Samples | Fraud | Used As |
|---------|---------|-------|---------|
| Base | 1,000,000 | 11,029 | Client 1 |
| Variant I | 1,000,000 | 11,029 | Client 2 |
| Variant II | 1,000,000 | 11,029 | Client 3 |
| Variant III | 1,000,000 | 11,030 | Client 4 |
| Variant IV | 1,000,000 | 11,030 | Client 5 |
| Variant V | 1,000,000 | 11,030 | Client 6 |

- **Total**: 6M samples, ~66K fraud across all variants
- 31 common features (customer demographics, transaction patterns, device info)
- Highly imbalanced (~1% fraud rate)
- Variant III & V have 2 extra columns (`x1`, `x2`) — auto-dropped during loading

Place datasets in: `data/Base.csv`, `data/Variant I.csv`, etc.

---

## 🎯 Performance

### Current Results (Improved Architecture)

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.85** ✅ |
| **Prediction Separation** | **43%** ✅ |
| **Fraud Mean Prediction** | **70.5%** |
| **Legit Mean Prediction** | **27.5%** |
| **Optimal Threshold** | **0.84** (F1-optimized) |

### Imbalance Handling

**Balanced Distribution** — NO SMOTE, NO synthetic data:
- Each federated client receives 50:50 fraud/legitimate samples
- Real data only — no synthetic artifacts
- Class weights set to {0:1, 1:1}

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.11+, TensorFlow 2.16+, NumPy, Pandas, Scikit-learn, Matplotlib, Imbalanced-learn

### Training

```bash
# Recommended: Multi-variant (6 datasets, 6 clients, best performance)
python train_bank_account.py --multi-variant --rounds 30

# Single dataset (Base.csv only)
python train_bank_account.py --num-clients 5 --rounds 30

# Quick test (verify setup)
python train_bank_account.py --multi-variant --sample-size 10000 --rounds 5
```

### Evaluation & Testing

```bash
python check_model.py              # Evaluate model metrics
python test_end_to_end.py          # Full system demo
python interactive_fraud_test.py   # Interactive testing
```

---

## 🏗️ Architecture

### Model Architecture

```
Input(31) → Dense(256, ReLU, L2) → Dropout(20%)
          → Dense(128, ReLU, L2) → Dropout(20%)
          → Dense(64, ReLU, L2)  → Dropout(20%)
          → Dense(32, ReLU, L2)  → Dropout(20%)
          → Dense(1, Sigmoid)    → P(fraud)
```

### System Components

| Component | Description |
|-----------|-------------|
| **Data Layer** | Multi-variant loader, balanced distribution, shared scaler |
| **Model Layer** | DNN with L2 regularization, He initialization |
| **FL Layer** | FedAvg server, 6 clients (one per variant) |
| **Explainability** | Constrained CFs, Privacy validator, Actionable recourse |

### Training Configuration

| Setting | Value |
|---------|-------|
| Algorithm | Federated Averaging (FedAvg) |
| Clients | 6 (multi-variant) or 5 (single dataset) |
| Rounds | 30 |
| Local Epochs | 10 |
| Batch Size | 64 |
| Learning Rate | 0.0005 (Adam) |
| L2 Regularization | 0.001 |
| Dropout | 0.2 |
| Differential Privacy | Disabled |

---

## 📁 Project Structure

```
.
├── train_bank_account.py              # Main training script (single + multi-variant)
├── check_model.py                     # Post-training evaluation
├── test_end_to_end.py                 # Full system demonstration
├── interactive_fraud_test.py          # Interactive testing interface
├── config.py                          # All configuration settings
├── data_loader_bank.py                # Data loading + multi-variant support
├── requirements.txt                   # Dependencies
│
├── models/
│   └── fraud_detector.py              # DNN with L2 regularization
│
├── federated_learning/
│   ├── server.py                      # FedAvg server + evaluation
│   ├── client.py                      # Local training client
│   └── aggregation.py                 # Aggregation strategies
│
├── explainability/
│   ├── constrained_counterfactuals.py # Privacy-preserving CF generation
│   ├── privacy_validator.py           # Privacy + feasibility validation
│   ├── actionable_recourse.py         # User-friendly recourse
│   └── visualization.py              # Training plots, ROC, confusion matrix
│
├── utils/
│   └── metrics.py                     # Evaluation metrics
│
├── data/
│   ├── Base.csv                       # Bank Account Fraud Dataset
│   ├── Variant I.csv - Variant V.csv  # Dataset variants
│   └── protected_attributes.json      # Privacy configuration
│
├── results/                           # Training outputs
│   ├── fraud_model_*.h5               # Saved models
│   ├── optimal_threshold.txt          # F1-optimized threshold
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
└── docs/
    ├── SYSTEM_STATUS.md               # Current system status
    └── CONSTRAINED_CF_GUIDE.md        # CF explanation guide
```

---

## 💡 Example Output

### Fraud Detection

```
🔍 Fraud Detection Result:
   Prediction: 🚨 FRAUD
   Fraud Probability: 89.7% (threshold: 84.0%)
   Confidence: High
```

### Privacy-Guaranteed Explanation

```
🔒 PRIVACY-GUARANTEED FRAUD EXPLANATION
======================================================================

🚨 Transaction Flagged as FRAUDULENT
   Fraud Probability: 89.7%

🔒 PRIVACY GUARANTEE:
   The following sensitive attributes are PROTECTED:
   • income: -1.58 (unchanged)
   • customer_age: -1.09 (unchanged)
   • employment_status: 1.83 (unchanged)

📋 ACTIONABLE RECOURSE:
   To clear this alert, you can:

   Option 1:
   ✓ Improve credit score from -0.88 to 1.50
   ✓ Use verified device
   → Estimated fraud probability: 25.3%

✅ PRIVACY COMPLIANCE:
   • No sensitive information revealed
   • Only actionable attributes suggested
   • All suggestions verified for feasibility
```

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
FL_CONFIG = {
    'num_rounds': 30,
    'num_clients': 5,
    'local_epochs': 10,
    'batch_size': 64
}

MODEL_CONFIG = {
    'hidden_layers': [256, 128, 64, 32],
    'dropout_rate': 0.2,
    'learning_rate': 0.0005,
    'l2_reg': 0.001
}
```

Edit `data/protected_attributes.json` for privacy settings.

---

## 🎓 Research Context

### Key References

1. **McMahan et al. (2017)** — "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
2. **Jesus et al. (2022)** — "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets" (NeurIPS, Dataset)
3. **Abadi et al. (2016)** — "Deep Learning with Differential Privacy"
4. **Li et al. (2022)** — "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization"
5. **Lipton et al. (2014)** — "Optimal Thresholding of Classifiers to Maximize F1 Measure"

### Key Findings

- SMOTE synthetic data is **incompatible** with federated learning aggregation
- BatchNormalization **breaks** FedAvg due to local statistics mismatch
- Differential Privacy noise must be carefully tuned or it **destroys** model weights
- Balanced distribution (50:50 per client) is critical for imbalanced FL
- Multi-variant training provides natural data heterogeneity for realistic FL

---

## 📝 License

This project is for research and educational purposes.

---

**Built with**: TensorFlow, Scikit-learn, Pandas, NumPy, Matplotlib, Imbalanced-learn

**Key Innovation**: Privacy-preserving fraud detection with formal privacy guarantees in both training and explanations, using multi-variant federated learning across 6 heterogeneous datasets.
