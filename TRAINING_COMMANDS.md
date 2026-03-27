# Training Commands for Federated Fraud Detection

## 🎯 Recommended: Multi-Variant Training

Uses all 6 dataset variants (Base + Variant I-V), each as a separate federated client:

```bash
python train_bank_account.py --multi-variant --rounds 30
```

| Setting | Value |
|---------|-------|
| Clients | 6 (one per variant) |
| Data per client | ~11K fraud + 11K legit (balanced) |
| Total data | 6M samples |
| Rounds | 30 |
| Local epochs | 10 |
| Architecture | [256, 128, 64, 32] + L2 reg |
| Learning rate | 0.0005 |
| Expected AUC | 0.87-0.92 |

---

## ⚡ Quick Test (Verify Setup)

```bash
python train_bank_account.py --multi-variant --sample-size 10000 --rounds 5
```

**Time**: ~10 minutes  
**Purpose**: Verify pipeline works before full training

---

## 📦 Single Dataset Training

```bash
# Full dataset
python train_bank_account.py --num-clients 5 --rounds 30

# Quick test
python train_bank_account.py --sample-size 200000 --num-clients 5 --rounds 15
```

**AUC**: ~0.85  
**Distribution**: Balanced (50:50 per client from Base.csv)

---

## 📊 After Training

```bash
# Evaluate model performance and threshold analysis
python check_model.py

# Full system demo (detection + explanation + privacy validation)
python test_end_to_end.py

# Interactive testing
python interactive_fraud_test.py
```

---

## 🔧 Command Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--multi-variant` | Use all 6 dataset variants | Off |
| `--sample-size N` | Limit samples per variant/dataset | Full dataset |
| `--num-clients N` | Number of clients (single-dataset mode) | 5 |
| `--rounds N` | Federated training rounds | 30 |
| `--local-epochs N` | Local epochs per round | 10 |

---

## ⚙️ Current Model Architecture

```python
# config.py
MODEL_CONFIG = {
    'hidden_layers': [256, 128, 64, 32],  # Deeper network
    'dropout_rate': 0.2,
    'learning_rate': 0.0005,
    'l2_reg': 0.001,
}

FL_CONFIG = {
    'num_rounds': 30,
    'local_epochs': 10,
    'batch_size': 64,
}
```

---

## ⚠️ Important Notes

1. **Differential Privacy is disabled** — was destroying model weights (see `config.py`)
2. **No SMOTE** — incompatible with federated learning
3. **Balanced distribution** — each client gets 50:50 fraud/legit
4. **Optimal threshold** saved to `results/optimal_threshold.txt` after training
5. **Class weights** are {0:1, 1:1} because balanced distribution handles imbalance
6. **L2 regularization** (0.001) prevents overfitting on balanced 50:50 data
7. **Variant III & V** extra columns (`x1`, `x2`) are auto-dropped
