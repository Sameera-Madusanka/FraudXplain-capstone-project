# Training Commands for Federated Fraud Detection

## 🎯 Recommended: Multi-Variant Training

Uses all 6 dataset variants (Base + Variant I-V), each as a separate federated client:

```bash
python train_bank_account.py --multi-variant --rounds 20
```

| Setting | Value |
|---------|-------|
| Clients | 6 (one per variant) |
| Data per client | ~11K fraud + 11K legit (balanced) |
| Total data | 6M samples |
| Rounds | 20 |
| Local epochs | 10 |
| Expected AUC | 0.85-0.92 |

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
python train_bank_account.py --num-clients 5 --rounds 20

# Quick test
python train_bank_account.py --sample-size 200000 --num-clients 5 --rounds 15
```

**AUC**: ~0.84  
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
| `--rounds N` | Federated training rounds | 10 |
| `--local-epochs N` | Local epochs per round | 5 |

---

## ⚠️ Important Notes

1. **Differential Privacy is disabled** — was destroying model weights (see `config.py`)
2. **No SMOTE** — incompatible with federated learning
3. **Balanced distribution** — each client gets 50:50 fraud/legit
4. **Optimal threshold** is saved to `results/optimal_threshold.txt` after training
5. **Class weights** are {0:1, 1:1} because balanced distribution handles imbalance
