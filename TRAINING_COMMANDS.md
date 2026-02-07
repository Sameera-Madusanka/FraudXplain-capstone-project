# Training Commands for Federated Fraud Detection

## Quick Test (Small Sample)
Run this first to verify everything works with a small dataset:

```bash
# Test with 10,000 samples, 3 clients, 5 rounds
python train_bank_account.py --sample-size 10000 --num-clients 3 --rounds 5 --local-epochs 3
```

**Expected time**: ~5-10 minutes

---

## Standard Training (Recommended)
For good results with reasonable training time:

```bash
# 50,000 samples, 5 clients, 10 rounds
python train_bank_account.py --sample-size 50000 --num-clients 5 --rounds 10 --local-epochs 5
```

**Expected time**: ~20-30 minutes

---

## Full Dataset Training
For best results (uses entire ~1M sample dataset):

```bash
# Full dataset, 5 clients, 20 rounds
python train_bank_account.py --num-clients 5 --rounds 20 --local-epochs 5
```

**Expected time**: ~2-3 hours
**Note**: Requires significant RAM (~8GB+)

---

## Custom Training
Customize the parameters as needed:

```bash
python train_bank_account.py \
    --sample-size 100000 \
    --num-clients 10 \
    --rounds 15 \
    --local-epochs 3
```

### Parameters:
- `--sample-size`: Number of samples to use (omit for full dataset)
- `--num-clients`: Number of federated clients (banks)
- `--rounds`: Number of federated training rounds
- `--local-epochs`: Epochs each client trains per round

---

## What the Script Does

1. **Loads Data**: Bank Account Fraud Dataset
2. **Distributes**: Splits data across federated clients
3. **Trains**: Federated learning with FedAvg aggregation
4. **Evaluates**: Computes metrics (AUC, Precision, Recall, F1)
5. **Explains**: Generates constrained counterfactual explanations
6. **Validates**: Checks privacy guarantees
7. **Saves**: Model, plots, and explanations to `results/`

---

## Output Files

After training, check the `results/` folder:
- `training_history.png` - Training progress
- `confusion_matrix.png` - Classification results
- `roc_curve.png` - ROC curve
- `example_explanation.txt` - Sample counterfactual explanation
- `fraud_model_YYYYMMDD_HHMMSS.h5` - Trained model

---

## Troubleshooting

### Out of Memory
Reduce `--sample-size`:
```bash
python train_bank_account.py --sample-size 20000
```

### Too Slow
Reduce `--rounds` and `--local-epochs`:
```bash
python train_bank_account.py --rounds 5 --local-epochs 2
```

### TensorFlow Warnings
Set environment variable:
```bash
$env:TF_ENABLE_ONEDNN_OPTS=0
python train_bank_account.py --sample-size 10000
```

---

## Recommended Workflow

1. **Quick Test** (verify setup):
   ```bash
   python train_bank_account.py --sample-size 10000 --rounds 3
   ```

2. **Standard Training** (for demo/presentation):
   ```bash
   python train_bank_account.py --sample-size 50000 --rounds 10
   ```

3. **Review Results**:
   - Check `results/` folder
   - Review `example_explanation.txt`
   - Examine plots

4. **Full Training** (for final results):
   ```bash
   python train_bank_account.py --rounds 20
   ```
