# Class Weight Fix for SMOTE 10-90 Ratio

## Problem

After changing SMOTE to `sampling_strategy=0.1` (10-90 ratio), the class weights were still set to `{0: 1, 1: 100}` which was configured for the old 50-50 SMOTE ratio.

## Why This Matters

**Class weights tell the model how much to penalize errors**:

### Old Configuration (50-50 SMOTE)
```python
SMOTE: 50% fraud, 50% legitimate
Class weight: {0: 1, 1: 100}
Interpretation: Penalize fraud errors 100x more than legitimate errors
```

This made sense when fraud was artificially 50% of training data.

### New Configuration (10-90 SMOTE)
```python
SMOTE: 10% fraud, 90% legitimate  
Class weight: {0: 1, 1: 100} ← WRONG!
Problem: Over-penalizing fraud errors
Result: Model predicts everything as fraud to avoid penalty
```

## The Fix

**Adjusted class weight to match 10-90 ratio**:

```python
# config.py line 34
'class_weight': {0: 1, 1: 9},  # 90/10 = 9
```

**Why 9?**
- Legitimate: 90% of data
- Fraud: 10% of data
- Weight ratio: 90/10 = 9

This tells the model: "Fraud errors are 9x more important than legitimate errors" which matches the 10-90 distribution.

## Expected Result

With proper class weights:
- Model will learn balanced decision boundary
- False positive rate should drop from 100% to <10%
- Legitimate transactions: 5-20% fraud probability
- Fraudulent transactions: 80-95% fraud probability

## Next Step

Retrain the model:
```bash
python train_bank_account.py --sample-size 200000 --num-clients 5 --rounds 15
```

The combination of:
1. SMOTE 10-90 ratio (not 50-50)
2. Class weight 1:9 (not 1:100)

Should fix the catastrophic false positive problem.
