"""
Analyze Bank Account Fraud Dataset structure
"""

import pandas as pd

# Load a sample of the dataset
print("Loading Bank Account Fraud Dataset...")
df = pd.read_csv('data/Base.csv', nrows=10000)

print(f"\n{'='*70}")
print("BANK ACCOUNT FRAUD DATASET ANALYSIS")
print(f"{'='*70}\n")

print(f"Dataset Shape: {df.shape}")
print(f"  Rows: {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]}")

print(f"\n{'='*70}")
print("COLUMNS")
print(f"{'='*70}")
columns = df.columns.tolist()
for i, col in enumerate(columns, 1):
    print(f"{i:2d}. {col}")

print(f"\n{'='*70}")
print("TARGET VARIABLE")
print(f"{'='*70}")
print(f"\nTarget column: '{columns[0]}'")
print(f"\nClass distribution:")
print(df[columns[0]].value_counts())
fraud_rate = df[columns[0]].mean()
print(f"\nFraud rate: {fraud_rate:.4%}")
print(f"Imbalance ratio: {1/fraud_rate:.1f}:1")

print(f"\n{'='*70}")
print("DATA TYPES")
print(f"{'='*70}")
print(df.dtypes)

print(f"\n{'='*70}")
print("SAMPLE DATA (First 3 rows)")
print(f"{'='*70}")
print(df.head(3))

print(f"\n{'='*70}")
print("MISSING VALUES")
print(f"{'='*70}")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found!")

print(f"\n{'='*70}")
print("BASIC STATISTICS")
print(f"{'='*70}")
print(df.describe())

print("\n✅ Dataset analysis complete!")
