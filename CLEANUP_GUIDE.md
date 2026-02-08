# Files Safe to Remove - Credit Card Dataset Cleanup

## Files to Remove (Old Credit Card Dataset)

### Safe to Remove ✅
These files are NOT used by your working Bank Account system:

1. **main.py** - Old training script for Credit Card dataset
   - Uses: `data_loader.py` (Credit Card)
   - Replaced by: `train_bank_account.py`

2. **demo.py** - Old demo for Credit Card dataset
   - Uses: `data_loader.py` (Credit Card)
   - Replaced by: `demo_constrained_cf.py`

3. **data_loader.py** - Credit Card dataset loader
   - Only used by: `main.py` and `demo.py`
   - Replaced by: `data_loader_bank.py`

4. **analyze_dataset.py** - Dataset analysis script (optional)
   - Can be removed if not needed

5. **test_setup.py** - Temporary test script
   - Can be removed (was for debugging)

6. **run_training_debug.py** - Debug wrapper
   - Can be removed (was for debugging)

7. **error_log.txt** - Error log from debugging
   - Can be removed

8. **TENSORFLOW_FIX_GUIDE.md** - Installation troubleshooting guide
   - Can be removed (issue is fixed)

### Keep These ✅
Your working Bank Account system:

1. **train_bank_account.py** - Main training script (Bank Account)
2. **demo_constrained_cf.py** - Demo for constrained CFs
3. **data_loader_bank.py** - Bank Account dataset loader
4. **explainability/** - All constrained CF modules
5. **federated_learning/** - FL framework
6. **models/** - Model definitions
7. **utils/** - Metrics and utilities
8. **config.py** - Configuration
9. **data/** - Dataset and configs
10. **results/** - Training results

## Recommendation

Remove the old files to clean up the project. Your working system will not be affected.
