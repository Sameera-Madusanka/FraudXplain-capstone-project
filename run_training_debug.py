"""
Wrapper to run training with full error output
"""

import traceback
import sys

try:
    import train_bank_account
    
    # Simulate command line args
    class Args:
        sample_size = 10000
        num_clients = 3
        rounds = 5
        local_epochs = 3
    
    args = Args()
    train_bank_account.main(args)
    
except Exception as e:
    print("\n" + "=" * 80)
    print("ERROR OCCURRED:")
    print("=" * 80)
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    print("-" * 80)
    traceback.print_exc()
    print("=" * 80)
    sys.exit(1)
