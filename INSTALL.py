"""
Installation Guide for Federated Learning Fraud Detection

Follow these steps to set up the environment:
"""

print("="*80)
print("INSTALLATION GUIDE")
print("="*80)

print("\n1. Install TensorFlow (choose based on your system):")
print("   For CPU: pip install tensorflow")
print("   For GPU: pip install tensorflow-gpu")
print("   Or latest: pip install tf-nightly")

print("\n2. Install core dependencies:")
print("   pip install numpy pandas scikit-learn matplotlib seaborn tqdm imbalanced-learn")

print("\n3. (Optional) Install DiCE for advanced counterfactual explanations:")
print("   pip install dice-ml")

print("\n4. Verify installation:")
print("   python -c \"import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')\"")

print("\n5. Run the demo:")
print("   python demo.py --rounds 5 --clients 3")

print("\n" + "="*80)
print("QUICK START (if TensorFlow is already installed):")
print("="*80)
print("\npip install numpy pandas scikit-learn matplotlib seaborn tqdm imbalanced-learn")
print("python demo.py --rounds 5 --clients 3")
print("\n" + "="*80)
