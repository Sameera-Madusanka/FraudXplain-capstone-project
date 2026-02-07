"""
Simple test to verify all components work before full training
"""

import numpy as np
from data_loader_bank import BankAccountFraudLoader, FederatedBankAccountDistributor
from models.fraud_detector import FraudDetectionModel
from federated_learning.client import FederatedClient
from federated_learning.server import FederatedServer

print("Testing Bank Account Fraud Detection Setup...")
print("=" * 70)

# Step 1: Load data
print("\n1. Loading data...")
loader = BankAccountFraudLoader(dataset_path='data/Base.csv')
X_train, X_test, y_train, y_test, feature_names = loader.load_and_split(
    sample_size=1000,
    balance_classes=True
)
print(f"   ✓ Loaded {len(X_train)} training samples, {len(X_test)} test samples")

# Step 2: Distribute data
print("\n2. Distributing data to 2 clients...")
distributor = FederatedBankAccountDistributor(num_clients=2, distribution='iid')
client_data = distributor.distribute_data(X_train, y_train)
print(f"   ✓ Distributed to {len(client_data)} clients")

# Step 3: Create server and clients
print("\n3. Creating federated server and clients...")
input_dim = X_train.shape[1]
server = FederatedServer(input_dim=input_dim)

clients = []
for i, (X_client, y_client) in enumerate(client_data):
    client = FederatedClient(
        client_id=i+1,
        X_train=X_client,
        y_train=y_client,
        input_dim=input_dim
    )
    clients.append(client)
print(f"   ✓ Created server and {len(clients)} clients")

# Step 4: Run one training round
print("\n4. Running one training round...")
history = server.train_round(
    clients=clients,
    round_num=1,
    X_test=X_test,
    y_test=y_test,
    verbose=0
)
print(f"   ✓ Training round complete")
print(f"   AUC: {history['test_metrics']['auc']:.4f}")
print(f"   Precision: {history['test_metrics']['precision']:.4f}")
print(f"   Recall: {history['test_metrics']['recall']:.4f}")

print("\n" + "=" * 70)
print("✅ All components working correctly!")
print("=" * 70)
