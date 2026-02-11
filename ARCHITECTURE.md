# Architectural Design: Federated Learning Fraud Detection with Constrained Counterfactual Explanations

## System Overview

A privacy-preserving fraud detection system that combines **Federated Learning** for distributed training with **Constrained Counterfactual Explanations** for actionable, privacy-guaranteed insights.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                        │
│  • Interactive Testing (interactive_fraud_test.py)              │
│  • Training Scripts (train_bank_account.py)                     │
│  • Demo Applications (demo_constrained_cf.py)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  EXPLAINABILITY LAYER                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ Constrained CF   │  │ Privacy          │  │ Actionable    │ │
│  │ Generator        │  │ Validator        │  │ Recourse      │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│              FEDERATED LEARNING LAYER                           │
│  ┌──────────────────┐           ┌─────────────────────────────┐ │
│  │ Federated Server │◄─────────►│ Federated Clients (1-N)     │ │
│  │ • Global Model   │           │ • Local Models              │ │
│  │ • Aggregation    │           │ • Local Training            │ │
│  │ • Evaluation     │           │ • Private Data              │ │
│  └──────────────────┘           └─────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    MODEL LAYER                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Fraud Detection Model (Deep Neural Network)             │   │
│  │ • Input: 31 features                                     │   │
│  │ • Output: Fraud probability [0-1]                        │   │
│  │ • Architecture: Dense layers with dropout                │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     DATA LAYER                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ Bank Account     │  │ Data             │  │ Protected     │ │
│  │ Fraud Dataset    │  │ Distributor      │  │ Attributes    │ │
│  │ (~1M samples)    │  │ (IID/Non-IID)    │  │ Config        │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Data Layer

#### 1.1 Bank Account Fraud Dataset Loader
**File**: `data_loader_bank.py`

**Responsibilities**:
- Load Bank Account Fraud Dataset (NeurIPS 2022)
- Handle 31 features (numerical + categorical)
- Categorical encoding (Label Encoding)
- Class balancing with SMOTE
- Train/test splitting

**Key Classes**:
```python
class BankAccountFraudLoader:
    - load_and_split()
    - _encode_categorical()
    - _balance_classes()
```

#### 1.2 Federated Data Distributor
**File**: `data_loader_bank.py`

**Responsibilities**:
- Distribute data across federated clients
- Support IID and Non-IID distributions
- Maintain data privacy (no data sharing)

**Key Classes**:
```python
class FederatedBankAccountDistributor:
    - distribute_data()
    - _distribute_iid()
    - _distribute_non_iid()
```

#### 1.3 Protected Attributes Configuration
**File**: `data/protected_attributes.json`

**Defines**:
- **Protected**: Sensitive attributes (income, age, employment, housing)
- **Actionable**: User-controllable attributes (credit score, payment type)
- **Immutable**: Historical data (previous addresses, device history)

---

### 2. Model Layer

#### 2.1 Fraud Detection Model
**File**: `models/fraud_detector.py`

**Architecture**:
```
Input (31 features)
    ↓
Dense(128, ReLU) + Dropout(0.3)
    ↓
Dense(64, ReLU) + Dropout(0.3)
    ↓
Dense(32, ReLU) + Dropout(0.2)
    ↓
Dense(1, Sigmoid)
    ↓
Output (Fraud Probability)
```

**Key Features**:
- Binary classification (fraud/legitimate)
- Dropout for regularization
- Adam optimizer
- Binary cross-entropy loss
- Metrics: Accuracy, Precision, Recall, AUC

**Key Classes**:
```python
class FraudDetectionModel:
    - __init__(input_dim)
    - predict(X)
    - evaluate(X, y)
    - save(path)
    - load(path)
```

---

### 3. Federated Learning Layer

#### 3.1 Federated Server
**File**: `federated_learning/server.py`

**Responsibilities**:
- Maintain global model
- Orchestrate training rounds
- Aggregate client updates (FedAvg)
- Evaluate global model
- Track training history

**Key Methods**:
```python
class FederatedServer:
    - train_round(clients, round_num, X_test, y_test)
    - aggregate_weights(client_weights, client_samples)
    - get_global_weights()
    - set_global_weights(weights)
```

**Aggregation Algorithm**: FedAvg (Federated Averaging)
```
global_weights = Σ(client_weights_i × num_samples_i) / Σ(num_samples_i)
```

#### 3.2 Federated Client
**File**: `federated_learning/client.py`

**Responsibilities**:
- Maintain local model
- Train on private local data
- Send only model updates (not data)
- Track local statistics

**Key Methods**:
```python
class FederatedClient:
    - train(epochs, batch_size)
    - get_weights()
    - set_weights(weights)
```

**Privacy Guarantee**: Raw data never leaves client

---

### 4. Explainability Layer

#### 4.1 Constrained Counterfactual Generator
**File**: `explainability/constrained_counterfactuals.py`

**Core Innovation**: Generates counterfactual explanations that:
- Never modify protected attributes
- Only suggest changes to actionable features
- Minimize total changes for feasibility

**Algorithm**:
```python
1. Load protected attributes configuration
2. Identify actionable feature indices
3. Optimize only actionable features:
   - Minimize: distance(original, counterfactual)
   - Subject to: prediction(counterfactual) = target_class
   - Constraint: protected_features unchanged
4. Validate constraints
5. Generate explanation
```

**Key Methods**:
```python
class ConstrainedCounterfactualGenerator:
    - generate_constrained_counterfactual(instance, target_class)
    - generate_privacy_guaranteed_explanation(instance, counterfactuals)
    - _validate_constraints(original, counterfactual)
```

#### 4.2 Privacy Validator
**File**: `explainability/privacy_validator.py`

**Responsibilities**:
- Validate no protected attributes changed
- Check feasibility of suggestions
- Generate compliance reports

**Validation Rules**:
1. Protected attributes must be identical
2. Changes must be within feasible ranges
3. Suggested values must be realistic

**Key Classes**:
```python
class PrivacyValidator:
    - validate_counterfactual(original, counterfactual)
    - check_explanation_privacy(explanation)

class FeasibilityChecker:
    - check_feasibility(original, counterfactual)
```

#### 4.3 Actionable Recourse Generator
**File**: `explainability/actionable_recourse.py`

**Responsibilities**:
- Convert counterfactuals to user-friendly language
- Provide multiple recourse options
- Indicate difficulty levels
- Show expected outcomes

**Key Methods**:
```python
class ActionableRecourseGenerator:
    - generate_recourse(original_instance, counterfactuals, original_pred)
    - _format_recourse_option(changes, difficulty)
```

---

## Data Flow

### Training Flow

```
1. Data Loading
   Bank Account Dataset → Data Loader → Preprocessed Data
                                              ↓
2. Federated Distribution
   Preprocessed Data → Distributor → Client 1, Client 2, ..., Client N
                                              ↓
3. Federated Training (Per Round)
   Server: Broadcast global weights
       ↓
   Clients: Train locally on private data
       ↓
   Clients: Send weight updates to server
       ↓
   Server: Aggregate updates (FedAvg)
       ↓
   Server: Update global model
       ↓
   Server: Evaluate on test set
       ↓
   Repeat for N rounds
                                              ↓
4. Model Persistence
   Trained Model → Save to disk (results/fraud_model_*.h5)
```

### Inference Flow (with Explanations)

```
1. Transaction Input
   User/System → Transaction (31 features)
                        ↓
2. Fraud Detection
   Transaction → Model → Fraud Probability
                        ↓
3. Decision
   If Fraud Probability > 0.5:
       ↓
4. Constrained CF Generation
   Transaction → CF Generator → Constrained Counterfactuals
                                       ↓
5. Privacy Validation
   Counterfactuals → Privacy Validator → Validation Report
                                       ↓
6. Actionable Recourse
   Counterfactuals → Recourse Generator → User-Friendly Guidance
                                       ↓
7. Output
   • Fraud Prediction
   • Privacy-Guaranteed Explanation
   • Actionable Recourse Options
```

---

## Key Design Patterns

### 1. Strategy Pattern
**Used in**: Data Distribution
- `IIDDistribution` strategy
- `NonIIDDistribution` strategy

### 2. Template Method Pattern
**Used in**: Federated Training
- Base training loop in server
- Specific aggregation methods

### 3. Builder Pattern
**Used in**: Model Construction
- Layer-by-layer model building
- Configuration-based construction

### 4. Observer Pattern
**Used in**: Training Monitoring
- History tracking
- Metric logging

---

## Privacy Architecture

### Privacy Guarantees

#### 1. Federated Learning Privacy
```
Data Privacy:
  ✓ Raw data never leaves client devices
  ✓ Only model weights shared
  ✓ Differential privacy (optional, configurable)

Communication:
  ✓ Encrypted weight updates
  ✓ Secure aggregation
```

#### 2. Explanation Privacy
```
Protected Attributes (Never Changed):
  ✓ income
  ✓ customer_age
  ✓ employment_status
  ✓ housing_status
  ✓ date_of_birth_distinct_emails_4w
  ✓ foreign_request

Actionable Attributes (Can Suggest):
  ✓ credit_risk_score
  ✓ proposed_credit_limit
  ✓ payment_type
  ✓ device_os
  ✓ session_length_in_minutes
  ✓ has_other_cards
  ✓ email_is_free
  ✓ phone_home_valid
  ✓ phone_mobile_valid
  ✓ keep_alive_session

Immutable Attributes (Historical):
  ✓ prev_address_months_count
  ✓ bank_months_count
  ✓ velocity metrics
  ✓ device_fraud_count
  (15 total)
```

#### 3. Formal Validation
```python
For each counterfactual CF:
  Assert: protected_attributes(original) == protected_attributes(CF)
  Assert: changes only in actionable_attributes
  Assert: all changes are feasible
```

---

## Configuration Management

### System Configuration
**File**: `config.py`

```python
FL_CONFIG = {
    'num_rounds': 20,
    'num_clients': 5,
    'client_fraction': 1.0,
    'local_epochs': 5,
    'batch_size': 32,
    'min_clients': 2
}

MODEL_CONFIG = {
    'hidden_layers': [128, 64, 32],
    'dropout_rates': [0.3, 0.3, 0.2],
    'learning_rate': 0.001,
    'activation': 'relu'
}

PRIVACY_CONFIG = {
    'use_differential_privacy': False,
    'noise_multiplier': 0.1,
    'l2_norm_clip': 1.0
}
```

### Protected Attributes Configuration
**File**: `data/protected_attributes.json`

```json
{
  "protected_attributes": {
    "attributes": ["income", "customer_age", ...],
    "rationale": "Sensitive personal information"
  },
  "actionable_attributes": {
    "attributes": ["credit_risk_score", ...],
    "rationale": "User can control these"
  },
  "immutable_attributes": {
    "attributes": ["prev_address_months_count", ...],
    "rationale": "Historical data, cannot change"
  }
}
```

---

## Scalability & Performance

### Horizontal Scaling
- **Clients**: Can scale to 100+ clients
- **Data**: Handles 1M+ samples
- **Parallel Training**: Clients train independently

### Optimization Strategies
1. **Batch Processing**: Mini-batch gradient descent
2. **Early Stopping**: Prevent overfitting
3. **Model Compression**: Lightweight architecture
4. **Efficient Aggregation**: Weighted averaging

### Performance Metrics
- **Training Time**: ~30-60 min (full dataset, 20 rounds)
- **Inference Time**: <100ms per transaction
- **CF Generation**: ~1-2 seconds per explanation

---

## Technology Stack

### Core Framework
- **Deep Learning**: TensorFlow 2.16.1 / Keras
- **Data Processing**: Pandas, NumPy
- **ML Utilities**: Scikit-learn
- **Imbalanced Learning**: imbalanced-learn (SMOTE)

### Visualization
- **Plotting**: Matplotlib, Seaborn
- **Metrics**: Confusion Matrix, ROC Curve, Training History

### Development
- **Language**: Python 3.11
- **Environment**: Virtual Environment / Conda
- **Version Control**: Git

---

## Security Considerations

### Data Security
- ✓ No raw data transmission
- ✓ Local data storage only
- ✓ Encrypted model updates (optional)

### Model Security
- ✓ Model poisoning detection (optional)
- ✓ Byzantine-robust aggregation (optional)
- ✓ Secure model storage

### Explanation Security
- ✓ Formal privacy validation
- ✓ No sensitive attribute leakage
- ✓ Feasibility constraints

---

## Deployment Architecture

### Development Environment
```
Local Machine
  ├── Data: data/Base.csv
  ├── Models: results/fraud_model_*.h5
  ├── Configs: data/protected_attributes.json
  └── Logs: logs/
```

### Production Environment (Proposed)
```
┌─────────────────────────────────────────┐
│         Load Balancer                   │
└────────────┬────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼────┐
│ Server │      │ Server  │
│ Node 1 │      │ Node 2  │
└───┬────┘      └────┬────┘
    │                │
    └────────┬───────┘
             │
    ┌────────▼────────┐
    │  Model Storage  │
    │  (Shared)       │
    └─────────────────┘

Federated Clients (Financial Institutions)
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Bank 1  │  │ Bank 2  │  │ Bank 3  │
│ (Local  │  │ (Local  │  │ (Local  │
│  Data)  │  │  Data)  │  │  Data)  │
└─────────┘  └─────────┘  └─────────┘
```

---

## Testing Architecture

### Unit Tests
- Model layer tests
- Data loader tests
- CF generator tests
- Privacy validator tests

### Integration Tests
- End-to-end training
- Federated aggregation
- Explanation generation

### System Tests
- `interactive_fraud_test.py`: Manual testing
- `test_end_to_end.py`: Automated testing
- `demo_constrained_cf.py`: Demo scenarios

---

## Future Enhancements

### Planned Features
1. **Differential Privacy**: Add noise to gradients
2. **Secure Aggregation**: Homomorphic encryption
3. **Model Compression**: Federated distillation
4. **Real-time Inference**: API deployment
5. **Multi-language Support**: Explanations in multiple languages

### Research Extensions
1. **Personalized Federated Learning**: Client-specific models
2. **Adversarial Robustness**: Defense against attacks
3. **Causal Counterfactuals**: Causal reasoning in explanations
4. **Multi-objective Optimization**: Balance multiple constraints

---

## Summary

This architecture demonstrates a **novel integration** of:
- **Federated Learning**: Privacy-preserving distributed training
- **Constrained Counterfactuals**: Privacy-guaranteed explanations
- **Formal Validation**: Automated privacy compliance

**Key Innovation**: First system to combine FL with constrained CFs for fraud detection, providing both privacy in training and privacy in explanations.
