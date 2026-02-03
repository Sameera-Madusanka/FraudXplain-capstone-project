# Federated Learning Fraud Detection with Counterfactual Explanations

This project implements a fraud detection system combining **Federated Learning** and **Counterfactual Explanations** based on the paper:

> **"Transparency and Privacy: The Role of Explainable AI and Federated Learning in Financial Fraud Detection"**  
> DOI: 10.1109/ACCESS.2024.3394528

## 🎯 Overview

The system addresses two critical challenges in financial fraud detection:
1. **Privacy**: Using Federated Learning to train models without sharing sensitive customer data
2. **Transparency**: Using Counterfactual Explanations to make fraud predictions interpretable

## 🏗️ Architecture

### Components

- **Federated Learning Framework**
  - Client-side local training on private data
  - Server-side FedAvg aggregation
  - Differential privacy mechanisms

- **Fraud Detection Model**
  - Deep neural network for binary classification
  - Handles imbalanced datasets
  - Optimized for fraud detection metrics

- **Explainability Module**
  - Counterfactual explanation generation
  - Shows minimal changes to flip predictions
  - Enhances trust and transparency

## 📁 Project Structure

```
.
├── config.py                          # Configuration and hyperparameters
├── data_loader.py                     # Data loading and distribution
├── main.py                            # Main training pipeline
├── demo.py                            # Interactive demonstration
├── requirements.txt                   # Python dependencies
│
├── federated_learning/
│   ├── client.py                      # Federated client implementation
│   ├── server.py                      # Federated server implementation
│   └── aggregation.py                 # Aggregation strategies (FedAvg)
│
├── models/
│   └── fraud_detector.py              # Neural network model
│
├── explainability/
│   ├── counterfactual_generator.py    # Counterfactual explanations
│   └── visualization.py               # Visualization utilities
│
├── utils/
│   └── metrics.py                     # Evaluation metrics
│
├── data/                              # Dataset directory
├── results/                           # Training results and plots
└── logs/                              # Training logs
```

## 🚀 Getting Started

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Demo

Run a quick demonstration with 3 clients and 10 training rounds:

```bash
python demo.py --rounds 10 --clients 3
```

### Full Training

Run the complete training pipeline:

```bash
python main.py
```

This will:
1. Load and distribute data among federated clients
2. Train the global model for 20 rounds
3. Evaluate performance on test set
4. Generate counterfactual explanations
5. Save results and visualizations

## 📊 Dataset

The system can work with:
- **Real data**: Credit card fraud dataset (place in `data/creditcard.csv`)
- **Synthetic data**: Automatically generated if no dataset is found

Expected format:
- Features: Transaction attributes (amount, time, anonymized features)
- Target: Binary label (0 = legitimate, 1 = fraud)

## 🔧 Configuration

Edit `config.py` to customize:

- **Federated Learning**: Number of clients, rounds, local epochs
- **Model Architecture**: Hidden layers, dropout, activation functions
- **Privacy**: Differential privacy parameters
- **Explanations**: Number of counterfactuals, diversity settings

## 📈 Results

After training, check the `results/` directory for:
- Training history plots
- Confusion matrix
- ROC curve
- Client data distribution

## 🔒 Privacy Features

- **No Raw Data Sharing**: Clients never share transaction data
- **Model Update Aggregation**: Only model weights are shared
- **Differential Privacy**: Optional noise addition for enhanced privacy
- **Secure Aggregation**: Encrypted updates (can be enabled)

## 💡 Explainability

The system generates counterfactual explanations:

```
Transaction flagged as FRAUDULENT (probability: 95%)

To make this transaction LEGITIMATE:
  • Change amount from $1,250.00 to $850.00 (-$400.00)
  • Change feature V4 from 2.45 to 0.32 (-2.13)
  → New prediction: LEGITIMATE (probability: 15%)
```

## 📊 Performance Metrics

The system tracks:
- **Accuracy**: Overall correctness
- **Precision**: Fraud detection accuracy
- **Recall**: Fraud detection coverage
- **F1-Score**: Balanced metric
- **AUC-ROC**: Model discrimination ability

## 🎓 Research Paper Reference

This implementation is based on:

**Tomisin Awosika, Raj Mani Shukla, and Bernardi Pranggono**  
*"Transparency and Privacy: The Role of Explainable AI and Federated Learning in Financial Fraud Detection"*  
IEEE Access, 2024  
DOI: 10.1109/ACCESS.2024.3394528

## 🛠️ Advanced Usage

### Custom Dataset

```python
from data_loader import FraudDataLoader

loader = FraudDataLoader(dataset_path='path/to/your/data.csv')
X_train, X_test, y_train, y_test = loader.load_data()
```

### Custom Model

```python
from models.fraud_detector import FraudDetectionModel

model = FraudDetectionModel(input_dim=30)
# Customize architecture in config.py
```

### Generate Explanations

```python
from explainability.counterfactual_generator import SimpleCounterfactualGenerator

explainer = SimpleCounterfactualGenerator(model)
cf_result = explainer.generate_counterfactual(instance, target_class=0)
```

## 📝 License

This project is for research and educational purposes.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional aggregation strategies
- More sophisticated counterfactual methods
- Real-world dataset integration
- Enhanced privacy mechanisms

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with**: TensorFlow, Scikit-learn, DiCE-ML, Matplotlib
