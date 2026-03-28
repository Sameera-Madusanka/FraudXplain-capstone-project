"""
FraudXplain Web Application — Flask Backend
Serves ML model predictions, explanations, and metrics via REST API
"""

import sys
import os
import json
import glob
import io
import traceback

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.fraud_detector import FraudDetectionModel
from explainability.constrained_counterfactuals import ConstrainedCounterfactualGenerator
from explainability.privacy_validator import PrivacyValidator
from explainability.actionable_recourse import ActionableRecourseGenerator
from config import MODEL_CONFIG, FL_CONFIG, TRAINING_CONFIG, PRIVACY_CONFIG

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    'income', 'name_email_similarity', 'prev_address_months_count',
    'current_address_months_count', 'customer_age', 'days_since_request',
    'intended_balcon_amount', 'payment_type', 'zip_count_4w',
    'velocity_6h', 'velocity_24h', 'velocity_4w',
    'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
    'employment_status', 'credit_risk_score', 'email_is_free',
    'housing_status', 'phone_home_valid', 'phone_mobile_valid',
    'bank_months_count', 'has_other_cards', 'proposed_credit_limit',
    'foreign_request', 'source', 'session_length_in_minutes',
    'device_os', 'keep_alive_session', 'device_distinct_emails_8w',
    'device_fraud_count', 'month'
]

FEATURE_DESCRIPTIONS = {
    'income': 'Annual income (normalized)',
    'name_email_similarity': 'Name/email match score',
    'prev_address_months_count': 'Months at previous address',
    'current_address_months_count': 'Months at current address',
    'customer_age': 'Customer age',
    'days_since_request': 'Days since last request',
    'intended_balcon_amount': 'Intended balance/loan amount',
    'payment_type': 'Payment method type',
    'zip_count_4w': 'Zip code usage (4 weeks)',
    'velocity_6h': 'Transaction velocity (6h)',
    'velocity_24h': 'Transaction velocity (24h)',
    'velocity_4w': 'Transaction velocity (4 weeks)',
    'bank_branch_count_8w': 'Bank branch visits (8 weeks)',
    'date_of_birth_distinct_emails_4w': 'Emails with same DOB (4 weeks)',
    'employment_status': 'Employment status',
    'credit_risk_score': 'Credit risk score',
    'email_is_free': 'Free email provider?',
    'housing_status': 'Housing situation',
    'phone_home_valid': 'Valid home phone?',
    'phone_mobile_valid': 'Valid mobile phone?',
    'bank_months_count': 'Months with bank',
    'has_other_cards': 'Has other cards?',
    'proposed_credit_limit': 'Proposed credit limit',
    'foreign_request': 'Foreign request?',
    'source': 'Application source',
    'session_length_in_minutes': 'Session duration (min)',
    'device_os': 'Device OS',
    'keep_alive_session': 'Keep-alive session',
    'device_distinct_emails_8w': 'Distinct emails from device (8w)',
    'device_fraud_count': 'Fraud count from device',
    'month': 'Transaction month'
}

# Fallback hardcoded samples (only used if data loading fails)
FALLBACK_SAMPLES = {
    'legitimate': {
        'name': 'Legitimate Transaction (Conservative Profile)',
        'values': [
            0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, -1.0, -1.0, 0.0, -1.0, 0.5, 0.5, -0.5,
            0.5, 0.5, 0.5, 1.0, 0.5, 0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, -0.5, -1.5, 0.0
        ]
    },
    'fraudulent': {
        'name': 'Fraudulent Transaction (Extreme Risk Profile)',
        'values': [
            -2.5, -1.5, -2.5, -2.5, -1.5, 3.0, 3.5, -1.5, 3.0,
            3.5, 3.5, 3.5, 3.0, 3.0, -2.5, -3.0, 1.5,
            -2.5, -1.5, -1.5, -2.5, -1.5, 3.0, 2.5, -1.5, 3.5,
            -1.5, 2.5, 3.0, 3.5, 0.0
        ]
    }
}

# Protected attributes for explainability
PROTECTED_FEATURES = [
    'income', 'customer_age', 'employment_status',
    'housing_status', 'date_of_birth_distinct_emails_4w', 'foreign_request'
]

# ---------------------------------------------------------------------------
# Load model & threshold on startup
# ---------------------------------------------------------------------------
model = None
threshold = 0.85
cf_generator = None
privacy_validator = None
recourse_generator = None


def load_model():
    """Load the latest trained model and threshold."""
    global model, threshold, cf_generator, privacy_validator, recourse_generator

    model_files = glob.glob(os.path.join(RESULTS_DIR, 'fraud_model_*.h5'))
    if not model_files:
        print("⚠️  No trained model found in results/")
        return False

    latest = max(model_files)
    print(f"Loading model: {latest}")
    model = FraudDetectionModel(input_dim=31)
    model.load(latest)
    print("✅ Model loaded")

    # Load threshold
    threshold_file = os.path.join(RESULTS_DIR, 'optimal_threshold.txt')
    if os.path.exists(threshold_file):
        with open(threshold_file) as f:
            threshold = float(f.read().strip())
        print(f"✅ Threshold: {threshold:.4f}")

    # Initialize explainability components
    try:
        cf_generator = ConstrainedCounterfactualGenerator(
            model=model,
            feature_names=FEATURE_NAMES
        )
        privacy_validator = PrivacyValidator(
            protected_attributes=PROTECTED_FEATURES,
            feature_names=FEATURE_NAMES
        )
        recourse_generator = ActionableRecourseGenerator(feature_names=FEATURE_NAMES)
        print("✅ Explainability components loaded")
    except Exception as e:
        print(f"⚠️  Explainability init error: {e}")

    return True


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'threshold': threshold
    })


@app.route('/api/feature-names', methods=['GET'])
def feature_names():
    features = []
    for name in FEATURE_NAMES:
        features.append({
            'name': name,
            'description': FEATURE_DESCRIPTIONS.get(name, ''),
            'protected': name in PROTECTED_FEATURES
        })
    return jsonify({'features': features})


@app.route('/api/sample-transactions', methods=['GET'])
def sample_transactions():
    """Return sample transactions — dynamically find real ones the model flags."""
    if model is None:
        return jsonify(FALLBACK_SAMPLES)

    try:
        from data_loader_bank import BankAccountFraudLoader
        loader = BankAccountFraudLoader(os.path.join(PROJECT_ROOT, 'data', 'Base.csv'))
        X_train, X_test, y_train, y_test, _ = loader.load_and_split(
            sample_size=10000, balance_classes=False
        )

        fraud_idx = np.where(y_test == 1)[0]
        legit_idx = np.where(y_test == 0)[0]

        # Find fraud sample the model actually scores highest
        fraud_probs = model.predict(X_test[fraud_idx]).flatten()
        best_fraud_local = np.argmax(fraud_probs)
        best_fraud = fraud_idx[best_fraud_local]
        best_prob = fraud_probs[best_fraud_local]

        # Find legit sample with lowest score
        legit_probs = model.predict(X_test[legit_idx[:300]]).flatten()
        best_legit = legit_idx[np.argmin(legit_probs)]

        return jsonify({
            'legitimate': {
                'name': 'Legitimate Transaction (Real Data)',
                'values': X_test[best_legit].tolist()
            },
            'fraudulent': {
                'name': f'Fraudulent Transaction (Real Data — Model Score: {best_prob:.1%})',
                'values': X_test[best_fraud].tolist()
            }
        })
    except Exception as e:
        print(f'Sample loading fallback: {e}')
        return jsonify(FALLBACK_SAMPLES)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict fraud probability for a single transaction."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing "features" array (31 values)'}), 400

    features = data['features']
    if len(features) != 31:
        return jsonify({'error': f'Expected 31 features, got {len(features)}'}), 400

    try:
        X = np.array(features, dtype=np.float32).reshape(1, -1)
        prob = float(model.predict(X).flatten()[0])
        is_fraud = prob > threshold

        return jsonify({
            'fraud_probability': round(prob, 4),
            'is_fraud': is_fraud,
            'threshold': threshold,
            'confidence': 'High' if abs(prob - threshold) > 0.15 else 'Medium' if abs(prob - threshold) > 0.05 else 'Low'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/explain', methods=['POST'])
def explain():
    """Generate full constrained counterfactual explanation with privacy validation."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    if cf_generator is None:
        return jsonify({'error': 'Explainability module not initialized'}), 503

    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing "features" array'}), 400

    features = data['features']
    if len(features) != 31:
        return jsonify({'error': f'Expected 31 features, got {len(features)}'}), 400

    try:
        X = np.array(features, dtype=np.float32)
        prob = float(model.predict(X.reshape(1, -1)).flatten()[0])
        is_fraud = prob > threshold

        result = {
            'fraud_probability': round(prob, 4),
            'is_fraud': is_fraud,
            'threshold': threshold,
            'protected_attributes': {},
            'counterfactuals': [],
            'privacy_validation': [],
            'actionable_recourse': '',
            'privacy_report': {}
        }

        # 1. Protected attributes (always shown)
        for feat in PROTECTED_FEATURES:
            idx = FEATURE_NAMES.index(feat)
            result['protected_attributes'][feat] = {
                'value': round(float(features[idx]), 4),
                'description': FEATURE_DESCRIPTIONS.get(feat, ''),
                'status': 'PROTECTED — never changed'
            }

        # 2. Generate constrained counterfactuals (for any high-probability transaction)
        #    Use 0.5 instead of threshold so CFs are generated for demo purposes
        raw_cfs = []
        if prob > 0.5:
            try:
                raw_cfs = cf_generator.generate_constrained_counterfactual(
                    instance=X,
                    target_class=0,
                    num_counterfactuals=3
                )
            except Exception as e:
                result['counterfactuals_error'] = str(e)

            # 3. Process each counterfactual
            for i, cf_dict in enumerate(raw_cfs):
                cf_entry = {
                    'id': i + 1,
                    'new_probability': round(float(cf_dict['prediction']), 4),
                    'new_class': 'Legitimate' if cf_dict['prediction'] < 0.5 else 'Fraud',
                    'privacy_validated': cf_dict.get('privacy_validated', False),
                    'changes': []
                }

                # Extract changes (already structured by CF generator)
                for change in cf_dict.get('changes', []):
                    cf_entry['changes'].append({
                        'feature': change['feature'],
                        'original': round(change['original_value'], 4),
                        'counterfactual': round(change['counterfactual_value'], 4),
                        'change_magnitude': round(abs(change['change']), 4),
                        'description': FEATURE_DESCRIPTIONS.get(change['feature'], ''),
                        'is_actionable': change.get('is_actionable', True)
                    })

                result['counterfactuals'].append(cf_entry)

                # 4. Privacy validation per counterfactual
                if privacy_validator:
                    try:
                        is_valid, violations = privacy_validator.validate_counterfactual(
                            original=X,
                            counterfactual=cf_dict['counterfactual']
                        )
                        result['privacy_validation'].append({
                            'counterfactual_id': i + 1,
                            'privacy_preserved': is_valid,
                            'violations': violations,
                            'status': '✅ Privacy GUARANTEED — no protected attributes changed' if is_valid else '❌ Privacy VIOLATION detected'
                        })
                    except Exception as e:
                        result['privacy_validation'].append({
                            'counterfactual_id': i + 1,
                            'error': str(e)
                        })

            # 5. Generate actionable recourse text
            if recourse_generator and raw_cfs:
                try:
                    recourse_text = recourse_generator.generate_recourse(
                        original_instance=X,
                        counterfactuals=raw_cfs,
                        original_pred=prob
                    )
                    result['actionable_recourse'] = recourse_text
                except Exception as e:
                    result['recourse_error'] = str(e)

            # 6. Overall privacy report
            if privacy_validator:
                try:
                    result['privacy_report'] = privacy_validator.get_privacy_report()
                    privacy_validator.reset_violations()
                except Exception as e:
                    result['privacy_report_error'] = str(e)

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch predict from CSV upload."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files accepted'}), 400

    try:
        df = pd.read_csv(file)

        # Try to use only the 31 features
        if len(df.columns) >= 31:
            X = df.iloc[:, :31].values.astype(np.float32)
        else:
            return jsonify({'error': f'CSV must have at least 31 columns, got {len(df.columns)}'}), 400

        predictions = model.predict(X).flatten()

        results = []
        for i, prob in enumerate(predictions):
            results.append({
                'row': i + 1,
                'fraud_probability': round(float(prob), 4),
                'is_fraud': bool(prob > threshold),
                'risk_level': 'High' if prob > 0.9 else 'Medium' if prob > threshold else 'Low'
            })

        fraud_count = sum(1 for r in results if r['is_fraud'])
        return jsonify({
            'total': len(results),
            'fraud_detected': fraud_count,
            'legitimate': len(results) - fraud_count,
            'threshold': threshold,
            'predictions': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model architecture and configuration."""
    return jsonify({
        'architecture': {
            'input_dim': 31,
            'hidden_layers': MODEL_CONFIG['hidden_layers'],
            'dropout_rate': MODEL_CONFIG['dropout_rate'],
            'activation': MODEL_CONFIG['activation'],
            'l2_reg': MODEL_CONFIG.get('l2_reg', 0),
            'output': 'sigmoid'
        },
        'training': {
            'algorithm': 'Federated Averaging (FedAvg)',
            'num_clients': FL_CONFIG['num_clients'],
            'num_rounds': FL_CONFIG['num_rounds'],
            'local_epochs': FL_CONFIG['local_epochs'],
            'batch_size': FL_CONFIG['batch_size'],
            'learning_rate': MODEL_CONFIG['learning_rate'],
            'loss': TRAINING_CONFIG['loss'],
            'class_weight': str(TRAINING_CONFIG['class_weight']),
            'differential_privacy': PRIVACY_CONFIG['use_differential_privacy']
        },
        'threshold': threshold,
        'features': len(FEATURE_NAMES),
        'dataset': 'Bank Account Fraud Dataset Suite (NeurIPS 2022)',
        'variants': ['Base', 'Variant I', 'Variant II', 'Variant III', 'Variant IV', 'Variant V']
    })


@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Return model performance metrics (if available)."""
    try:
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
        from data_loader_bank import BankAccountFraudLoader

        loader = BankAccountFraudLoader(os.path.join(PROJECT_ROOT, 'data', 'Base.csv'))
        X_train, X_test, y_train, y_test, _ = loader.load_and_split(
            sample_size=50000, balance_classes=False
        )
        y_proba = model.predict(X_test).flatten()
        y_pred = (y_proba > threshold).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        return jsonify({
            'auc_roc': round(auc, 4),
            'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            'f1_score': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            'confusion_matrix': {
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            },
            'threshold': threshold,
            'test_samples': len(y_test),
            'fraud_mean_prediction': round(float(y_proba[y_test == 1].mean()), 4),
            'legit_mean_prediction': round(float(y_proba[y_test == 0].mean()), 4),
            'separation': round(float(y_proba[y_test == 1].mean() - y_proba[y_test == 0].mean()), 4)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/images/<name>', methods=['GET'])
def serve_image(name):
    """Serve result images (ROC curve, confusion matrix, training history)."""
    allowed = ['roc_curve.png', 'confusion_matrix.png', 'training_history.png']
    if name not in allowed:
        return jsonify({'error': f'Image not found. Available: {allowed}'}), 404

    filepath = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(filepath):
        return jsonify({'error': f'{name} not found'}), 404

    return send_file(filepath, mimetype='image/png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("  FraudXplain Backend API")
    print("=" * 60)

    if load_model():
        print("\n🚀 Starting server on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("\n❌ Cannot start without a trained model.")
        print("   Run: python train_bank_account.py --multi-variant --rounds 30")
