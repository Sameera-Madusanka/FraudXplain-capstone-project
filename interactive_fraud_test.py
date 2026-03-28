"""
Interactive Fraud Detection Test
Allows manual input of transaction features via terminal
"""

import numpy as np
import glob
from models.fraud_detector import FraudDetectionModel
from explainability.constrained_counterfactuals import ConstrainedCounterfactualGenerator
from explainability.privacy_validator import PrivacyValidator
from explainability.actionable_recourse import ActionableRecourseGenerator


# Bank Account Fraud Dataset Features (31 features)
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

# Feature descriptions for user guidance
FEATURE_DESCRIPTIONS = {
    'income': 'Annual income (normalized, e.g., -2.0 to 3.0)',
    'name_email_similarity': 'Similarity between name and email (-2.0 to 2.0)',
    'prev_address_months_count': 'Months at previous address (normalized)',
    'current_address_months_count': 'Months at current address (normalized)',
    'customer_age': 'Customer age (normalized, e.g., -2.0 to 2.0)',
    'days_since_request': 'Days since last request (normalized)',
    'intended_balcon_amount': 'Intended balance/loan amount (normalized)',
    'payment_type': 'Payment type (normalized categorical)',
    'zip_count_4w': 'Zip code usage in 4 weeks (normalized)',
    'velocity_6h': 'Transaction velocity in 6 hours (normalized)',
    'velocity_24h': 'Transaction velocity in 24 hours (normalized)',
    'velocity_4w': 'Transaction velocity in 4 weeks (normalized)',
    'bank_branch_count_8w': 'Bank branch visits in 8 weeks (normalized)',
    'date_of_birth_distinct_emails_4w': 'Distinct emails with same DOB (normalized)',
    'employment_status': 'Employment status (normalized categorical)',
    'credit_risk_score': 'Credit risk score (normalized, e.g., -2.0 to 2.0)',
    'email_is_free': 'Is email from free provider? (normalized binary)',
    'housing_status': 'Housing status (normalized categorical)',
    'phone_home_valid': 'Is home phone valid? (normalized binary)',
    'phone_mobile_valid': 'Is mobile phone valid? (normalized binary)',
    'bank_months_count': 'Months with bank (normalized)',
    'has_other_cards': 'Has other credit cards? (normalized binary)',
    'proposed_credit_limit': 'Proposed credit limit (normalized)',
    'foreign_request': 'Is foreign request? (normalized binary)',
    'source': 'Application source (normalized categorical)',
    'session_length_in_minutes': 'Session length in minutes (normalized)',
    'device_os': 'Device operating system (normalized categorical)',
    'keep_alive_session': 'Keep alive session setting (normalized)',
    'device_distinct_emails_8w': 'Distinct emails from device (normalized)',
    'device_fraud_count': 'Fraud count from device (normalized)',
    'month': 'Month of transaction (normalized)'
}


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def get_sample_transactions():
    """Load real sample transactions that the model classifies correctly"""
    try:
        from data_loader_bank import BankAccountFraudLoader
        from models.fraud_detector import FraudDetectionModel
        import glob
        
        # Load model and threshold to find samples the model actually flags
        model_files = glob.glob('results/fraud_model_*.h5')
        threshold = 0.85
        try:
            with open('results/optimal_threshold.txt') as f:
                threshold = float(f.read().strip())
        except: pass
        
        mdl = None
        if model_files:
            mdl = FraudDetectionModel(input_dim=31)
            mdl.load(max(model_files))
        
        # Load a larger sample to find good examples
        loader = BankAccountFraudLoader(dataset_path='data/Base.csv')
        X_train, X_test, y_train, y_test, feature_names = loader.load_and_split(
            sample_size=10000,
            balance_classes=False  # Keep original distribution
        )
        
        legit_indices = np.where(y_test == 0)[0]
        fraud_indices = np.where(y_test == 1)[0]
        
        if len(legit_indices) > 0 and len(fraud_indices) > 0:
            # Find a fraud sample the model actually predicts as fraud
            best_fraud_idx = fraud_indices[0]
            best_fraud_prob = 0
            
            if mdl is not None:
                fraud_X = X_test[fraud_indices]
                fraud_probs = mdl.predict(fraud_X).flatten()
                # Find fraud samples above threshold
                above_threshold = np.where(fraud_probs > threshold)[0]
                if len(above_threshold) > 0:
                    # Pick a random sample from the high-probability ones to increase diversity
                    best_idx = np.random.choice(above_threshold)
                    best_fraud_idx = fraud_indices[best_idx]
                    best_fraud_prob = fraud_probs[best_idx]
                    print(f"  Found diverse fraud sample with model score: {best_fraud_prob:.1%}")
                else:
                    # No sample above threshold, pick highest scored one
                    best_idx = np.argmax(fraud_probs)
                    best_fraud_idx = fraud_indices[best_idx]
                    best_fraud_prob = fraud_probs[best_idx]
                    print(f"  Best fraud sample score: {best_fraud_prob:.1%} (below threshold {threshold:.1%})")
            
            # Find a legit sample the model predicts as legit (low score)
            best_legit_idx = legit_indices[0]
            if mdl is not None:
                legit_X = X_test[legit_indices[:200]]  # Check first 200
                legit_probs = mdl.predict(legit_X).flatten()
                best_legit_local = np.argmin(legit_probs)
                best_legit_idx = legit_indices[best_legit_local]
            
            return {
                'legitimate': {
                    'name': 'Legitimate Transaction (Real Data)',
                    'values': X_test[best_legit_idx].tolist()
                },
                'fraudulent': {
                    'name': f'Fraudulent Transaction (Real Data, Score: {best_fraud_prob:.1%})',
                    'values': X_test[best_fraud_idx].tolist()
                }
            }
    except Exception as e:
        print(f"⚠️  Could not load real data: {e}")
        print("Using fallback sample transactions...\n")
    
    # Fallback to conservative legitimate profile if data loading fails
    return {
        'legitimate': {
            'name': 'Legitimate Transaction (Conservative Profile)',
            'values': [
                0.0,   # income (average)
                0.5,   # name_email_similarity (good match)
                0.5,   # prev_address_months_count (some stability)
                1.0,   # current_address_months_count (stable)
                0.0,   # customer_age (average)
                0.0,   # days_since_request (normal)
                0.0,   # intended_balcon_amount (moderate)
                0.0,   # payment_type (normal)
                0.0,   # zip_count_4w (normal)
                -1.0,  # velocity_6h (low - good sign)
                -1.0,  # velocity_24h (low - good sign)
                -1.0,  # velocity_4w (low - good sign)
                0.0,   # bank_branch_count_8w (normal)
                -1.0,  # date_of_birth_distinct_emails_4w (low - good)
                0.5,   # employment_status (employed)
                0.5,   # credit_risk_score (decent)
                -0.5,  # email_is_free (not free - good)
                0.5,   # housing_status (stable)
                0.5,   # phone_home_valid (valid)
                0.5,   # phone_mobile_valid (valid)
                1.0,   # bank_months_count (established)
                0.5,   # has_other_cards (has some)
                0.0,   # proposed_credit_limit (moderate)
                -1.0,  # foreign_request (no - good)
                0.0,   # source (normal)
                0.0,   # session_length_in_minutes (normal)
                0.0,   # device_os (common)
                0.0,   # keep_alive_session (normal)
                -0.5,  # device_distinct_emails_8w (low - good)
                -1.5,  # device_fraud_count (very low - good)
                0.0    # month (normal)
            ]
        },
        'fraudulent': {
            'name': 'Fraudulent Transaction (High Risk Profile)',
            'values': [
                -1.5,  # income (low/suspicious)
                -0.8,  # name_email_similarity (low - mismatch)
                -1.5,  # prev_address_months_count (unstable)
                -1.5,  # current_address_months_count (unstable)
                -1.0,  # customer_age (young/suspicious)
                2.0,   # days_since_request (very recent - rushed)
                2.5,   # intended_balcon_amount (very high)
                -1.0,  # payment_type (unusual)
                2.0,   # zip_count_4w (high - suspicious)
                2.5,   # velocity_6h (very high)
                2.5,   # velocity_24h (very high)
                2.5,   # velocity_4w (very high)
                2.0,   # bank_branch_count_8w (high - suspicious)
                2.0,   # date_of_birth_distinct_emails_4w (high - suspicious)
                -1.5,  # employment_status (unemployed)
                -2.0,  # credit_risk_score (very poor)
                1.0,   # email_is_free (yes - suspicious)
                -1.0,  # housing_status (no stable housing)
                -1.0,  # phone_home_valid (invalid)
                -1.0,  # phone_mobile_valid (invalid)
                -1.5,  # bank_months_count (new customer)
                -1.0,  # has_other_cards (no)
                2.5,   # proposed_credit_limit (very high)
                1.0,   # foreign_request (yes)
                -1.0,  # source (suspicious source)
                -1.0,  # session_length_in_minutes (very short - rushed)
                -1.5,  # device_os (unusual)
                -1.0,  # keep_alive_session (suspicious)
                2.0,   # device_distinct_emails_8w (high - suspicious)
                2.5,   # device_fraud_count (high)
                0.0    # month (normal)
            ]
        }
    }


def manual_input_mode():
    """Allow user to manually input all 31 features"""
    print("\n📝 Manual Input Mode")
    print("Enter values for each feature (normalized values typically range from -3 to 3)")
    print("Press Enter to use default value (0.0)\n")
    
    transaction = []
    for i, feature in enumerate(FEATURE_NAMES):
        desc = FEATURE_DESCRIPTIONS.get(feature, '')
        default = 0.0
        
        while True:
            try:
                value_str = input(f"[{i+1}/31] {feature}\n        ({desc})\n        [default: {default}]: ").strip()
                
                if value_str == '':
                    value = default
                else:
                    value = float(value_str)
                
                transaction.append(value)
                break
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
    
    return np.array(transaction)


def quick_test_mode():
    """Use pre-defined sample transactions"""
    samples = get_sample_transactions()
    
    print("\n🚀 Quick Test Mode - Select a sample transaction:\n")
    print("1. Legitimate Transaction (Low Risk)")
    print("2. Fraudulent Transaction (High Risk)")
    print("3. Go back to main menu")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        return np.array(samples['legitimate']['values']), samples['legitimate']['name']
    elif choice == '2':
        return np.array(samples['fraudulent']['values']), samples['fraudulent']['name']
    else:
        return None, None


def test_transaction(model, transaction, transaction_name="Custom Transaction"):
    """Test a transaction and generate explanation if fraud detected"""
    
    print_header(f"Testing: {transaction_name}")
    
    # Load optimal threshold (computed by check_model.py or train_bank_account.py)
    threshold = 0.5  # default fallback
    try:
        with open('results/optimal_threshold.txt', 'r') as f:
            threshold = float(f.read().strip())
    except FileNotFoundError:
        pass
    
    # Make prediction
    fraud_prob = model.predict(transaction.reshape(1, -1))[0][0]
    is_fraud = fraud_prob > threshold
    
    print(f"\n   Fraud Detection Result:")
    print(f"   Prediction: {'FRAUD' if is_fraud else 'LEGITIMATE'}")
    print(f"   Fraud Probability: {fraud_prob:.2%}")
    print(f"   Threshold: {threshold:.2%}")
    print(f"   Confidence: {'High' if abs(fraud_prob - threshold) > 0.15 else 'Medium' if abs(fraud_prob - threshold) > 0.05 else 'Low'}")
    
    if is_fraud:
        print("\n" + "-"*80)
        print("🔒 Generating Privacy-Guaranteed Explanation...")
        print("-"*80)
        
        # Initialize CF generator
        cf_generator = ConstrainedCounterfactualGenerator(
            model=model,
            feature_names=FEATURE_NAMES,
            protected_attrs_config='data/protected_attributes.json'
        )
        
        # Generate counterfactuals
        try:
            counterfactuals = cf_generator.generate_constrained_counterfactual(
                instance=transaction,
                target_class=0,
                max_iterations=100,
                num_counterfactuals=3
            )
            
            # We skip printing the raw explanation string here because the 
            # ActionableRecourseGenerator below provides a much cleaner,
            # user-friendly, and actionable version of the same data!
            
            # Validate privacy
            print("\n" + "-"*80)
            print("🔐 Privacy Validation")
            print("-"*80)
            
            validator = PrivacyValidator(
                protected_attributes=cf_generator.config['protected_attributes']['attributes'],
                feature_names=FEATURE_NAMES
            )
            
            all_valid = True
            for i, cf in enumerate(counterfactuals):
                is_valid, violations = validator.validate_counterfactual(
                    transaction,
                    cf['counterfactual']
                )
                
                if is_valid:
                    print(f"✅ Option {i+1}: Privacy validated")
                else:
                    print(f"❌ Option {i+1}: Privacy violations:")
                    for violation in violations:
                        print(f"   - {violation}")
                    all_valid = False
            
            if all_valid:
                print("\n🔒 All explanations protect sensitive information!")
            
            # Generate actionable recourse
            print("\n" + "-"*80)
            print("💡 Actionable Recourse")
            print("-"*80)
            
            recourse_gen = ActionableRecourseGenerator(FEATURE_NAMES)
            recourse = recourse_gen.generate_recourse(
                original_instance=transaction,
                counterfactuals=counterfactuals,
                original_pred=fraud_prob
            )
            
            print(recourse)
            
        except Exception as e:
            print(f"❌ Error generating explanation: {e}")
    else:
        print("\n✅ Transaction approved - No explanation needed")


def main():
    """Main interactive loop"""
    
    print_header("INTERACTIVE FRAUD DETECTION SYSTEM")
    print("\nTest your fraud detection model with manual inputs!")
    
    # Load model
    print("\n📦 Loading trained model...")
    model_files = glob.glob('results/fraud_model_*.h5')
    
    if not model_files:
        print("❌ No trained model found!")
        print("Please train the model first:")
        print("   python train_bank_account.py --sample-size 10000 --num-clients 3 --rounds 5")
        return
    
    latest_model = max(model_files)
    print(f"   Loading: {latest_model}")
    
    model = FraudDetectionModel(input_dim=31)
    model.load(latest_model)
    print("   ✅ Model loaded successfully!")
    
    # Main loop
    while True:
        print_header("MAIN MENU")
        print("\nChoose an option:\n")
        print("1. Quick Test (Use sample transactions)")
        print("2. Manual Input (Enter all 31 features)")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            transaction, name = quick_test_mode()
            if transaction is not None:
                test_transaction(model, transaction, name)
                input("\nPress Enter to continue...")
        
        elif choice == '2':
            transaction = manual_input_mode()
            test_transaction(model, transaction, "Custom Transaction")
            input("\nPress Enter to continue...")
        
        elif choice == '3':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
