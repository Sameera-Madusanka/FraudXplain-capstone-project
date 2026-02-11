# Interactive Fraud Detection Test - Quick Guide

## How to Use

Run the interactive test:
```bash
python interactive_fraud_test.py
```

---

## Menu Options

### Option 1: Quick Test (Recommended for First Try)
Uses pre-built sample transactions:
- **Legitimate Transaction**: Low-risk profile (good credit, stable address, etc.)
- **Fraudulent Transaction**: High-risk profile (poor credit, suspicious velocity, etc.)

**What you'll see**:
1. Fraud probability prediction
2. If fraud detected:
   - Privacy-guaranteed explanation
   - Protected attributes (unchanged)
   - Actionable recourse options
   - Privacy validation results

---

### Option 2: Manual Input
Enter all 31 features manually through the terminal.

**Features you'll input** (all normalized -3 to 3):

1. `income` - Annual income
2. `name_email_similarity` - Name/email match
3. `prev_address_months_count` - Months at previous address
4. `current_address_months_count` - Months at current address
5. `customer_age` - Customer age
6. `days_since_request` - Days since last request
7. `intended_balcon_amount` - Loan/balance amount
8. `payment_type` - Payment method
9. `zip_count_4w` - Zip code usage (4 weeks)
10. `velocity_6h` - Transaction velocity (6 hours)
11. `velocity_24h` - Transaction velocity (24 hours)
12. `velocity_4w` - Transaction velocity (4 weeks)
13. `bank_branch_count_8w` - Branch visits (8 weeks)
14. `date_of_birth_distinct_emails_4w` - Emails with same DOB
15. `employment_status` - Employment
16. `credit_risk_score` - Credit score
17. `email_is_free` - Free email provider?
18. `housing_status` - Housing situation
19. `phone_home_valid` - Valid home phone?
20. `phone_mobile_valid` - Valid mobile phone?
21. `bank_months_count` - Months with bank
22. `has_other_cards` - Has other cards?
23. `proposed_credit_limit` - Requested credit limit
24. `foreign_request` - Foreign request?
25. `source` - Application source
26. `session_length_in_minutes` - Session duration
27. `device_os` - Device OS
28. `keep_alive_session` - Keep-alive setting
29. `device_distinct_emails_8w` - Emails from device
30. `device_fraud_count` - Fraud count from device
31. `month` - Transaction month

**Tips**:
- Press Enter to use default (0.0)
- Positive values = higher/better
- Negative values = lower/worse
- Typical range: -3 to 3

---

## Sample Transaction Values

### Legitimate Transaction Example
```
income: 1.5 (good income)
credit_risk_score: 1.8 (good credit)
employment_status: 1.5 (employed)
velocity_6h: -0.5 (low velocity)
device_fraud_count: -1.0 (no fraud history)
... etc
```

### Fraudulent Transaction Example
```
income: -1.5 (low income)
credit_risk_score: -2.0 (poor credit)
employment_status: -1.5 (unemployed)
velocity_6h: 2.5 (very high velocity - suspicious)
device_fraud_count: 2.5 (high fraud history)
... etc
```

---

## Expected Output

### For Legitimate Transaction:
```
🔍 Fraud Detection Result:
   Prediction: ✅ LEGITIMATE
   Fraud Probability: 15.3%
   Confidence: High

✅ Transaction approved - No explanation needed
```

### For Fraudulent Transaction:
```
🔍 Fraud Detection Result:
   Prediction: 🚨 FRAUD
   Fraud Probability: 89.7%
   Confidence: High

🔒 Generating Privacy-Guaranteed Explanation...

======================================================================
🔒 PRIVACY-GUARANTEED FRAUD EXPLANATION
======================================================================

🚨 Transaction Flagged as FRAUDULENT
   Fraud Probability: 89.7%

🔒 PRIVACY GUARANTEE:
   The following sensitive attributes are PROTECTED:
   • income: -1.50 (unchanged)
   • customer_age: -1.00 (unchanged)
   • employment_status: -1.50 (unchanged)
   ... etc

📋 ACTIONABLE RECOURSE:
   To clear this alert, you can:

   Option 1:
   ✓ Improve credit score from -2.00 to 1.50
   ✓ Use verified device
   → Estimated fraud probability: 25.3%

🔐 Privacy Validation
✅ Option 1: Privacy validated
✅ Option 2: Privacy validated
✅ Option 3: Privacy validated

🔒 All explanations protect sensitive information!

💡 Actionable Recourse
[Clear guidance on how to clear the fraud alert]
```

---

## What This Demonstrates

✅ **Fraud Detection**: Accurate predictions on transaction data  
✅ **Privacy Protection**: Sensitive attributes never changed  
✅ **Constrained Counterfactuals**: Only actionable features suggested  
✅ **Formal Validation**: Automated privacy compliance checking  
✅ **User-Friendly**: Clear, actionable guidance

---

## Quick Start

1. Run the script:
   ```bash
   python interactive_fraud_test.py
   ```

2. Choose Option 1 (Quick Test)

3. Select "Fraudulent Transaction" to see full explanation

4. Review the privacy-guaranteed explanation and recourse options

That's it! You'll see your complete system in action.
