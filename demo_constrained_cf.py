"""
Demo: Constrained Counterfactual Explanations
Demonstrates privacy-preserving actionable insights
"""

import numpy as np
from explainability.constrained_counterfactuals import ConstrainedCounterfactualGenerator
from explainability.privacy_validator import PrivacyValidator, FeasibilityChecker
from explainability.actionable_recourse import ActionableRecourseGenerator


def demo_constrained_counterfactuals():
    """
    Demonstrate constrained counterfactual explanations
    """
    print("\n" + "=" * 80)
    print("DEMO: Privacy-Guaranteed Actionable Insights")
    print("Constrained Counterfactual Explanations for Fraud Detection")
    print("=" * 80 + "\n")
    
    # Simulate Bank Account Fraud features
    feature_names = [
        'income', 'customer_age', 'employment_status', 'housing_status',  # Protected
        'proposed_credit_limit', 'credit_risk_score', 'payment_type',     # Actionable
        'prev_address_months_count', 'current_address_months_count',      # Immutable
        'device_os', 'keep_alive_session', 'device_distinct_emails_8w'
    ]
    
    # Create a mock fraudulent instance
    fraudulent_instance = np.array([
        45000,    # income (protected)
        28,       # customer_age (protected)
        1,        # employment_status (protected)
        0,        # housing_status (protected)
        8000,     # proposed_credit_limit (actionable - HIGH)
        420,      # credit_risk_score (actionable - LOW)
        2,        # payment_type (actionable)
        6,        # prev_address_months_count (immutable)
        3,        # current_address_months_count (immutable)
        1,        # device_os
        0,        # keep_alive_session
        5         # device_distinct_emails_8w
    ])
    
    print("📋 ORIGINAL TRANSACTION DETAILS:\n")
    print(f"  Income: ${fraudulent_instance[0]:,.0f} (PROTECTED)")
    print(f"  Age: {fraudulent_instance[1]:.0f} years (PROTECTED)")
    print(f"  Employment Status: {fraudulent_instance[2]:.0f} (PROTECTED)")
    print(f"  Housing Status: {fraudulent_instance[3]:.0f} (PROTECTED)")
    print(f"  Proposed Credit Limit: ${fraudulent_instance[4]:,.0f} (ACTIONABLE)")
    print(f"  Credit Risk Score: {fraudulent_instance[5]:.0f} (ACTIONABLE)")
    print(f"  Payment Type: {fraudulent_instance[6]:.0f} (ACTIONABLE)")
    print()
    
    # Simulate a counterfactual (manually created for demo)
    counterfactual_instance = fraudulent_instance.copy()
    # Only change actionable features
    counterfactual_instance[4] = 3000  # Reduce credit limit
    counterfactual_instance[5] = 650   # Improve credit score
    counterfactual_instance[6] = 1     # Change payment type
    
    print("🔄 COUNTERFACTUAL (What needs to change):\n")
    print(f"  Income: ${counterfactual_instance[0]:,.0f} (UNCHANGED ✓)")
    print(f"  Age: {counterfactual_instance[1]:.0f} years (UNCHANGED ✓)")
    print(f"  Employment Status: {counterfactual_instance[2]:.0f} (UNCHANGED ✓)")
    print(f"  Housing Status: {counterfactual_instance[3]:.0f} (UNCHANGED ✓)")
    print(f"  Proposed Credit Limit: ${counterfactual_instance[4]:,.0f} (CHANGED)")
    print(f"  Credit Risk Score: {counterfactual_instance[5]:.0f} (CHANGED)")
    print(f"  Payment Type: {counterfactual_instance[6]:.0f} (CHANGED)")
    print()
    
    # Privacy Validation
    print("🔒 PRIVACY VALIDATION:\n")
    protected_attrs = ['income', 'customer_age', 'employment_status', 'housing_status']
    validator = PrivacyValidator(protected_attrs, feature_names)
    
    is_valid, violations = validator.validate_counterfactual(
        fraudulent_instance, counterfactual_instance
    )
    
    if is_valid:
        print("  ✅ PRIVACY COMPLIANT: No protected attributes changed")
    else:
        print("  ❌ PRIVACY VIOLATION:")
        for violation in violations:
            print(f"     - {violation}")
    print()
    
    # Identify changes
    changes = []
    for i, (orig, cf) in enumerate(zip(fraudulent_instance, counterfactual_instance)):
        if abs(orig - cf) > 0.01:
            changes.append({
                'feature': feature_names[i],
                'feature_index': i,
                'original_value': float(orig),
                'counterfactual_value': float(cf),
                'change': float(cf - orig),
                'percent_change': float((cf - orig) / (abs(orig) + 1e-8) * 100),
                'is_actionable': feature_names[i] in [
                    'proposed_credit_limit', 'credit_risk_score', 'payment_type'
                ]
            })
    
    # Validate changes
    is_valid_changes, change_violations = validator.validate_changes(changes)
    if is_valid_changes:
        print("  ✅ CHANGES VALID: Only actionable attributes suggested")
    else:
        print("  ❌ INVALID CHANGES:")
        for violation in change_violations:
            print(f"     - {violation}")
    print()
    
    # Feasibility Check
    print("✓ FEASIBILITY CHECK:\n")
    feasibility_checker = FeasibilityChecker(feature_names)
    is_feasible, issues = feasibility_checker.check_feasibility(changes)
    
    if is_feasible:
        print("  ✅ FEASIBLE: All suggested changes are realistic")
    else:
        print("  ⚠️  FEASIBILITY ISSUES:")
        for issue in issues:
            print(f"     - {issue}")
    print()
    
    # Generate Actionable Recourse
    print("=" * 80)
    recourse_gen = ActionableRecourseGenerator(feature_names)
    
    counterfactuals = [{
        'counterfactual': counterfactual_instance,
        'prediction': 0.15,  # Simulated low fraud probability
        'class': 0,
        'changes': changes,
        'privacy_validated': True
    }]
    
    recourse = recourse_gen.generate_recourse(
        fraudulent_instance,
        counterfactuals,
        original_pred=0.92  # Simulated high fraud probability
    )
    
    print(recourse)
    
    # Comparison Table
    comparison = recourse_gen.generate_comparison_table(
        fraudulent_instance,
        counterfactual_instance,
        changes
    )
    print(comparison)
    
    # Privacy Report
    print("\n" + "=" * 80)
    print("📊 PRIVACY COMPLIANCE REPORT:\n")
    report = validator.get_privacy_report()
    print(f"  Total Violations: {report['total_violations']}")
    print(f"  Privacy Compliant: {'✅ YES' if report['privacy_compliant'] else '❌ NO'}")
    print(f"  Protected Attributes: {', '.join(report['protected_attributes'])}")
    print("=" * 80 + "\n")
    
    print("✅ DEMO COMPLETE!")
    print("\nKey Takeaways:")
    print("  1. Protected attributes (income, age, employment) NEVER changed")
    print("  2. Only actionable features suggested for modification")
    print("  3. Privacy guarantees formally validated")
    print("  4. User receives clear, actionable recourse")
    print()


if __name__ == "__main__":
    demo_constrained_counterfactuals()
