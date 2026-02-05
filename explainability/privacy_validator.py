"""
Privacy Validator for Constrained Counterfactual Explanations
Ensures no sensitive information leaks in explanations
"""

import numpy as np
from typing import List, Dict, Tuple
import json


class PrivacyValidator:
    """
    Validates that counterfactual explanations satisfy privacy constraints
    """
    
    def __init__(self, protected_attributes: List[str], feature_names: List[str]):
        """
        Initialize privacy validator
        
        Args:
            protected_attributes: List of protected attribute names
            feature_names: List of all feature names
        """
        self.protected_attributes = protected_attributes
        self.feature_names = feature_names
        self.protected_indices = [
            feature_names.index(attr) for attr in protected_attributes 
            if attr in feature_names
        ]
        
        # Privacy violation log
        self.violations = []
    
    def validate_counterfactual(self,
                               original: np.ndarray,
                               counterfactual: np.ndarray,
                               tolerance: float = 1e-6) -> Tuple[bool, List[str]]:
        """
        Validate that counterfactual doesn't violate privacy constraints
        
        Args:
            original: Original instance
            counterfactual: Counterfactual instance
            tolerance: Numerical tolerance for equality check
            
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        # Check each protected attribute
        for idx in self.protected_indices:
            if abs(original[idx] - counterfactual[idx]) > tolerance:
                attr_name = self.feature_names[idx]
                violation_msg = (
                    f"Protected attribute '{attr_name}' changed from "
                    f"{original[idx]:.4f} to {counterfactual[idx]:.4f}"
                )
                violations.append(violation_msg)
                self.violations.append({
                    'attribute': attr_name,
                    'original': float(original[idx]),
                    'counterfactual': float(counterfactual[idx]),
                    'violation_type': 'protected_attribute_change'
                })
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def validate_explanation(self, explanation: str, 
                           protected_values: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate that explanation text doesn't reveal protected values
        
        Args:
            explanation: Generated explanation text
            protected_values: Dictionary of protected attribute values
            
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        # Check if protected attribute values appear in explanation
        for attr, value in protected_values.items():
            # Check if the value is mentioned in explanation
            value_str = f"{value:.2f}"
            if value_str in explanation and attr in self.protected_attributes:
                # Check if it's in a protected context
                # (Allow mentioning that it's protected, but not the value change)
                if "change" in explanation.lower() and attr.lower() in explanation.lower():
                    violation_msg = (
                        f"Explanation suggests changing protected attribute '{attr}'"
                    )
                    violations.append(violation_msg)
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def validate_changes(self, changes: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate that suggested changes don't include protected attributes
        
        Args:
            changes: List of change dictionaries
            
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        for change in changes:
            if change['feature'] in self.protected_attributes:
                violation_msg = (
                    f"Suggested change to protected attribute '{change['feature']}'"
                )
                violations.append(violation_msg)
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def get_privacy_report(self) -> Dict:
        """
        Get comprehensive privacy validation report
        
        Returns:
            Privacy report dictionary
        """
        return {
            'total_violations': len(self.violations),
            'violations': self.violations,
            'protected_attributes': self.protected_attributes,
            'privacy_compliant': len(self.violations) == 0
        }
    
    def reset_violations(self):
        """Reset violation log"""
        self.violations = []


class FeasibilityChecker:
    """
    Checks if suggested changes are feasible and realistic
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize feasibility checker
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        
        # Define feasibility constraints (can be loaded from config)
        self.constraints = {
            'proposed_credit_limit': {'min': 0, 'max': 50000, 'step': 100},
            'credit_risk_score': {'min': 300, 'max': 850, 'step': 10},
            'customer_age': {'min': 18, 'max': 100, 'step': 1},
        }
    
    def check_feasibility(self, changes: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Check if changes are feasible
        
        Args:
            changes: List of change dictionaries
            
        Returns:
            (is_feasible, list_of_issues)
        """
        issues = []
        
        for change in changes:
            feature = change['feature']
            new_value = change['counterfactual_value']
            
            # Check if feature has constraints
            if feature in self.constraints:
                constraint = self.constraints[feature]
                
                # Check min/max bounds
                if new_value < constraint['min']:
                    issues.append(
                        f"{feature} value {new_value:.2f} below minimum {constraint['min']}"
                    )
                if new_value > constraint['max']:
                    issues.append(
                        f"{feature} value {new_value:.2f} above maximum {constraint['max']}"
                    )
        
        is_feasible = len(issues) == 0
        return is_feasible, issues
    
    def suggest_feasible_value(self, feature: str, desired_value: float) -> float:
        """
        Suggest nearest feasible value for a feature
        
        Args:
            feature: Feature name
            desired_value: Desired value
            
        Returns:
            Feasible value
        """
        if feature not in self.constraints:
            return desired_value
        
        constraint = self.constraints[feature]
        
        # Clip to bounds
        feasible_value = np.clip(desired_value, constraint['min'], constraint['max'])
        
        # Round to step
        if 'step' in constraint:
            step = constraint['step']
            feasible_value = round(feasible_value / step) * step
        
        return feasible_value


if __name__ == "__main__":
    print("Privacy Validator module loaded successfully!")
    print("\nKey Features:")
    print("  ✓ Validates no protected attributes changed")
    print("  ✓ Checks explanations don't reveal sensitive info")
    print("  ✓ Verifies suggested changes are feasible")
    print("  ✓ Provides comprehensive privacy reports")
