"""
Actionable Recourse Generator
Generates user-friendly, actionable explanations from constrained counterfactuals
"""

import numpy as np
from typing import List, Dict, Tuple


class ActionableRecourseGenerator:
    """
    Converts constrained counterfactuals into actionable recourse
    Provides clear, user-friendly guidance on how to change prediction
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize actionable recourse generator
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        
        # Define user-friendly descriptions for features
        self.feature_descriptions = {
            'proposed_credit_limit': 'requested credit limit',
            'credit_risk_score': 'credit score',
            'payment_type': 'payment method',
            'device_os': 'device operating system',
            'keep_alive_session': 'session keep-alive setting',
            'device_distinct_emails_8w': 'number of distinct emails used',
            'customer_age': 'age',
            'income': 'annual income',
            'employment_status': 'employment status',
            'housing_status': 'housing situation'
        }
        
        # Define actionable suggestions templates
        self.action_templates = {
            'proposed_credit_limit': {
                'increase': 'Request a higher credit limit of ${new_value:,.0f}',
                'decrease': 'Reduce your credit limit request to ${new_value:,.0f}'
            },
            'credit_risk_score': {
                'increase': 'Improve your credit score to {new_value:.0f} points',
                'decrease': 'Your credit score would need to be {new_value:.0f}'
            },
            'payment_type': {
                'change': 'Change payment method from "{old_value}" to "{new_value}"'
            },
            'device_os': {
                'change': 'Use a device with {new_value} operating system'
            }
        }
    
    def generate_recourse(self,
                         original_instance: np.ndarray,
                         counterfactuals: List[Dict],
                         original_pred: float,
                         max_options: int = 3) -> str:
        """
        Generate actionable recourse from counterfactuals
        
        Args:
            original_instance: Original instance
            counterfactuals: List of counterfactual dictionaries
            original_pred: Original prediction probability
            max_options: Maximum number of options to show
            
        Returns:
            Actionable recourse string
        """
        original_class = "FRAUDULENT" if original_pred > 0.5 else "LEGITIMATE"
        
        recourse = "\n" + "=" * 70 + "\n"
        recourse += "🚨 FRAUD ALERT - ACTIONABLE RECOURSE\n"
        recourse += "=" * 70 + "\n\n"
        
        # Alert header
        recourse += f"Transaction flagged as {original_class}\n"
        recourse += f"Fraud Confidence: {original_pred:.1%}\n\n"
        
        if not counterfactuals:
            recourse += "⚠️  No actionable recourse available at this time.\n"
            recourse += "Please contact support for manual review.\n"
            return recourse
        
        # Actionable options
        recourse += "📋 TO CLEAR THIS ALERT, YOU CAN:\n\n"
        
        for i, cf in enumerate(counterfactuals[:max_options], 1):
            if not cf['changes']:
                continue
            
            recourse += f"Option {i}"
            
            # Add difficulty indicator
            num_changes = len(cf['changes'])
            if num_changes == 1:
                recourse += " (Easiest):\n"
            elif num_changes == 2:
                recourse += " (Moderate):\n"
            else:
                recourse += " (Alternative):\n"
            
            # List actionable changes
            for change in cf['changes'][:5]:  # Limit to top 5 changes
                action = self._format_action(change)
                recourse += f"  ✓ {action}\n"
            
            # Show expected outcome
            recourse += f"  → Expected fraud probability after changes: {cf['prediction']:.1%}\n"
            
            if cf['prediction'] < 0.5:
                recourse += "  ✅ This should clear the fraud alert\n"
            
            recourse += "\n"
        
        # Additional guidance
        recourse += "💡 ADDITIONAL GUIDANCE:\n"
        recourse += self._generate_guidance(counterfactuals)
        
        recourse += "\n" + "=" * 70 + "\n"
        
        return recourse
    
    def _format_action(self, change: Dict) -> str:
        """
        Format a single change as an actionable statement
        
        Args:
            change: Change dictionary
            
        Returns:
            Formatted action string
        """
        feature = change['feature']
        old_value = change['original_value']
        new_value = change['counterfactual_value']
        delta = change['change']
        
        # Use template if available
        if feature in self.action_templates:
            template = self.action_templates[feature]
            
            if delta > 0 and 'increase' in template:
                return template['increase'].format(
                    old_value=old_value, new_value=new_value, delta=delta
                )
            elif delta < 0 and 'decrease' in template:
                return template['decrease'].format(
                    old_value=old_value, new_value=new_value, delta=abs(delta)
                )
            elif 'change' in template:
                return template['change'].format(
                    old_value=old_value, new_value=new_value
                )
        
        # Default formatting
        friendly_name = self.feature_descriptions.get(feature, feature)
        
        if delta > 0:
            return f"Increase {friendly_name} from {old_value:.2f} to {new_value:.2f}"
        else:
            return f"Decrease {friendly_name} from {old_value:.2f} to {new_value:.2f}"
    
    def _generate_guidance(self, counterfactuals: List[Dict]) -> str:
        """
        Generate additional guidance based on counterfactuals
        
        Args:
            counterfactuals: List of counterfactual dictionaries
            
        Returns:
            Guidance string
        """
        guidance = ""
        
        # Analyze common changes across counterfactuals
        all_features = set()
        for cf in counterfactuals:
            for change in cf['changes']:
                all_features.add(change['feature'])
        
        if 'proposed_credit_limit' in all_features:
            guidance += "  • Adjusting your credit limit request can significantly impact fraud risk\n"
        
        if 'credit_risk_score' in all_features:
            guidance += "  • Improving your credit score would help reduce fraud alerts\n"
        
        if 'payment_type' in all_features:
            guidance += "  • Consider using a different payment method\n"
        
        if not guidance:
            guidance = "  • Make the suggested changes to reduce fraud risk\n"
        
        return guidance
    
    def generate_comparison_table(self,
                                 original: np.ndarray,
                                 counterfactual: np.ndarray,
                                 changes: List[Dict]) -> str:
        """
        Generate a comparison table showing before/after
        
        Args:
            original: Original instance
            counterfactual: Counterfactual instance
            changes: List of changes
            
        Returns:
            Comparison table string
        """
        table = "\n📊 BEFORE vs AFTER COMPARISON:\n\n"
        table += f"{'Feature':<30} {'Before':<15} {'After':<15} {'Change':<15}\n"
        table += "-" * 75 + "\n"
        
        for change in changes[:10]:  # Limit to top 10
            feature = change['feature']
            old_val = change['original_value']
            new_val = change['counterfactual_value']
            delta = change['change']
            
            table += f"{feature:<30} {old_val:<15.2f} {new_val:<15.2f} {delta:+15.2f}\n"
        
        table += "-" * 75 + "\n"
        
        return table


if __name__ == "__main__":
    print("Actionable Recourse Generator module loaded successfully!")
    print("\nKey Features:")
    print("  ✓ User-friendly actionable explanations")
    print("  ✓ Multiple recourse options")
    print("  ✓ Difficulty indicators")
    print("  ✓ Expected outcomes")
