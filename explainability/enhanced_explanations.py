"""
Enhanced Counterfactual Explanations with Better Interpretability
Handles both anonymized (PCA) and interpretable features
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class EnhancedExplanationGenerator:
    """
    Generates more interpretable counterfactual explanations
    Handles anonymized features with additional context
    """
    
    def __init__(self, feature_names: List[str], dataset_type: str = 'credit_card'):
        """
        Initialize enhanced explainer
        
        Args:
            feature_names: List of feature names
            dataset_type: 'credit_card' (PCA features) or 'bank_account' (interpretable)
        """
        self.feature_names = feature_names
        self.dataset_type = dataset_type
        
        # Define feature interpretations for credit card dataset
        self.pca_interpretations = {
            'V1': 'Transaction pattern component 1 (possibly related to transaction type)',
            'V2': 'Transaction pattern component 2 (possibly related to merchant category)',
            'V3': 'Transaction pattern component 3 (possibly related to location)',
            'V4': 'Transaction pattern component 4 (possibly related to time of day)',
            'V5': 'Transaction pattern component 5',
            'V6': 'Transaction pattern component 6',
            'V7': 'Transaction pattern component 7',
            'V8': 'Transaction pattern component 8',
            'V9': 'Transaction pattern component 9',
            'V10': 'Transaction pattern component 10',
            'V11': 'Transaction pattern component 11',
            'V12': 'Transaction pattern component 12',
            'V13': 'Transaction pattern component 13',
            'V14': 'Transaction pattern component 14',
            'V15': 'Transaction pattern component 15',
            'V16': 'Transaction pattern component 16',
            'V17': 'Transaction pattern component 17',
            'V18': 'Transaction pattern component 18',
            'V19': 'Transaction pattern component 19',
            'V20': 'Transaction pattern component 20',
            'V21': 'Transaction pattern component 21',
            'V22': 'Transaction pattern component 22',
            'V23': 'Transaction pattern component 23',
            'V24': 'Transaction pattern component 24',
            'V25': 'Transaction pattern component 25',
            'V26': 'Transaction pattern component 26',
            'V27': 'Transaction pattern component 27',
            'V28': 'Transaction pattern component 28',
            'Time': 'Time elapsed since first transaction (seconds)',
            'Amount': 'Transaction amount (currency)'
        }
    
    def generate_enhanced_explanation(self, 
                                     original: np.ndarray,
                                     counterfactual: np.ndarray,
                                     original_pred: float,
                                     cf_pred: float,
                                     changes: List[Dict]) -> str:
        """
        Generate enhanced natural language explanation
        
        Args:
            original: Original instance
            counterfactual: Counterfactual instance
            original_pred: Original prediction probability
            cf_pred: Counterfactual prediction probability
            changes: List of feature changes
            
        Returns:
            Enhanced explanation string
        """
        original_class = "FRAUDULENT" if original_pred > 0.5 else "LEGITIMATE"
        cf_class = "LEGITIMATE" if cf_pred < 0.5 else "FRAUDULENT"
        
        explanation = "="*70 + "\n"
        explanation += "COUNTERFACTUAL EXPLANATION FOR FRAUD DETECTION\n"
        explanation += "="*70 + "\n\n"
        
        # Original prediction
        explanation += f"📊 ORIGINAL PREDICTION:\n"
        explanation += f"   Classification: {original_class}\n"
        explanation += f"   Fraud Probability: {original_pred:.2%}\n\n"
        
        # Counterfactual prediction
        explanation += f"🔄 COUNTERFACTUAL PREDICTION:\n"
        explanation += f"   Classification: {cf_class}\n"
        explanation += f"   Fraud Probability: {cf_pred:.2%}\n\n"
        
        # Changes needed
        explanation += f"💡 CHANGES NEEDED TO FLIP PREDICTION:\n"
        explanation += f"   (Showing top {min(len(changes), 10)} most important changes)\n\n"
        
        if self.dataset_type == 'credit_card':
            explanation += self._format_pca_changes(changes[:10])
        else:
            explanation += self._format_interpretable_changes(changes[:10])
        
        # Summary
        explanation += "\n" + "="*70 + "\n"
        explanation += "📝 SUMMARY:\n"
        explanation += f"   • Total features changed: {len(changes)}\n"
        explanation += f"   • Prediction change: {original_pred:.2%} → {cf_pred:.2%}\n"
        
        if cf_pred < 0.5 and original_pred > 0.5:
            explanation += f"   ✅ Successfully changed from FRAUD to LEGITIMATE\n"
        elif cf_pred > 0.5 and original_pred < 0.5:
            explanation += f"   ✅ Successfully changed from LEGITIMATE to FRAUD\n"
        else:
            explanation += f"   ⚠️  Prediction class did not flip\n"
        
        explanation += "="*70 + "\n"
        
        return explanation
    
    def _format_pca_changes(self, changes: List[Dict]) -> str:
        """Format changes for PCA features with interpretations"""
        formatted = ""
        
        for i, change in enumerate(changes, 1):
            feature = change['feature']
            orig_val = change['original_value']
            cf_val = change['counterfactual_value']
            delta = change['change']
            
            formatted += f"   {i}. {feature}: {orig_val:.4f} → {cf_val:.4f} "
            formatted += f"({delta:+.4f})\n"
            
            # Add interpretation if available
            if feature in self.pca_interpretations:
                formatted += f"      ℹ️  {self.pca_interpretations[feature]}\n"
            
            formatted += "\n"
        
        # Add note about PCA features
        formatted += "   ⚠️  NOTE: V1-V28 are anonymized features from PCA transformation.\n"
        formatted += "      These represent complex patterns in the original transaction data.\n"
        formatted += "      Only 'Time' and 'Amount' are directly interpretable.\n"
        
        return formatted
    
    def _format_interpretable_changes(self, changes: List[Dict]) -> str:
        """Format changes for interpretable features"""
        formatted = ""
        
        for i, change in enumerate(changes, 1):
            feature = change['feature']
            orig_val = change['original_value']
            cf_val = change['counterfactual_value']
            delta = change['change']
            
            formatted += f"   {i}. {feature}:\n"
            formatted += f"      Original: {orig_val:.2f}\n"
            formatted += f"      Counterfactual: {cf_val:.2f}\n"
            formatted += f"      Change: {delta:+.2f}\n\n"
        
        return formatted
    
    def create_visual_summary(self, changes: List[Dict], top_n: int = 5) -> str:
        """
        Create a visual summary of top changes
        
        Args:
            changes: List of changes
            top_n: Number of top changes to show
            
        Returns:
            Visual summary string
        """
        summary = "\n📊 TOP FEATURE CHANGES (Visual Summary):\n\n"
        
        top_changes = changes[:top_n]
        max_feature_len = max(len(c['feature']) for c in top_changes)
        
        for change in top_changes:
            feature = change['feature'].ljust(max_feature_len)
            delta = change['change']
            
            # Create visual bar
            bar_length = int(abs(delta) * 10)
            bar_length = min(bar_length, 30)  # Cap at 30 chars
            
            if delta > 0:
                bar = "+" + "█" * bar_length
            else:
                bar = "-" + "█" * bar_length
            
            summary += f"   {feature} {bar} {delta:+.4f}\n"
        
        return summary
    
    def generate_actionable_insights(self, changes: List[Dict]) -> str:
        """
        Generate actionable insights from changes
        
        Args:
            changes: List of changes
            
        Returns:
            Actionable insights string
        """
        insights = "\n💼 ACTIONABLE INSIGHTS:\n\n"
        
        # Find Amount changes
        amount_changes = [c for c in changes if c['feature'] == 'Amount']
        if amount_changes:
            change = amount_changes[0]
            if change['change'] < 0:
                insights += f"   • Reducing transaction amount by ${abs(change['change']):.2f} "
                insights += "would make it less suspicious\n"
            else:
                insights += f"   • Increasing transaction amount by ${change['change']:.2f} "
                insights += "would make it less suspicious\n"
        
        # Find Time changes
        time_changes = [c for c in changes if c['feature'] == 'Time']
        if time_changes:
            change = time_changes[0]
            hours_change = change['change'] / 3600
            insights += f"   • Transaction timing differs by {abs(hours_change):.1f} hours "
            insights += "from typical legitimate patterns\n"
        
        # General pattern changes
        pca_changes = [c for c in changes if c['feature'].startswith('V')]
        if pca_changes:
            insights += f"   • {len(pca_changes)} transaction pattern features need adjustment\n"
            insights += "   • This suggests the transaction profile differs from legitimate patterns\n"
        
        if not amount_changes and not time_changes and not pca_changes:
            insights += "   • No specific actionable insights available\n"
        
        return insights


# Example usage
if __name__ == "__main__":
    # Test with credit card features
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    
    explainer = EnhancedExplanationGenerator(
        feature_names=feature_names,
        dataset_type='credit_card'
    )
    
    # Simulate changes
    changes = [
        {'feature': 'Amount', 'original_value': 1250.0, 'counterfactual_value': 850.0, 
         'change': -400.0, 'percent_change': -32.0},
        {'feature': 'V4', 'original_value': 2.145, 'counterfactual_value': 0.823, 
         'change': -1.322, 'percent_change': -61.6},
        {'feature': 'V14', 'original_value': -1.234, 'counterfactual_value': 0.123, 
         'change': 1.357, 'percent_change': 110.0},
    ]
    
    explanation = explainer.generate_enhanced_explanation(
        original=np.array([0] * 30),
        counterfactual=np.array([0] * 30),
        original_pred=0.945,
        cf_pred=0.123,
        changes=changes
    )
    
    print(explanation)
    print(explainer.create_visual_summary(changes))
    print(explainer.generate_actionable_insights(changes))
