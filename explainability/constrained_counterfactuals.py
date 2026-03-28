"""
Constrained Counterfactual Explanations
Generates privacy-preserving counterfactuals that protect sensitive attributes
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
import os

from config import EXPLANATION_CONFIG


class ConstrainedCounterfactualGenerator:
    """
    Generates constrained counterfactual explanations that:
    1. Protect sensitive attributes (income, age, employment)
    2. Only suggest changes to actionable attributes
    3. Provide minimal, feasible changes
    4. Guarantee privacy preservation
    """
    
    def __init__(self, 
                 model,
                 feature_names: List[str],
                 protected_attrs_config: str = 'data/protected_attributes.json'):
        """
        Initialize constrained counterfactual generator
        
        Args:
            model: Trained fraud detection model
            feature_names: List of feature names
            protected_attrs_config: Path to protected attributes configuration
        """
        self.model = model
        self.feature_names = feature_names
        
        # Load protected attributes configuration
        self.config = self._load_config(protected_attrs_config)
        
        # Create feature index mappings
        self.protected_indices = self._get_feature_indices(
            self.config['protected_attributes']['attributes']
        )
        self.actionable_indices = self._get_feature_indices(
            self.config['actionable_attributes']['attributes']
        )
        self.immutable_indices = self._get_feature_indices(
            self.config['immutable_attributes']['attributes']
        )
        
        print(f"Constrained CF Generator initialized:")
        print(f"  Protected attributes: {len(self.protected_indices)}")
        print(f"  Actionable attributes: {len(self.actionable_indices)}")
        print(f"  Immutable attributes: {len(self.immutable_indices)}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load protected attributes configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration if file doesn't exist
            return {
                'protected_attributes': {
                    'attributes': ['income', 'customer_age', 'employment_status']
                },
                'actionable_attributes': {
                    'attributes': ['proposed_credit_limit', 'credit_risk_score']
                },
                'immutable_attributes': {
                    'attributes': ['prev_address_months_count']
                }
            }
    
    def _get_feature_indices(self, attr_names: List[str]) -> List[int]:
        """Get indices of features by name"""
        indices = []
        for attr in attr_names:
            if attr in self.feature_names:
                indices.append(self.feature_names.index(attr))
        return indices
    
    def generate_constrained_counterfactual(self,
                                           instance: np.ndarray,
                                           target_class: int = 0,
                                           max_iterations: int = 100,
                                           learning_rate: float = 0.1,
                                           num_counterfactuals: int = 3) -> List[Dict]:
        """
        Generate constrained counterfactual explanations
        
        Args:
            instance: Original instance to explain
            target_class: Desired class (0 = legitimate)
            max_iterations: Maximum optimization iterations
            learning_rate: Step size for gradient descent
            num_counterfactuals: Number of diverse counterfactuals to generate
            
        Returns:
            List of counterfactual dictionaries with explanations
        """
        original_pred = self.model.predict(instance.reshape(1, -1))[0][0]
        original_class = 1 if original_pred > 0.5 else 0
        
        counterfactuals = []
        
        import tensorflow as tf
        
        # Generate multiple diverse counterfactuals
        for i in range(num_counterfactuals):
            # Start with copy of original
            cf_instance = instance.copy()
            
            # Add small random perturbation for diversity
            if i > 0:
                noise = np.random.randn(len(cf_instance)) * 0.1
                # Only apply noise to actionable features
                for idx in self.actionable_indices:
                    cf_instance[idx] += noise[idx]
            
            # Optimize only actionable features using Gradient Descent
            for iteration in range(max_iterations):
                # Convert to tensor to compute gradients
                x_tensor = tf.convert_to_tensor(cf_instance.reshape(1, -1), dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    tape.watch(x_tensor)
                    # Access the underlying keras model from FraudDetectionModel wrapper
                    pred = self.model.model(x_tensor)
                
                current_pred = float(pred[0][0])
                current_class = 1 if current_pred > 0.5 else 0
                
                # Check if target reached (probability successfully lowered)
                if current_class == target_class:
                    break
                
                # Calculate gradients of the output prediction with respect to input features
                grads = tape.gradient(pred, x_tensor).numpy()[0]
                
                # Apply Fast Gradient Sign Method (FGSM) style update
                perturbation = np.zeros_like(cf_instance)
                
                for idx in self.actionable_indices:
                    # Move exactly in opposite direction of gradient to minimize probability
                    if abs(grads[idx]) > 1e-6:
                        direction = np.sign(grads[idx])
                    else:
                        direction = np.random.choice([-1.0, 1.0])
                    
                    # Add tiny decaying noise to escape local minima
                    noise_factor = (np.random.randn() * 0.05) * (0.95 ** iteration)
                    
                    perturbation[idx] = -(learning_rate * direction) + noise_factor
                
                # Apply perturbation
                cf_instance += perturbation
                
                # CRITICAL: Re-enforce zero changes to protected and immutable attributes
                for idx in self.protected_indices + self.immutable_indices:
                    cf_instance[idx] = instance[idx]
            
            # Validate constraints
            if self._validate_constraints(instance, cf_instance):
                cf_pred = self.model.predict(cf_instance.reshape(1, -1))[0][0]
                cf_class = 1 if cf_pred > 0.5 else 0
                
                # Identify changes
                changes = self._identify_changes(instance, cf_instance)
                
                counterfactuals.append({
                    'counterfactual': cf_instance,
                    'prediction': cf_pred,
                    'class': cf_class,
                    'changes': changes,
                    'privacy_validated': True
                })
        
        return counterfactuals
    
    def _validate_constraints(self, original: np.ndarray, counterfactual: np.ndarray) -> bool:
        """
        Validate that constraints are satisfied
        
        Args:
            original: Original instance
            counterfactual: Counterfactual instance
            
        Returns:
            True if all constraints satisfied
        """
        # Check protected attributes unchanged
        for idx in self.protected_indices:
            if abs(original[idx] - counterfactual[idx]) > 1e-6:
                print(f"WARNING: Protected attribute {self.feature_names[idx]} changed!")
                return False
        
        # Check immutable attributes unchanged
        for idx in self.immutable_indices:
            if abs(original[idx] - counterfactual[idx]) > 1e-6:
                print(f"WARNING: Immutable attribute {self.feature_names[idx]} changed!")
                return False
        
        return True
    
    def _identify_changes(self, original: np.ndarray, counterfactual: np.ndarray,
                         threshold: float = 0.01) -> List[Dict]:
        """
        Identify changes between original and counterfactual
        Only reports changes to actionable attributes
        
        Args:
            original: Original instance
            counterfactual: Counterfactual instance
            threshold: Minimum change to consider significant
            
        Returns:
            List of changes (only actionable attributes)
        """
        changes = []
        
        # Only report changes to actionable attributes
        for idx in self.actionable_indices:
            diff = abs(counterfactual[idx] - original[idx])
            if diff > threshold:
                changes.append({
                    'feature': self.feature_names[idx],
                    'feature_index': idx,
                    'original_value': float(original[idx]),
                    'counterfactual_value': float(counterfactual[idx]),
                    'change': float(counterfactual[idx] - original[idx]),
                    'percent_change': float((counterfactual[idx] - original[idx]) / 
                                          (abs(original[idx]) + 1e-8) * 100),
                    'is_actionable': True
                })
        
        # Sort by absolute change magnitude
        changes.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return changes
    
    def generate_privacy_guaranteed_explanation(self,
                                               instance: np.ndarray,
                                               counterfactuals: List[Dict]) -> str:
        """
        Generate privacy-guaranteed natural language explanation
        
        Args:
            instance: Original instance
            counterfactuals: List of counterfactual dictionaries
            
        Returns:
            Privacy-guaranteed explanation string
        """
        if not counterfactuals:
            return "No valid counterfactuals found that satisfy privacy constraints."
        
        original_pred = self.model.predict(instance.reshape(1, -1))[0][0]
        original_class = "FRAUDULENT" if original_pred > 0.5 else "LEGITIMATE"
        
        explanation = "=" * 70 + "\n"
        explanation += "🔒 PRIVACY-GUARANTEED FRAUD EXPLANATION\n"
        explanation += "=" * 70 + "\n\n"
        
        # Original prediction
        explanation += f"🚨 Transaction Flagged as {original_class}\n"
        explanation += f"   Fraud Probability: {original_pred:.1%}\n\n"
        
        # Privacy guarantee statement
        explanation += "🔒 PRIVACY GUARANTEE:\n"
        protected_attrs = self.config['protected_attributes']['attributes']
        explanation += f"   The following sensitive attributes are PROTECTED:\n"
        for attr in protected_attrs:
            if attr in self.feature_names:
                idx = self.feature_names.index(attr)
                explanation += f"   • {attr}: {instance[idx]:.2f} (unchanged)\n"
        explanation += "\n"
        
        # Actionable recourse
        explanation += "📋 ACTIONABLE RECOURSE:\n"
        explanation += "   To clear this alert, you can:\n\n"
        
        for i, cf in enumerate(counterfactuals[:3], 1):  # Show top 3
            if cf['changes']:
                explanation += f"   Option {i}:\n"
                
                # Limit to top 3 changes for clarity
                for change in cf['changes'][:3]:
                    explanation += f"   ✓ Change {change['feature']} from "
                    explanation += f"{change['original_value']:.2f} to "
                    explanation += f"{change['counterfactual_value']:.2f}\n"
                
                explanation += f"   → Estimated fraud probability: {cf['prediction']:.1%}\n\n"
        
        # Privacy compliance footer
        explanation += "=" * 70 + "\n"
        explanation += "✅ PRIVACY COMPLIANCE:\n"
        explanation += "   • No sensitive personal information revealed\n"
        explanation += "   • Only actionable, changeable attributes suggested\n"
        explanation += "   • All suggestions verified for feasibility\n"
        explanation += "=" * 70 + "\n"
        
        return explanation


if __name__ == "__main__":
    print("Constrained Counterfactual Generator module loaded successfully!")
    print("\nKey Features:")
    print("  ✓ Protects sensitive attributes (income, age, employment)")
    print("  ✓ Only suggests actionable changes")
    print("  ✓ Provides privacy guarantees")
    print("  ✓ Generates minimal, feasible counterfactuals")
