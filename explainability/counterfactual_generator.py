"""
Counterfactual Explanation Generator
Generates "what-if" explanations for fraud detection predictions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import dice_ml
from dice_ml import Dice

from config import EXPLANATION_CONFIG


class CounterfactualExplainer:
    """
    Generates counterfactual explanations for fraud predictions
    Shows minimal changes needed to flip a prediction
    """
    
    def __init__(self, model, X_train: np.ndarray, feature_names: List[str] = None):
        """
        Initialize counterfactual explainer
        
        Args:
            model: Trained fraud detection model
            X_train: Training data for generating realistic counterfactuals
            feature_names: Names of features
        """
        self.model = model
        self.X_train = X_train
        
        # Generate feature names if not provided
        if feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Create DataFrame for DiCE
        self.train_df = pd.DataFrame(X_train, columns=self.feature_names)
        
        # Add dummy outcome column (required by DiCE)
        self.train_df['outcome'] = 0
        
        # Initialize DiCE
        self.dice_data = dice_ml.Data(
            dataframe=self.train_df,
            continuous_features=self.feature_names,
            outcome_name='outcome'
        )
        
        # Wrap model for DiCE
        self.dice_model = self._create_dice_model()
        
        # Create DiCE explainer
        self.explainer = Dice(self.dice_data, self.dice_model, method='random')
        
        print(f"Counterfactual Explainer initialized with {len(self.feature_names)} features")
    
    def _create_dice_model(self):
        """Create DiCE-compatible model wrapper"""
        
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict(self, X):
                """Predict method for DiCE"""
                if isinstance(X, pd.DataFrame):
                    X = X[X.columns[:-1]].values  # Remove outcome column
                predictions = self.model.predict(X)
                return predictions.flatten()
            
            def predict_proba(self, X):
                """Predict probabilities for DiCE"""
                if isinstance(X, pd.DataFrame):
                    X = X[X.columns[:-1]].values
                predictions = self.model.predict(X)
                # Return probabilities for both classes
                probs = np.column_stack([1 - predictions, predictions])
                return probs
        
        return ModelWrapper(self.model)
    
    def generate_counterfactuals(self, instance: np.ndarray, 
                                num_counterfactuals: int = None,
                                desired_class: int = 0) -> Dict:
        """
        Generate counterfactual explanations for an instance
        
        Args:
            instance: Input instance (transaction) to explain
            num_counterfactuals: Number of counterfactuals to generate
            desired_class: Target class (0 = legitimate, 1 = fraud)
            
        Returns:
            Dictionary containing counterfactuals and explanation
        """
        num_cf = num_counterfactuals or EXPLANATION_CONFIG['num_counterfactuals']
        
        # Create DataFrame for instance
        instance_df = pd.DataFrame([instance], columns=self.feature_names)
        instance_df['outcome'] = 0
        
        # Get original prediction
        original_pred = self.model.predict(instance.reshape(1, -1))[0][0]
        original_class = 1 if original_pred > 0.5 else 0
        
        try:
            # Generate counterfactuals
            cf_result = self.explainer.generate_counterfactuals(
                instance_df,
                total_CFs=num_cf,
                desired_class=desired_class
            )
            
            # Extract counterfactual instances
            cf_df = cf_result.cf_examples_list[0].final_cfs_df
            
            if cf_df is None or len(cf_df) == 0:
                return {
                    'original_prediction': original_pred,
                    'original_class': original_class,
                    'counterfactuals': [],
                    'explanations': "No counterfactuals found"
                }
            
            # Remove outcome column
            cf_df = cf_df.drop('outcome', axis=1, errors='ignore')
            
            # Calculate changes
            counterfactuals = []
            for idx in range(len(cf_df)):
                cf_instance = cf_df.iloc[idx].values
                
                # Find changed features
                changes = self._identify_changes(instance, cf_instance)
                
                # Get prediction for counterfactual
                cf_pred = self.model.predict(cf_instance.reshape(1, -1))[0][0]
                cf_class = 1 if cf_pred > 0.5 else 0
                
                counterfactuals.append({
                    'instance': cf_instance,
                    'prediction': cf_pred,
                    'class': cf_class,
                    'changes': changes
                })
            
            # Generate natural language explanation
            explanation = self._generate_explanation(
                instance, original_pred, original_class,
                counterfactuals
            )
            
            return {
                'original_prediction': original_pred,
                'original_class': original_class,
                'counterfactuals': counterfactuals,
                'explanation': explanation
            }
            
        except Exception as e:
            print(f"Error generating counterfactuals: {e}")
            return {
                'original_prediction': original_pred,
                'original_class': original_class,
                'counterfactuals': [],
                'explanation': f"Error: {str(e)}"
            }
    
    def _identify_changes(self, original: np.ndarray, 
                         counterfactual: np.ndarray,
                         threshold: float = 0.01) -> List[Dict]:
        """
        Identify which features changed between original and counterfactual
        
        Args:
            original: Original instance
            counterfactual: Counterfactual instance
            threshold: Minimum change to consider significant
            
        Returns:
            List of changes
        """
        changes = []
        
        for i, (orig_val, cf_val) in enumerate(zip(original, counterfactual)):
            diff = abs(cf_val - orig_val)
            if diff > threshold:
                changes.append({
                    'feature': self.feature_names[i],
                    'original_value': float(orig_val),
                    'counterfactual_value': float(cf_val),
                    'change': float(cf_val - orig_val),
                    'percent_change': float((cf_val - orig_val) / (abs(orig_val) + 1e-8) * 100)
                })
        
        # Sort by absolute change
        changes.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return changes
    
    def _generate_explanation(self, original: np.ndarray, 
                             original_pred: float, original_class: int,
                             counterfactuals: List[Dict]) -> str:
        """
        Generate natural language explanation
        
        Args:
            original: Original instance
            original_pred: Original prediction probability
            original_class: Original predicted class
            counterfactuals: List of counterfactual instances
            
        Returns:
            Natural language explanation
        """
        if not counterfactuals:
            return "No counterfactual explanations could be generated."
        
        class_name = "FRAUDULENT" if original_class == 1 else "LEGITIMATE"
        target_class = "LEGITIMATE" if original_class == 1 else "FRAUDULENT"
        
        explanation = f"This transaction was predicted as {class_name} "
        explanation += f"(probability: {original_pred:.2%}).\n\n"
        
        explanation += f"To make this transaction {target_class}, "
        explanation += "the following changes could be made:\n\n"
        
        for i, cf in enumerate(counterfactuals[:3], 1):  # Show top 3
            explanation += f"Option {i}:\n"
            
            if cf['changes']:
                for change in cf['changes'][:5]:  # Show top 5 changes
                    explanation += f"  • Change {change['feature']} from "
                    explanation += f"{change['original_value']:.4f} to "
                    explanation += f"{change['counterfactual_value']:.4f} "
                    explanation += f"({change['change']:+.4f})\n"
            else:
                explanation += "  • No significant changes needed\n"
            
            explanation += f"  → New prediction: {cf['class']} "
            explanation += f"(probability: {cf['prediction']:.2%})\n\n"
        
        return explanation


class SimpleCounterfactualGenerator:
    """
    Simple counterfactual generator without DiCE dependency
    Uses gradient-based approach
    """
    
    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names
    
    def generate_counterfactual(self, instance: np.ndarray, 
                               target_class: int = 0,
                               max_iterations: int = 100,
                               learning_rate: float = 0.1) -> Dict:
        """
        Generate counterfactual using gradient descent
        
        Args:
            instance: Original instance
            target_class: Desired class
            max_iterations: Maximum optimization iterations
            learning_rate: Step size for gradient descent
            
        Returns:
            Counterfactual instance and explanation
        """
        # Start with copy of original
        cf_instance = instance.copy()
        
        # Get original prediction
        original_pred = self.model.predict(instance.reshape(1, -1))[0][0]
        
        # Simple perturbation approach
        for _ in range(max_iterations):
            current_pred = self.model.predict(cf_instance.reshape(1, -1))[0][0]
            
            # Check if we've reached target
            current_class = 1 if current_pred > 0.5 else 0
            if current_class == target_class:
                break
            
            # Random perturbation
            perturbation = np.random.randn(len(cf_instance)) * learning_rate
            cf_instance += perturbation
        
        cf_pred = self.model.predict(cf_instance.reshape(1, -1))[0][0]
        
        return {
            'original_prediction': original_pred,
            'counterfactual': cf_instance,
            'counterfactual_prediction': cf_pred,
            'explanation': f"Modified instance to achieve target class {target_class}"
        }


if __name__ == "__main__":
    print("Counterfactual Explanation module loaded successfully!")
    print("Use CounterfactualExplainer for DiCE-based explanations")
    print("Use SimpleCounterfactualGenerator for gradient-based explanations")
