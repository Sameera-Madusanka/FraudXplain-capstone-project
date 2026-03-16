"""
Focal Loss for Imbalanced Classification
Addresses class imbalance without synthetic data (SMOTE alternative)

Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)

How it works:
- Down-weights easy examples (obvious legitimate transactions)
- Up-weights hard examples (borderline/ambiguous transactions)
- Focuses model learning on the most informative samples

Parameters:
- gamma: Focusing parameter. Higher = more focus on hard examples
  - gamma=0: Equivalent to binary cross-entropy
  - gamma=2: Standard focal loss (recommended)
  - gamma=5: Very aggressive focusing
  
- alpha: Class balancing weight for the positive (fraud) class
  - alpha=0.5: Equal weight
  - alpha=0.75: Moderate fraud focus (recommended)
  - alpha=0.9: Heavy fraud focus
"""

import tensorflow as tf


def focal_loss(gamma=2.0, alpha=0.75):
    """
    Focal Loss for binary classification
    
    Automatically focuses on hard-to-classify samples, making it
    ideal for imbalanced datasets like fraud detection (~1% fraud).
    
    Args:
        gamma: Focusing parameter (default: 2.0)
               Higher values = more focus on hard examples
        alpha: Balancing weight for positive class (default: 0.75)
               Higher values = more importance on fraud detection
    
    Returns:
        Focal loss function compatible with Keras
    """
    def focal_loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        # Binary cross-entropy components
        bce_pos = -y_true * tf.math.log(y_pred)
        bce_neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        
        # Focal modulating factor: (1 - pt)^gamma
        # pt = p for positive class, (1-p) for negative class
        focal_pos = tf.pow(1.0 - y_pred, gamma)  # Hard positive examples
        focal_neg = tf.pow(y_pred, gamma)          # Hard negative examples
        
        # Apply alpha balancing and focal modulation
        loss = alpha * focal_pos * bce_pos + (1.0 - alpha) * focal_neg * bce_neg
        
        return tf.reduce_mean(loss)
    
    # Set function name for Keras serialization
    focal_loss_fn.__name__ = f'focal_loss_g{gamma}_a{alpha}'
    
    return focal_loss_fn
