import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

# 1. Define the Focal Loss Function
@register_keras_serializable(package="Custom", name="focal_loss_fixed")
def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss function.
    Args:
        y_true: Ground truth labels (tensor of shape (batch_size, num_classes)).
        y_pred: Predicted probabilities (tensor of shape (batch_size, num_classes)).
        gamma: Focusing parameter to reduce the relative loss for well-classified examples.
        alpha: Weighting factor to balance the importance of positive and negative examples.
    Returns:
        A scalar tensor representing the mean focal loss.
    """
    # Ensure y_true and y_pred are of the same data type
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    loss = alpha * tf.keras.backend.pow((1 - p_t), gamma) * bce
    return tf.keras.backend.mean(loss)

# 2. Wrapper for Parameterization with proper serialization
@register_keras_serializable(package="Custom", name="focal_loss")
def focal_loss(gamma=2.0, alpha=0.25):
    """
    A wrapper function to allow you to specify the gamma and alpha
    parameters when using the loss.
    """
    @register_keras_serializable(package="Custom", name="focal_loss_fn")
    def focal_loss_fn(y_true, y_pred):
        return focal_loss_fixed(y_true, y_pred, gamma=gamma, alpha=alpha)
    
    # These attributes help with serialization
    focal_loss_fn.__name__ = "focal_loss_fn"
    focal_loss_fn._uses_gamma = gamma  # Store parameters as attributes
    focal_loss_fn._uses_alpha = alpha
    
    return focal_loss_fn