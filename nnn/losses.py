"""
Loss functions for neural networks.

All loss functions support automatic differentiation.
"""

import numpy as np
from .tensor import Tensor
from .activations import softmax


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Mean Squared Error (MSE) loss.
    
    MSE = (1/n) * Σ(pred_i - target_i)²
    
    Args:
        pred: Predicted values tensor.
        target: Target values tensor.
        
    Returns:
        Scalar tensor containing the MSE loss.
    """
    # Element-wise squared difference
    diff = pred - target
    squared_diff = diff * diff
    
    # Mean over all elements
    loss = squared_diff.mean()
    return loss


def cross_entropy_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Cross-Entropy loss for classification.
    
    CE = -Σ target_i * log(pred_i)
    
    For multi-class classification, target should be one-hot encoded
    or class indices. This implementation handles both.
    
    Args:
        pred: Predicted logits (before softmax) of shape (batch_size, num_classes).
        target: Target labels. Can be:
            - One-hot encoded: shape (batch_size, num_classes)
            - Class indices: shape (batch_size,) with integer class labels
        
    Returns:
        Scalar tensor containing the cross-entropy loss.
    """
    # Apply softmax to get probabilities
    probs = softmax(pred, axis=-1)
    
    # Handle both one-hot and class index targets
    if target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1):
        # Class indices: convert to one-hot
        num_classes = pred.shape[-1]
        batch_size = target.shape[0]
        
        # Flatten if needed
        target_indices = target.data.flatten().astype(int)
        
        # Create one-hot encoding
        target_one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
        target_one_hot[np.arange(batch_size), target_indices] = 1.0
        target_one_hot = Tensor(target_one_hot, requires_grad=False)
    else:
        # Already one-hot encoded
        target_one_hot = target
    
    # Compute cross-entropy: -Σ target * log(prob)
    # Use tensor's log method for proper autograd
    log_probs = probs.log()
    
    # Negative log likelihood
    neg_log_likelihood = -(target_one_hot * log_probs)
    
    # Sum over classes and mean over batch
    loss = neg_log_likelihood.sum() / pred.shape[0]
    
    return loss


def binary_cross_entropy_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Binary Cross-Entropy loss.
    
    BCE = -[target * log(pred) + (1 - target) * log(1 - pred)]
    
    Args:
        pred: Predicted probabilities (after sigmoid) of shape (batch_size,).
        target: Target binary labels of shape (batch_size,).
        
    Returns:
        Scalar tensor containing the binary cross-entropy loss.
    """
    # Ensure target is a tensor
    if not isinstance(target, Tensor):
        target = Tensor(target, requires_grad=False)
    
    # Compute log terms with autograd (log already handles numerical stability)
    log_pred = pred.log()
    
    # Compute log(1 - pred)
    one = Tensor(np.ones_like(pred.data), requires_grad=False)
    one_minus_pred = one - pred
    log_one_minus_pred = one_minus_pred.log()
    
    # Compute BCE
    loss = -(target * log_pred + (one - target) * log_one_minus_pred)
    
    return loss.mean()


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    L1 (Mean Absolute Error) loss.
    
    L1 = (1/n) * Σ|pred_i - target_i|
    
    Args:
        pred: Predicted values tensor.
        target: Target values tensor.
        
    Returns:
        Scalar tensor containing the L1 loss.
    """
    diff = pred - target
    abs_diff = Tensor(
        np.abs(diff.data),
        requires_grad=diff.requires_grad,
        _op='abs',
        _children=(diff,)
    )
    
    # Define backward for abs
    def _abs_backward(grad):
        return (grad * np.sign(diff.data),)
    abs_diff._backward_fn = _abs_backward
    
    return abs_diff.mean()
