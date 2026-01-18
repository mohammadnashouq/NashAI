"""
Examples demonstrating loss functions.

This file shows how to use various loss functions for different tasks.
"""

import numpy as np
from nnn import Tensor
from nnn.losses import mse_loss, cross_entropy_loss, binary_cross_entropy_loss, l1_loss


def example_mse_loss():
    """Example: Mean Squared Error loss."""
    print("=" * 60)
    print("Mean Squared Error (MSE) Loss")
    print("=" * 60)
    
    # Regression example
    predictions = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    targets = Tensor([[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]], requires_grad=False)
    
    loss = mse_loss(predictions, targets)
    
    print("Predictions:")
    print(predictions.data)
    print("\nTargets:")
    print(targets.data)
    print(f"\nMSE Loss: {loss.data.item():.4f}")
    
    # Manual calculation
    diff = predictions.data - targets.data
    manual_mse = np.mean(diff ** 2)
    print(f"Manual calculation: {manual_mse:.4f}")
    
    # Test gradient
    loss.backward()
    print(f"\nGradient shape: {predictions.grad.shape}")
    print("Gradient:")
    print(predictions.grad)
    print()


def example_mse_perfect_prediction():
    """Example: MSE with perfect predictions."""
    print("=" * 60)
    print("MSE Loss - Perfect Prediction")
    print("=" * 60)
    
    predictions = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    targets = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    
    loss = mse_loss(predictions, targets)
    print(f"MSE Loss (perfect match): {loss.data.item():.6f}")
    print("Should be 0.0")
    print()


def example_cross_entropy_loss():
    """Example: Cross-Entropy loss for multi-class classification."""
    print("=" * 60)
    print("Cross-Entropy Loss (Multi-class)")
    print("=" * 60)
    
    # Logits: 3 samples, 4 classes
    logits = Tensor([
        [2.0, 1.0, 0.1, 0.1],  # Class 0 is most likely
        [0.1, 2.0, 1.0, 0.1],  # Class 1 is most likely
        [0.1, 0.1, 0.1, 2.0]   # Class 3 is most likely
    ], requires_grad=True)
    
    # Targets as class indices
    targets = Tensor([0, 1, 3], requires_grad=False)
    
    loss = cross_entropy_loss(logits, targets)
    
    print("Logits (before softmax):")
    print(logits.data)
    print(f"\nTarget classes: {targets.data}")
    print(f"Cross-Entropy Loss: {loss.data.item():.4f}")
    
    # Test gradient
    loss.backward()
    print(f"\nGradient shape: {logits.grad.shape}")
    print("Gradient:")
    print(logits.grad)
    print()


def example_cross_entropy_one_hot():
    """Example: Cross-Entropy with one-hot encoded targets."""
    print("=" * 60)
    print("Cross-Entropy Loss - One-Hot Encoding")
    print("=" * 60)
    
    logits = Tensor([
        [2.0, 1.0, 0.1],
        [0.1, 2.0, 1.0]
    ], requires_grad=True)
    
    # One-hot encoded targets
    targets_one_hot = Tensor([
        [1.0, 0.0, 0.0],  # Class 0
        [0.0, 1.0, 0.0]   # Class 1
    ], requires_grad=False)
    
    loss = cross_entropy_loss(logits, targets_one_hot)
    
    print("Logits:")
    print(logits.data)
    print("\nTargets (one-hot):")
    print(targets_one_hot.data)
    print(f"\nCross-Entropy Loss: {loss.data.item():.4f}")
    print()


def example_binary_cross_entropy():
    """Example: Binary Cross-Entropy loss."""
    print("=" * 60)
    print("Binary Cross-Entropy Loss")
    print("=" * 60)
    
    from nnn.activations import sigmoid
    
    # Logits (before sigmoid)
    logits = Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
    
    # Apply sigmoid to get probabilities
    probs = sigmoid(logits)
    
    # Binary targets
    targets = Tensor([0.0, 0.0, 1.0, 1.0], requires_grad=False)
    
    loss = binary_cross_entropy_loss(probs, targets)
    
    print("Logits:")
    print(logits.data)
    print("\nProbabilities (after sigmoid):")
    print(probs.data)
    print(f"\nTargets: {targets.data}")
    print(f"Binary Cross-Entropy Loss: {loss.data.item():.4f}")
    
    # Test gradient
    loss.backward()
    print(f"\nGradient:")
    print(logits.grad)
    print()


def example_l1_loss():
    """Example: L1 (Mean Absolute Error) loss."""
    print("=" * 60)
    print("L1 Loss (Mean Absolute Error)")
    print("=" * 60)
    
    predictions = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    targets = Tensor([[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]], requires_grad=False)
    
    loss = l1_loss(predictions, targets)
    
    print("Predictions:")
    print(predictions.data)
    print("\nTargets:")
    print(targets.data)
    print(f"\nL1 Loss: {loss.data.item():.4f}")
    
    # Manual calculation
    manual_l1 = np.mean(np.abs(predictions.data - targets.data))
    print(f"Manual calculation: {manual_l1:.4f}")
    
    # Test gradient
    loss.backward()
    print(f"\nGradient:")
    print(predictions.grad)
    print()


def example_loss_comparison():
    """Example: Compare different loss functions."""
    print("=" * 60)
    print("Loss Function Comparison")
    print("=" * 60)
    
    predictions = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    targets = Tensor([[1.5, 2.5], [2.5, 3.5]], requires_grad=False)
    
    losses = {
        'MSE': mse_loss(predictions, targets),
        'L1': l1_loss(predictions, targets)
    }
    
    print("Predictions:")
    print(predictions.data)
    print("\nTargets:")
    print(targets.data)
    print("\nLoss values:")
    for name, loss in losses.items():
        print(f"  {name}: {loss.data.item():.4f}")
    print()


def example_classification_losses():
    """Example: Classification loss functions."""
    print("=" * 60)
    print("Classification Loss Functions")
    print("=" * 60)
    
    # Multi-class example
    logits_multi = Tensor([
        [2.0, 1.0, 0.1],
        [0.1, 2.0, 1.0]
    ], requires_grad=True)
    targets_multi = Tensor([0, 1], requires_grad=False)
    
    ce_loss = cross_entropy_loss(logits_multi, targets_multi)
    print("Multi-class Cross-Entropy:")
    print(f"  Loss: {ce_loss.data.item():.4f}")
    
    # Binary example
    from nnn.activations import sigmoid
    logits_binary = Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
    probs_binary = sigmoid(logits_binary)
    targets_binary = Tensor([0.0, 0.0, 1.0, 1.0], requires_grad=False)
    
    bce_loss = binary_cross_entropy_loss(probs_binary, targets_binary)
    print("\nBinary Cross-Entropy:")
    print(f"  Loss: {bce_loss.data.item():.4f}")
    print()


def example_loss_gradients():
    """Example: Gradient computation for different losses."""
    print("=" * 60)
    print("Loss Function Gradients")
    print("=" * 60)
    
    # Regression losses
    pred_reg = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target_reg = Tensor([1.5, 2.5, 3.5], requires_grad=False)
    
    mse = mse_loss(pred_reg, target_reg)
    mse.backward()
    print("MSE Loss:")
    print(f"  Loss: {mse.data.item():.4f}")
    print(f"  Gradient: {pred_reg.grad}")
    
    pred_reg.zero_grad()
    l1 = l1_loss(pred_reg, target_reg)
    l1.backward()
    print("\nL1 Loss:")
    print(f"  Loss: {l1.data.item():.4f}")
    print(f"  Gradient: {pred_reg.grad}")
    print()


def example_loss_in_training():
    """Example: Using losses in a training scenario."""
    print("=" * 60)
    print("Loss Functions in Training")
    print("=" * 60)
    
    from nnn import Dense, Sequential
    from nnn.activations import relu
    
    # Simple model
    model = Sequential(
        Dense(5, 10, activation=relu),
        Dense(10, 1, activation=None)
    )
    
    # Generate data
    X = Tensor(np.random.randn(8, 5).astype(np.float32), requires_grad=False)
    y = Tensor(np.random.randn(8, 1).astype(np.float32), requires_grad=False)
    
    # Forward pass
    predictions = model(X)
    
    # Compute loss
    loss = mse_loss(predictions, y)
    
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Prediction shape: {predictions.shape}")
    print(f"\nLoss: {loss.data.item():.4f}")
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Check if gradients are computed
    params = model.parameters()
    print(f"\nGradients computed: {all(p.grad is not None for p in params)}")
    print()


def example_loss_properties():
    """Example: Properties of different loss functions."""
    print("=" * 60)
    print("Loss Function Properties")
    print("=" * 60)
    
    # Perfect prediction
    perfect_pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    perfect_target = Tensor([1.0, 2.0, 3.0], requires_grad=False)
    
    mse_perfect = mse_loss(perfect_pred, perfect_target)
    l1_perfect = l1_loss(perfect_pred, perfect_target)
    
    print("Perfect predictions:")
    print(f"  MSE: {mse_perfect.data.item():.6f} (should be 0)")
    print(f"  L1:  {l1_perfect.data.item():.6f} (should be 0)")
    
    # Large error
    large_pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    large_target = Tensor([10.0, 20.0, 30.0], requires_grad=False)
    
    mse_large = mse_loss(large_pred, large_target)
    l1_large = l1_loss(large_pred, large_target)
    
    print("\nLarge errors:")
    print(f"  MSE: {mse_large.data.item():.4f}")
    print(f"  L1:  {l1_large.data.item():.4f}")
    print("Note: MSE penalizes large errors more (quadratic vs linear)")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LOSSES EXAMPLES - Loss Functions")
    print("=" * 60 + "\n")
    
    example_mse_loss()
    example_mse_perfect_prediction()
    example_cross_entropy_loss()
    example_cross_entropy_one_hot()
    example_binary_cross_entropy()
    example_l1_loss()
    example_loss_comparison()
    example_classification_losses()
    example_loss_gradients()
    example_loss_in_training()
    example_loss_properties()
    
    print("=" * 60)
    print("All losses examples completed!")
    print("=" * 60)
