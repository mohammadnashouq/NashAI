"""
Quick test to verify CNN gradient flow is working correctly.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# NashAI imports
from nnn.tensor import Tensor
from nnn.layers import Dense
from nnn.conv import Conv2D, MaxPool2D, Flatten
from nnn.activations import relu
from nnn.losses import cross_entropy_loss
from nnn.optim import Adam


def one_hot_encode(y, num_classes=10):
    """Convert labels to one-hot encoding."""
    one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot


class SimpleCNN:
    """Simple CNN for testing."""
    
    def __init__(self, num_classes=10):
        self.conv1 = Conv2D(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.flatten = Flatten()
        # After conv and pool: 28 -> 14, with 8 channels: 8 * 14 * 14 = 1568
        self.fc1 = Dense(8 * 14 * 14, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x
    
    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.fc1.parameters())
        return params
    
    def zero_grad(self):
        self.conv1.zero_grad()
        self.fc1.zero_grad()


def test_gradient_flow():
    """Test that gradients flow correctly through the CNN."""
    print("Testing CNN gradient flow...")
    
    # Create small random data
    np.random.seed(42)
    batch_size = 4
    X = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
    y = np.random.randint(0, 10, batch_size)
    y_oh = one_hot_encode(y)
    
    # Create model
    model = SimpleCNN()
    optimizer = Adam(model.parameters(), lr=0.01)
    
    print(f"\nInitial parameter values:")
    print(f"  conv1.weight mean: {model.conv1.weight.data.mean():.6f}")
    print(f"  fc1.weight mean:   {model.fc1.weight.data.mean():.6f}")
    
    # Store initial weights
    conv1_weight_init = model.conv1.weight.data.copy()
    fc1_weight_init = model.fc1.weight.data.copy()
    
    # Forward pass
    X_tensor = Tensor(X, requires_grad=True)
    y_tensor = Tensor(y_oh)
    
    logits = model.forward(X_tensor)
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits mean:  {logits.data.mean():.6f}")
    
    loss = cross_entropy_loss(logits, y_tensor)
    print(f"Initial loss: {loss.data.item():.6f}")
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Check gradients
    print(f"\nGradient check:")
    
    if model.conv1.weight.grad is not None:
        from nnn.tensor import Tensor as T
        if isinstance(model.conv1.weight.grad, T):
            grad_data = model.conv1.weight.grad.data
        else:
            grad_data = model.conv1.weight.grad
        print(f"  conv1.weight.grad exists: True")
        print(f"  conv1.weight.grad mean: {grad_data.mean():.6f}")
        print(f"  conv1.weight.grad norm: {np.linalg.norm(grad_data):.6f}")
    else:
        print(f"  conv1.weight.grad exists: False  <-- PROBLEM!")
    
    if model.fc1.weight.grad is not None:
        from nnn.tensor import Tensor as T
        if isinstance(model.fc1.weight.grad, T):
            grad_data = model.fc1.weight.grad.data
        else:
            grad_data = model.fc1.weight.grad
        print(f"  fc1.weight.grad exists:   True")
        print(f"  fc1.weight.grad mean:   {grad_data.mean():.6f}")
        print(f"  fc1.weight.grad norm:   {np.linalg.norm(grad_data):.6f}")
    else:
        print(f"  fc1.weight.grad exists:   False  <-- PROBLEM!")
    
    # Optimizer step
    optimizer.step()
    
    # Check that weights have changed
    conv1_weight_diff = np.abs(model.conv1.weight.data - conv1_weight_init).mean()
    fc1_weight_diff = np.abs(model.fc1.weight.data - fc1_weight_init).mean()
    
    print(f"\nWeight changes after optimizer step:")
    print(f"  conv1.weight change: {conv1_weight_diff:.6f}")
    print(f"  fc1.weight change:   {fc1_weight_diff:.6f}")
    
    if conv1_weight_diff > 0:
        print(f"  conv1.weight updated: True")
    else:
        print(f"  conv1.weight updated: False  <-- PROBLEM!")
    
    if fc1_weight_diff > 0:
        print(f"  fc1.weight updated:   True")
    else:
        print(f"  fc1.weight updated:   False  <-- PROBLEM!")
    
    # Training loop test
    print(f"\n" + "="*50)
    print("Testing training loop (5 iterations)...")
    print("="*50)
    
    losses = []
    for i in range(5):
        X_tensor = Tensor(X, requires_grad=True)
        y_tensor = Tensor(y_oh)
        
        logits = model.forward(X_tensor)
        loss = cross_entropy_loss(logits, y_tensor)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.data.item())
        print(f"  Iteration {i+1}: Loss = {loss.data.item():.6f}")
    
    # Check if loss decreased
    if losses[-1] < losses[0]:
        print(f"\n[PASS] Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
        print("       CNN is learning correctly!")
        return True
    else:
        print(f"\n[FAIL] Loss did NOT decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")
        print("       There might still be an issue with gradient flow.")
        return False


if __name__ == '__main__':
    success = test_gradient_flow()
    print(f"\n{'='*50}")
    print(f"Test {'PASSED' if success else 'FAILED'}")
    print(f"{'='*50}")
