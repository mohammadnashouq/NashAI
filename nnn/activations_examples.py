"""
Examples demonstrating activation functions.

This file shows how to use various activation functions and their properties.
"""

import numpy as np
from nnn import Tensor
from nnn.activations import relu, gelu, sigmoid, tanh, softmax, leaky_relu


def example_relu():
    """Example: ReLU activation function."""
    print("=" * 60)
    print("ReLU Activation Function")
    print("=" * 60)
    
    # Create input
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    
    # Apply ReLU
    y = relu(x)
    
    print(f"Input:  {x.data}")
    print(f"Output: {y.data}")
    print(f"ReLU(x) = max(0, x)")
    
    # Test gradient
    y.sum().backward()
    print(f"\nGradient: {x.grad}")
    print("Note: Gradient is 1 where x > 0, else 0")
    print()


def example_gelu():
    """Example: GELU activation function."""
    print("=" * 60)
    print("GELU Activation Function")
    print("=" * 60)
    
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = gelu(x)
    
    print(f"Input:  {x.data}")
    print(f"Output: {y.data}")
    print(f"\nGELU is smooth and non-zero for negative values")
    
    # Test gradient
    y.sum().backward()
    print(f"Gradient: {x.grad}")
    print()


def example_sigmoid():
    """Example: Sigmoid activation function."""
    print("=" * 60)
    print("Sigmoid Activation Function")
    print("=" * 60)
    
    x = Tensor([-3.0, -1.0, 0.0, 1.0, 3.0], requires_grad=True)
    y = sigmoid(x)
    
    print(f"Input:  {x.data}")
    print(f"Output: {y.data}")
    print(f"\nSigmoid maps values to (0, 1) range")
    print(f"Output range: [{y.data.min():.4f}, {y.data.max():.4f}]")
    
    # Test gradient
    y.sum().backward()
    print(f"\nGradient: {x.grad}")
    print("Note: Gradient = sigmoid(x) * (1 - sigmoid(x))")
    print()


def example_tanh():
    """Example: Tanh activation function."""
    print("=" * 60)
    print("Tanh Activation Function")
    print("=" * 60)
    
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = tanh(x)
    
    print(f"Input:  {x.data}")
    print(f"Output: {y.data}")
    print(f"\nTanh maps values to (-1, 1) range")
    print(f"Output range: [{y.data.min():.4f}, {y.data.max():.4f}]")
    
    # Test gradient
    y.sum().backward()
    print(f"\nGradient: {x.grad}")
    print("Note: Gradient = 1 - tanh²(x) = sech²(x)")
    print()


def example_softmax():
    """Example: Softmax activation function."""
    print("=" * 60)
    print("Softmax Activation Function")
    print("=" * 60)
    
    # Create logits (before softmax)
    logits = Tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], requires_grad=True)
    
    # Apply softmax
    probs = softmax(logits, axis=-1)
    
    print("Logits (input):")
    print(logits.data)
    print("\nProbabilities (softmax output):")
    print(probs.data)
    print("\nProperties:")
    print(f"  Sum per row: {probs.data.sum(axis=1)}")
    print(f"  All values in [0, 1]: {np.all((probs.data >= 0) & (probs.data <= 1))}")
    print(f"  Rows sum to 1: {np.allclose(probs.data.sum(axis=1), 1.0)}")
    
    # Test gradient
    probs.sum().backward()
    print(f"\nGradient shape: {logits.grad.shape}")
    print("Gradient (first sample):")
    print(logits.grad[0])
    print()


def example_leaky_relu():
    """Example: Leaky ReLU activation function."""
    print("=" * 60)
    print("Leaky ReLU Activation Function")
    print("=" * 60)
    
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    
    # Different alpha values
    for alpha in [0.01, 0.1, 0.5]:
        y = leaky_relu(x, alpha=alpha)
        print(f"\nAlpha = {alpha}:")
        print(f"  Input:  {x.data}")
        print(f"  Output: {y.data}")
    
    # Test gradient
    y = leaky_relu(x, alpha=0.01)
    y.sum().backward()
    print(f"\nGradient (alpha=0.01): {x.grad}")
    print("Note: Gradient is 1 where x > 0, else alpha")
    print()


def example_activation_comparison():
    """Example: Compare different activations."""
    print("=" * 60)
    print("Activation Function Comparison")
    print("=" * 60)
    
    x = Tensor(np.linspace(-3, 3, 11), requires_grad=True)
    
    activations = {
        'ReLU': relu,
        'Leaky ReLU (α=0.01)': lambda t: leaky_relu(t, 0.01),
        'Sigmoid': sigmoid,
        'Tanh': tanh,
        'GELU': gelu
    }
    
    print("Input values:")
    print(x.data)
    print("\nOutput values:")
    for name, activation_fn in activations.items():
        y = activation_fn(x)
        print(f"\n{name}:")
        print(f"  Output: {y.data}")
        print(f"  Range: [{y.data.min():.4f}, {y.data.max():.4f}]")
    print()


def example_activation_in_network():
    """Example: Using activations in a network."""
    print("=" * 60)
    print("Activations in Neural Network")
    print("=" * 60)
    
    from nnn import Dense, Sequential
    
    # Network with different activations
    model = Sequential(
        Dense(10, 20, activation=relu),
        Dense(20, 15, activation=gelu),
        Dense(15, 10, activation=tanh),
        Dense(10, 5, activation=sigmoid),
        Dense(5, 1, activation=None)
    )
    
    # Forward pass
    input_tensor = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=False)
    output = model(input_tensor)
    
    print("Model with different activations:")
    print(model)
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.data.min():.4f}, {output.data.max():.4f}]")
    print()


def example_softmax_classification():
    """Example: Softmax for multi-class classification."""
    print("=" * 60)
    print("Softmax for Classification")
    print("=" * 60)
    
    # Simulate logits from a classifier (3 classes, 5 samples)
    logits = Tensor(np.random.randn(5, 3).astype(np.float32), requires_grad=True)
    
    print("Logits (before softmax):")
    print(logits.data)
    
    # Apply softmax
    probs = softmax(logits, axis=-1)
    
    print("\nProbabilities (after softmax):")
    print(probs.data)
    
    # Get predictions
    predictions = np.argmax(probs.data, axis=-1)
    print(f"\nPredicted classes: {predictions}")
    
    # Show confidence
    max_probs = probs.data.max(axis=1)
    print(f"Confidence (max probability): {max_probs}")
    print()


def example_activation_gradients():
    """Example: Gradient flow through activations."""
    print("=" * 60)
    print("Gradient Flow Through Activations")
    print("=" * 60)
    
    x = Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
    
    activations_to_test = [
        ('ReLU', relu),
        ('Sigmoid', sigmoid),
        ('Tanh', tanh),
        ('GELU', gelu)
    ]
    
    print("Input:", x.data)
    print("\nGradients after backward:")
    
    for name, activation_fn in activations_to_test:
        x.zero_grad()
        y = activation_fn(x)
        loss = y.sum()
        loss.backward()
        print(f"{name:10s}: {x.grad}")
    print()


def example_activation_properties():
    """Example: Mathematical properties of activations."""
    print("=" * 60)
    print("Activation Function Properties")
    print("=" * 60)
    
    x = Tensor([0.0], requires_grad=True)
    
    properties = {
        'ReLU': {
            'fn': relu,
            'zero_point': 0.0,
            'range': '[0, ∞)',
            'smooth': False
        },
        'Sigmoid': {
            'fn': sigmoid,
            'zero_point': 0.5,
            'range': '(0, 1)',
            'smooth': True
        },
        'Tanh': {
            'fn': tanh,
            'zero_point': 0.0,
            'range': '(-1, 1)',
            'smooth': True
        },
        'GELU': {
            'fn': gelu,
            'zero_point': 0.0,
            'range': '(-∞, ∞)',
            'smooth': True
        }
    }
    
    for name, props in properties.items():
        y = props['fn'](x)
        print(f"\n{name}:")
        print(f"  f(0) = {y.data.item():.4f}")
        print(f"  Range: {props['range']}")
        print(f"  Smooth: {props['smooth']}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ACTIVATIONS EXAMPLES - Activation Functions")
    print("=" * 60 + "\n")
    
    example_relu()
    example_gelu()
    example_sigmoid()
    example_tanh()
    example_softmax()
    example_leaky_relu()
    example_activation_comparison()
    example_activation_in_network()
    example_softmax_classification()
    example_activation_gradients()
    example_activation_properties()
    
    print("=" * 60)
    print("All activations examples completed!")
    print("=" * 60)
