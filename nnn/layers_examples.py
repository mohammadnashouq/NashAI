"""
Examples demonstrating neural network layers.

This file shows how to use Dense layers and Sequential containers.
"""

import numpy as np
from nnn import Tensor, Dense, Sequential
from nnn.activations import relu, sigmoid, tanh, gelu, softmax


def example_single_dense_layer():
    """Example: Single dense layer."""
    print("=" * 60)
    print("Single Dense Layer")
    print("=" * 60)
    
    # Create a dense layer: 10 inputs -> 5 outputs
    layer = Dense(in_features=10, out_features=5, use_bias=True, activation=relu)
    
    print(f"Layer: {layer}")
    print(f"Weight shape: {layer.weight.shape}")
    print(f"Bias shape: {layer.bias.shape}")
    print(f"\nWeight matrix:")
    print(layer.weight.data)
    print(f"\nBias vector:")
    print(layer.bias.data)
    
    # Forward pass
    batch_size = 4
    input_data = np.random.randn(batch_size, 10).astype(np.float32)
    input_tensor = Tensor(input_data, requires_grad=False)
    
    output = layer(input_tensor)
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (first sample):")
    print(output.data[0])
    print()


def example_dense_without_bias():
    """Example: Dense layer without bias."""
    print("=" * 60)
    print("Dense Layer Without Bias")
    print("=" * 60)
    
    layer = Dense(in_features=8, out_features=4, use_bias=False, activation=relu)
    
    print(f"Layer: {layer}")
    print(f"Bias: {layer.bias}")  # Should be None
    
    # Forward pass
    input_tensor = Tensor(np.random.randn(3, 8).astype(np.float32), requires_grad=False)
    output = layer(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print()


def example_dense_different_activations():
    """Example: Dense layers with different activation functions."""
    print("=" * 60)
    print("Dense Layers with Different Activations")
    print("=" * 60)
    
    input_data = np.random.randn(2, 5).astype(np.float32)
    input_tensor = Tensor(input_data, requires_grad=False)
    
    activations = {
        'ReLU': relu,
        'Sigmoid': sigmoid,
        'Tanh': tanh,
        'GELU': gelu,
        'None': None
    }
    
    for name, activation in activations.items():
        layer = Dense(5, 3, use_bias=True, activation=activation)
        output = layer(input_tensor)
        print(f"{name:10s}: Output range [{output.data.min():.4f}, {output.data.max():.4f}]")
    print()


def example_sequential_simple():
    """Example: Simple sequential network."""
    print("=" * 60)
    print("Simple Sequential Network")
    print("=" * 60)
    
    # Create a simple network: 10 -> 8 -> 5 -> 1
    model = Sequential(
        Dense(10, 8, activation=relu),
        Dense(8, 5, activation=relu),
        Dense(5, 1, activation=None)
    )
    
    print("Model architecture:")
    print(model)
    
    # Forward pass
    input_tensor = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=False)
    output = model(input_tensor)
    
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print()


def example_sequential_classification():
    """Example: Sequential network for classification."""
    print("=" * 60)
    print("Sequential Network for Classification")
    print("=" * 60)
    
    # Network for 784 (MNIST-like) -> 10 classes
    model = Sequential(
        Dense(784, 256, activation=relu),
        Dense(256, 128, activation=relu),
        Dense(128, 64, activation=relu),
        Dense(64, 10, activation=None)  # Logits, no activation
    )
    
    print("Classification model:")
    print(model)
    
    # Get all parameters
    params = model.parameters()
    print(f"\nTotal parameters: {len(params)}")
    total_params = sum(p.data.size for p in params)
    print(f"Total parameter count: {total_params:,}")
    
    # Forward pass
    input_tensor = Tensor(np.random.randn(32, 784).astype(np.float32), requires_grad=False)
    output = model(input_tensor)
    
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (first sample, logits):")
    print(output.data[0])
    print()


def example_sequential_regression():
    """Example: Sequential network for regression."""
    print("=" * 60)
    print("Sequential Network for Regression")
    print("=" * 60)
    
    # Network for regression: 20 features -> 1 output
    model = Sequential(
        Dense(20, 64, activation=relu),
        Dense(64, 32, activation=relu),
        Dense(32, 16, activation=relu),
        Dense(16, 1, activation=None)  # Single output, no activation
    )
    
    print("Regression model:")
    print(model)
    
    # Forward pass
    input_tensor = Tensor(np.random.randn(16, 20).astype(np.float32), requires_grad=False)
    output = model(input_tensor)
    
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Predictions (first 5 samples):")
    print(output.data[:5].flatten())
    print()


def example_parameter_access():
    """Example: Accessing and inspecting parameters."""
    print("=" * 60)
    print("Parameter Access and Inspection")
    print("=" * 60)
    
    model = Sequential(
        Dense(10, 5, activation=relu),
        Dense(5, 2, activation=None)
    )
    
    params = model.parameters()
    print(f"Total parameter tensors: {len(params)}")
    
    for i, param in enumerate(params):
        print(f"\nParameter {i}:")
        print(f"  Shape: {param.shape}")
        print(f"  Requires grad: {param.requires_grad}")
        print(f"  Data range: [{param.data.min():.4f}, {param.data.max():.4f}]")
        if param.grad is not None:
            print(f"  Gradient range: [{param.grad.min():.4f}, {param.grad.max():.4f}]")
    print()


def example_gradient_flow():
    """Example: Gradient flow through layers."""
    print("=" * 60)
    print("Gradient Flow Through Layers")
    print("=" * 60)
    
    model = Sequential(
        Dense(4, 3, activation=relu),
        Dense(3, 2, activation=None)
    )
    
    # Create input and target
    input_tensor = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=False)
    target = Tensor(np.random.randn(2, 2).astype(np.float32), requires_grad=False)
    
    # Forward pass
    output = model(input_tensor)
    
    # Simple MSE loss
    diff = output - target
    loss = (diff * diff).mean()
    
    print(f"Loss before backward: {loss.data.item():.4f}")
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Check gradients
    params = model.parameters()
    print(f"\nGradients after backward:")
    for i, param in enumerate(params):
        if param.grad is not None:
            print(f"  Parameter {i}: gradient norm = {np.linalg.norm(param.grad):.6f}")
        else:
            print(f"  Parameter {i}: no gradient")
    print()


def example_deep_network():
    """Example: Deep network with many layers."""
    print("=" * 60)
    print("Deep Network Example")
    print("=" * 60)
    
    # Very deep network
    model = Sequential(
        Dense(100, 200, activation=relu),
        Dense(200, 150, activation=relu),
        Dense(150, 100, activation=relu),
        Dense(100, 50, activation=relu),
        Dense(50, 25, activation=relu),
        Dense(25, 10, activation=None)
    )
    
    print("Deep network architecture:")
    print(model)
    
    # Count parameters
    params = model.parameters()
    total_params = sum(p.data.size for p in params)
    print(f"\nTotal parameters: {total_params:,}")
    
    # Forward pass
    input_tensor = Tensor(np.random.randn(8, 100).astype(np.float32), requires_grad=False)
    output = model(input_tensor)
    
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print()


def example_layer_initialization():
    """Example: Inspecting layer initialization."""
    print("=" * 60)
    print("Layer Initialization Inspection")
    print("=" * 60)
    
    # Create multiple layers and inspect their initialization
    layers = [
        Dense(10, 20, activation=None),
        Dense(20, 5, activation=None),
        Dense(5, 1, activation=None)
    ]
    
    for i, layer in enumerate(layers):
        weight = layer.weight.data
        print(f"\nLayer {i+1} (10->20->5->1):")
        print(f"  Weight mean: {weight.mean():.6f}")
        print(f"  Weight std: {weight.std():.6f}")
        print(f"  Weight range: [{weight.min():.4f}, {weight.max():.4f}]")
        
        if layer.bias is not None:
            bias = layer.bias.data
            print(f"  Bias mean: {bias.mean():.6f}")
            print(f"  Bias std: {bias.std():.6f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LAYERS EXAMPLES - Neural Network Layers")
    print("=" * 60 + "\n")
    
    example_single_dense_layer()
    example_dense_without_bias()
    example_dense_different_activations()
    example_sequential_simple()
    example_sequential_classification()
    example_sequential_regression()
    example_parameter_access()
    example_gradient_flow()
    example_deep_network()
    example_layer_initialization()
    
    print("=" * 60)
    print("All layers examples completed!")
    print("=" * 60)
