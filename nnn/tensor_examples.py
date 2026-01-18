"""
Examples demonstrating Tensor operations and automatic differentiation.

This file shows how to use the autograd engine for various operations.
"""

import numpy as np
from nnn import Tensor, zeros, ones, randn, rand


def example_basic_operations():
    """Example: Basic tensor operations."""
    print("=" * 60)
    print("Basic Tensor Operations")
    print("=" * 60)
    
    # Create tensors
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    
    print(f"a = {a.data}")
    print(f"b = {b.data}")
    
    # Addition
    c = a + b
    print(f"\na + b = {c.data}")
    
    # Subtraction
    d = a - b
    print(f"a - b = {d.data}")
    
    # Multiplication (element-wise)
    e = a * b
    print(f"a * b = {e.data}")
    
    # Division
    f = a / b
    print(f"a / b = {f.data}")
    
    # Power
    g = a ** 2
    print(f"a^2 = {g.data}")
    
    # Negation
    h = -a
    print(f"-a = {h.data}")
    print()


def example_autograd_simple():
    """Example: Simple automatic differentiation."""
    print("=" * 60)
    print("Simple Autograd Example")
    print("=" * 60)
    
    # Create tensors with gradient tracking
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    
    # Compute: z = x * y + x^2
    z = x * y + x ** 2
    
    # Backpropagate
    z.backward()
    
    print(f"x = {x.data.item():.2f}")
    print(f"y = {y.data.item():.2f}")
    print(f"z = x*y + x^2 = {z.data.item():.2f}")
    print(f"\nGradients:")
    print(f"∂z/∂x = {x.grad.item():.2f} (should be y + 2x = {y.data.item() + 2*x.data.item():.2f})")
    print(f"∂z/∂y = {y.grad.item():.2f} (should be x = {x.data.item():.2f})")
    print()


def example_autograd_chain_rule():
    """Example: Chain rule in action."""
    print("=" * 60)
    print("Chain Rule Example")
    print("=" * 60)
    
    # Compute: f(x) = (x^2 + 1)^3
    # df/dx = 3(x^2 + 1)^2 * 2x = 6x(x^2 + 1)^2
    
    x = Tensor(2.0, requires_grad=True)
    
    # Forward pass
    x_squared = x ** 2
    inner = x_squared + Tensor(1.0, requires_grad=False)
    f = inner ** 3
    
    # Backward pass
    f.backward()
    
    expected_grad = 6 * x.data.item() * (x.data.item()**2 + 1)**2
    
    print(f"x = {x.data.item():.2f}")
    print(f"f(x) = (x² + 1)³ = {f.data.item():.2f}")
    print(f"∂f/∂x = {x.grad.item():.2f}")
    print(f"Expected: {expected_grad:.2f}")
    print()


def example_matrix_operations():
    """Example: Matrix operations."""
    print("=" * 60)
    print("Matrix Operations")
    print("=" * 60)
    
    # Create matrices
    A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    
    print("A =")
    print(A.data)
    print("\nB =")
    print(B.data)
    
    # Matrix multiplication
    C = A @ B
    print("\nA @ B =")
    print(C.data)
    
    # Transpose
    A_T = A.T
    print("\nA^T =")
    print(A_T.data)
    
    # Sum
    sum_A = A.sum()
    print(f"\nSum of A = {sum_A.data.item():.2f}")
    
    # Mean
    mean_A = A.mean()
    print(f"Mean of A = {mean_A.data.item():.2f}")
    print()


def example_matrix_autograd():
    """Example: Automatic differentiation with matrices."""
    print("=" * 60)
    print("Matrix Autograd Example")
    print("=" * 60)
    
    # Compute: loss = sum((A @ B)^2)
    A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    
    # Forward pass
    C = A @ B
    C_squared = C * C
    loss = C_squared.sum()
    
    # Backward pass
    loss.backward()
    
    print("A =")
    print(A.data)
    print("\nB =")
    print(B.data)
    print(f"\nLoss = sum((A @ B)²) = {loss.data.item():.2f}")
    print("\nGradient w.r.t. A:")
    print(A.grad)
    print("\nGradient w.r.t. B:")
    print(B.grad)
    print()


def example_broadcasting():
    """Example: Broadcasting in tensor operations."""
    print("=" * 60)
    print("Broadcasting Example")
    print("=" * 60)
    
    # Vector + scalar
    v = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    s = Tensor(10.0, requires_grad=True)
    
    result = v + s
    print(f"v = {v.data}")
    print(f"s = {s.data.item():.2f}")
    print(f"v + s = {result.data}")
    
    # Backward
    result.sum().backward()
    print(f"\nGradient of v: {v.grad}")
    print(f"Gradient of s: {s.grad.item():.2f}")
    print()


def example_reshape_transpose():
    """Example: Reshape and transpose operations."""
    print("=" * 60)
    print("Reshape and Transpose")
    print("=" * 60)
    
    # Create tensor
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    print("Original shape:", x.shape)
    print("x =")
    print(x.data)
    
    # Reshape
    x_reshaped = x.reshape(3, 2)
    print("\nReshaped to (3, 2):")
    print(x_reshaped.data)
    
    # Transpose
    x_transposed = x.T
    print("\nTransposed:")
    print(x_transposed.data)
    print()


def example_sum_mean():
    """Example: Sum and mean operations with gradients."""
    print("=" * 60)
    print("Sum and Mean Operations")
    print("=" * 60)
    
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    
    # Sum over all elements
    total_sum = x.sum()
    total_sum.backward()
    print("x =")
    print(x.data)
    print(f"\nSum = {total_sum.data.item():.2f}")
    print("Gradient (should be all ones):")
    print(x.grad)
    
    # Reset
    x.zero_grad()
    
    # Mean
    mean_val = x.mean()
    mean_val.backward()
    print(f"\nMean = {mean_val.data.item():.2f}")
    print("Gradient (should be 1/n = 1/6):")
    print(x.grad)
    print()


def example_log_exp():
    """Example: Logarithm and exponential operations."""
    print("=" * 60)
    print("Log and Exp Operations")
    print("=" * 60)
    
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Natural logarithm
    log_x = x.log()
    print(f"x = {x.data}")
    print(f"log(x) = {log_x.data}")
    
    # Exponential
    exp_x = x.exp()
    print(f"exp(x) = {exp_x.data}")
    
    # Combined: log(exp(x)) = x
    log_exp_x = x.exp().log()
    print(f"log(exp(x)) = {log_exp_x.data}")
    
    # Backward
    log_x.sum().backward()
    print(f"\nGradient of log(x): {x.grad}")
    print()


def example_complex_computation():
    """Example: Complex computation graph."""
    print("=" * 60)
    print("Complex Computation Graph")
    print("=" * 60)
    
    # Compute: f(x, y) = log(x^2 + y^2) * exp(x*y)
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    
    # Forward pass
    x_sq = x ** 2
    y_sq = y ** 2
    sum_sq = x_sq + y_sq
    log_term = sum_sq.log()
    
    xy = x * y
    exp_term = xy.exp()
    
    result = log_term * exp_term
    
    # Backward pass
    result.backward()
    
    print(f"x = {x.data.item():.2f}")
    print(f"y = {y.data.item():.2f}")
    print(f"f(x,y) = log(x² + y²) * exp(xy) = {result.data.item():.4f}")
    print(f"\n∂f/∂x = {x.grad.item():.4f}")
    print(f"∂f/∂y = {y.grad.item():.4f}")
    print()


def example_utility_functions():
    """Example: Utility functions for creating tensors."""
    print("=" * 60)
    print("Utility Functions")
    print("=" * 60)
    
    # Zeros
    z = zeros((2, 3))
    print("zeros(2, 3) =")
    print(z.data)
    
    # Ones
    o = ones((2, 3))
    print("\nones(2, 3) =")
    print(o.data)
    
    # Random normal
    rn = randn(2, 3)
    print("\nrandn(2, 3) =")
    print(rn.data)
    
    # Random uniform
    ru = rand(2, 3)
    print("\nrand(2, 3) =")
    print(ru.data)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TENSOR EXAMPLES - Autograd Engine")
    print("=" * 60 + "\n")
    
    example_basic_operations()
    example_autograd_simple()
    example_autograd_chain_rule()
    example_matrix_operations()
    example_matrix_autograd()
    example_broadcasting()
    example_reshape_transpose()
    example_sum_mean()
    example_log_exp()
    example_complex_computation()
    example_utility_functions()
    
    print("=" * 60)
    print("All tensor examples completed!")
    print("=" * 60)
