"""
Activation functions for neural networks.

All activation functions support automatic differentiation.
"""

import numpy as np
from .tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """
    Rectified Linear Unit (ReLU) activation.
    
    ReLU(x) = max(0, x)
    
    Args:
        x: Input tensor.
        
    Returns:
        Tensor with ReLU applied element-wise.
    """
    out = Tensor(
        np.maximum(0, x.data),
        requires_grad=x.requires_grad,
        _op='relu',
        _children=(x,)
    )
    
    def _backward(grad):
        # Gradient is 1 where x > 0, else 0
        grad_self = grad * (x.data > 0).astype(np.float32)
        return (grad_self,)
    
    out._backward_fn = _backward
    return out


def gelu(x: Tensor) -> Tensor:
    """
    Gaussian Error Linear Unit (GELU) activation.
    
    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution.
    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    Args:
        x: Input tensor.
        
    Returns:
        Tensor with GELU applied element-wise.
    """
    # GELU approximation
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    coeff = 0.044715
    gelu_data = 0.5 * x.data * (1 + np.tanh(sqrt_2_over_pi * (x.data + coeff * x.data ** 3)))
    
    out = Tensor(
        gelu_data,
        requires_grad=x.requires_grad,
        _op='gelu',
        _children=(x,)
    )
    
    def _backward(grad):
        # Gradient of GELU
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        coeff = 0.044715
        x_cubed = x.data ** 3
        inner = sqrt_2_over_pi * (x.data + coeff * x_cubed)
        tanh_inner = np.tanh(inner)
        
        # d/dx GELU(x) = 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * d/dx(...)
        sech_squared = 1 - tanh_inner ** 2
        d_inner = sqrt_2_over_pi * (1 + 3 * coeff * x.data ** 2)
        
        gelu_grad = 0.5 * (1 + tanh_inner) + 0.5 * x.data * sech_squared * d_inner
        grad_self = grad * gelu_grad
        return (grad_self,)
    
    out._backward_fn = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid activation function.
    
    σ(x) = 1 / (1 + exp(-x))
    
    Args:
        x: Input tensor.
        
    Returns:
        Tensor with sigmoid applied element-wise.
    """
    # Clip to avoid overflow
    x_clipped = np.clip(x.data, -500, 500)
    sigmoid_data = 1.0 / (1.0 + np.exp(-x_clipped))
    
    out = Tensor(
        sigmoid_data,
        requires_grad=x.requires_grad,
        _op='sigmoid',
        _children=(x,)
    )
    
    def _backward(grad):
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        sigmoid_grad = out.data * (1 - out.data)
        grad_self = grad * sigmoid_grad
        return (grad_self,)
    
    out._backward_fn = _backward
    return out


def tanh(x: Tensor) -> Tensor:
    """
    Hyperbolic tangent activation.
    
    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Args:
        x: Input tensor.
        
    Returns:
        Tensor with tanh applied element-wise.
    """
    tanh_data = np.tanh(x.data)
    
    out = Tensor(
        tanh_data,
        requires_grad=x.requires_grad,
        _op='tanh',
        _children=(x,)
    )
    
    def _backward(grad):
        # d/dx tanh(x) = 1 - tanh²(x) = sech²(x)
        tanh_grad = 1 - out.data ** 2
        grad_self = grad * tanh_grad
        return (grad_self,)
    
    out._backward_fn = _backward
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Softmax activation function.
    
    softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    
    Args:
        x: Input tensor.
        axis: Axis along which to apply softmax.
        
    Returns:
        Tensor with softmax applied along specified axis.
    """
    # Subtract max for numerical stability
    x_shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    softmax_data = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    out = Tensor(
        softmax_data,
        requires_grad=x.requires_grad,
        _op='softmax',
        _children=(x,)
    )
    
    def _backward(grad):
        # Gradient of softmax is more complex
        # d/dx_i softmax(x_j) = softmax(x_i) * (δ_ij - softmax(x_j))
        # where δ_ij is Kronecker delta
        
        # For each element, gradient depends on all outputs
        grad_self = np.zeros_like(x.data)
        
        # Reshape for broadcasting
        softmax_reshaped = np.expand_dims(out.data, axis=axis)
        grad_reshaped = np.expand_dims(grad, axis=axis)
        
        # Compute Jacobian: J_ij = softmax_i * (δ_ij - softmax_j)
        # For each i: grad_i = Σ_j grad_j * J_ji = Σ_j grad_j * softmax_j * (δ_ji - softmax_i)
        # = softmax_i * (grad_i - Σ_j grad_j * softmax_j)
        
        # Sum of grad * softmax along axis
        grad_softmax_sum = np.sum(grad * out.data, axis=axis, keepdims=True)
        
        # Gradient formula
        grad_self = out.data * (grad - grad_softmax_sum)
        
        return (grad_self,)
    
    out._backward_fn = _backward
    return out


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """
    Leaky ReLU activation.
    
    LeakyReLU(x) = max(αx, x) = x if x > 0 else αx
    
    Args:
        x: Input tensor.
        alpha: Slope for negative values.
        
    Returns:
        Tensor with Leaky ReLU applied element-wise.
    """
    out = Tensor(
        np.where(x.data > 0, x.data, alpha * x.data),
        requires_grad=x.requires_grad,
        _op='leaky_relu',
        _children=(x,)
    )
    
    def _backward(grad):
        # Gradient is 1 where x > 0, else alpha
        grad_self = grad * np.where(x.data > 0, 1.0, alpha)
        return (grad_self,)
    
    out._backward_fn = _backward
    return out
