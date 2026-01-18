"""
Neural network module with automatic differentiation.

This module provides a complete deep learning framework from scratch,
including:
- Automatic differentiation (autograd)
- Neural network layers
- Activation functions
- Loss functions
- Optimizers
"""

from .tensor import Tensor, zeros, ones, randn, rand
from .layers import Dense, Sequential
from .activations import relu, gelu, sigmoid, tanh, softmax, leaky_relu
from .losses import mse_loss, cross_entropy_loss, binary_cross_entropy_loss, l1_loss
from .optim import Optimizer, SGD, Adam, RMSprop

__all__ = [
    # Tensor and autograd
    'Tensor',
    'zeros',
    'ones',
    'randn',
    'rand',
    # Layers
    'Dense',
    'Sequential',
    # Activations
    'relu',
    'gelu',
    'sigmoid',
    'tanh',
    'softmax',
    'leaky_relu',
    # Losses
    'mse_loss',
    'cross_entropy_loss',
    'binary_cross_entropy_loss',
    'l1_loss',
    # Optimizers
    'Optimizer',
    'SGD',
    'Adam',
    'RMSprop',
]
