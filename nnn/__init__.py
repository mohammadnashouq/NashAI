"""
Neural network module with automatic differentiation.

This module provides a complete deep learning framework from scratch,
including:
- Automatic differentiation (autograd)
- Neural network layers (Dense, Conv2D, RNN, LSTM, GRU, Attention)
- Activation functions
- Loss functions
- Optimizers
"""

from .tensor import Tensor, zeros, ones, randn, rand
from .layers import Dense, Sequential
from .activations import relu, gelu, sigmoid, tanh, softmax, leaky_relu
from .losses import mse_loss, cross_entropy_loss, binary_cross_entropy_loss, l1_loss
from .optim import Optimizer, SGD, Adam, RMSprop

# CNN layers
from .conv import Conv2D, MaxPool2D, AvgPool2D, BatchNorm2D, Dropout, Flatten

# RNN layers
from .rnn import RNN, RNNCell, LSTM, LSTMCell, GRU, GRUCell

# Attention layers
from .attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    create_causal_mask,
    create_padding_mask
)

__all__ = [
    # Tensor and autograd
    'Tensor',
    'zeros',
    'ones',
    'randn',
    'rand',
    # Dense layers
    'Dense',
    'Sequential',
    # CNN layers
    'Conv2D',
    'MaxPool2D',
    'AvgPool2D',
    'BatchNorm2D',
    'Dropout',
    'Flatten',
    # RNN layers
    'RNN',
    'RNNCell',
    'LSTM',
    'LSTMCell',
    'GRU',
    'GRUCell',
    # Attention
    'scaled_dot_product_attention',
    'MultiHeadAttention',
    'create_causal_mask',
    'create_padding_mask',
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
