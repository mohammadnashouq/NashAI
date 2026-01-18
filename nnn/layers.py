"""
Neural network layers.

This module implements various layers for building neural networks.
"""

import numpy as np
from typing import Optional, List
from .tensor import Tensor, randn, zeros


class Dense:
    """
    Dense (fully connected) layer.
    
    Implements: output = activation(input @ W + b)
    where W is the weight matrix and b is the bias vector.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        activation: Optional[callable] = None
    ):
        """
        Initialize a Dense layer.
        
        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            use_bias: Whether to use bias term.
            activation: Activation function to apply (optional).
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.activation = activation
        
        # Initialize weights using Xavier/Glorot initialization
        # W ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32),
            requires_grad=True
        )
        
        # Initialize bias to zeros
        if use_bias:
            self.bias = Tensor(
                np.zeros((out_features,), dtype=np.float32),
                requires_grad=True
            )
        else:
            self.bias = None
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features).
            
        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        # Linear transformation: x @ W + b
        out = x @ self.weight
        
        if self.use_bias:
            # Add bias (broadcasting)
            out = out + self.bias
        
        # Apply activation if specified
        if self.activation is not None:
            out = self.activation(out)
        
        return out
    
    def parameters(self) -> List[Tensor]:
        """Get all trainable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
    
    def zero_grad(self):
        """Reset gradients to zero."""
        self.weight.zero_grad()
        if self.bias is not None:
            self.bias.zero_grad()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Dense(in_features={self.in_features}, out_features={self.out_features}, use_bias={self.use_bias})"


class Sequential:
    """
    Sequential container for stacking layers.
    
    Allows building networks by stacking layers sequentially.
    """
    
    def __init__(self, *layers):
        """
        Initialize Sequential container.
        
        Args:
            *layers: Variable number of layers to stack.
        """
        self.layers = list(layers)
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after passing through all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> List[Tensor]:
        """Get all trainable parameters from all layers."""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        """Reset gradients for all layers."""
        for layer in self.layers:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()
    
    def __repr__(self) -> str:
        """String representation."""
        layer_strs = [str(layer) for layer in self.layers]
        return f"Sequential(\n  " + ",\n  ".join(layer_strs) + "\n)"
