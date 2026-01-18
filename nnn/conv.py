"""
Convolutional neural network layers.

This module implements CNN components including:
- Conv2D: 2D Convolution with im2col optimization
- MaxPool2D / AvgPool2D: Pooling layers
- BatchNorm2D: Batch normalization
- Dropout: Regularization
- Flatten: Reshape for fully connected layers

All layers use NCHW format: (batch, channels, height, width)
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from .tensor import Tensor


def _im2col(input_data: np.ndarray, kernel_h: int, kernel_w: int,
            stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Transform input patches to columns for efficient convolution.
    
    Args:
        input_data: Input array of shape (N, C, H, W)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: Stride for convolution
        padding: Zero padding
        
    Returns:
        Column matrix of shape (N * out_h * out_w, C * kernel_h * kernel_w)
    """
    N, C, H, W = input_data.shape
    
    # Add padding
    if padding > 0:
        input_padded = np.pad(
            input_data,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        input_padded = input_data
    
    H_padded, W_padded = input_padded.shape[2], input_padded.shape[3]
    
    # Output dimensions
    out_h = (H_padded - kernel_h) // stride + 1
    out_w = (W_padded - kernel_w) // stride + 1
    
    # Create column matrix
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=input_data.dtype)
    
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = input_padded[:, :, y:y_max:stride, x:x_max:stride]
    
    # Reshape: (N, C, kH, kW, out_h, out_w) -> (N * out_h * out_w, C * kH * kW)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    
    return col


def _col2im(col: np.ndarray, input_shape: Tuple[int, int, int, int],
            kernel_h: int, kernel_w: int, stride: int = 1, 
            padding: int = 0) -> np.ndarray:
    """
    Transform columns back to image (inverse of im2col).
    
    Args:
        col: Column matrix of shape (N * out_h * out_w, C * kernel_h * kernel_w)
        input_shape: Original input shape (N, C, H, W)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: Stride for convolution
        padding: Zero padding
        
    Returns:
        Image array of shape (N, C, H, W)
    """
    N, C, H, W = input_shape
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    
    out_h = (H_padded - kernel_h) // stride + 1
    out_w = (W_padded - kernel_w) // stride + 1
    
    # Reshape col: (N * out_h * out_w, C * kH * kW) -> (N, out_h, out_w, C, kH, kW)
    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    
    # Create output with padding
    img = np.zeros((N, C, H_padded, W_padded), dtype=col.dtype)
    
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    # Remove padding
    if padding > 0:
        return img[:, :, padding:-padding, padding:-padding]
    return img


class Conv2D:
    """
    2D Convolutional Layer.
    
    Applies a 2D convolution over an input signal composed of several input planes.
    Uses im2col for efficient computation.
    
    Input shape: (N, C_in, H, W)
    Output shape: (N, C_out, H_out, W_out)
    
    where:
        H_out = (H + 2*padding - kernel_size) // stride + 1
        W_out = (W + 2*padding - kernel_size) // stride + 1
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        use_bias: bool = True
    ):
        """
        Initialize Conv2D layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (filters)
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides
            use_bias: Whether to add a learnable bias
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        
        # Xavier/Glorot initialization
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = out_channels * self.kernel_size[0] * self.kernel_size[1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        # Weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        self.weight = Tensor(
            np.random.uniform(-limit, limit, 
                (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
            ).astype(np.float32),
            requires_grad=True
        )
        
        if use_bias:
            self.bias = Tensor(
                np.zeros(out_channels, dtype=np.float32),
                requires_grad=True
            )
        else:
            self.bias = None
        
        # Cache for backward pass
        self._cache = {}
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, C_in, H, W)
            
        Returns:
            Output tensor of shape (N, C_out, H_out, W_out)
        """
        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        
        # Output dimensions
        out_h = (H + 2 * self.padding - kH) // self.stride + 1
        out_w = (W + 2 * self.padding - kW) // self.stride + 1
        
        # im2col transformation
        x_col = _im2col(x.data, kH, kW, self.stride, self.padding)
        
        # Reshape weights: (out_channels, in_channels, kH, kW) -> (out_channels, in_channels * kH * kW)
        w_col = self.weight.data.reshape(self.out_channels, -1)
        
        # Convolution as matrix multiplication
        out_data = x_col @ w_col.T  # (N * out_h * out_w, out_channels)
        
        # Reshape to output shape
        out_data = out_data.reshape(N, out_h, out_w, self.out_channels)
        out_data = out_data.transpose(0, 3, 1, 2)  # (N, out_channels, out_h, out_w)
        
        # Add bias
        if self.use_bias:
            out_data = out_data + self.bias.data.reshape(1, -1, 1, 1)
        
        # Cache for backward
        self._cache['x'] = x
        self._cache['x_col'] = x_col
        self._cache['x_shape'] = x.shape
        
        # Create output tensor
        out = Tensor(
            out_data,
            requires_grad=x.requires_grad or self.weight.requires_grad,
            _op='conv2d',
            _children=(x, self.weight, self.bias) if self.use_bias else (x, self.weight)
        )
        
        def _backward(grad):
            N, C_out, out_h, out_w = grad.shape
            kH, kW = self.kernel_size
            
            # Gradient w.r.t. bias
            if self.use_bias and self.bias.requires_grad:
                grad_bias = np.sum(grad, axis=(0, 2, 3))
                if self.bias.grad is None:
                    self.bias.grad = Tensor(np.zeros_like(self.bias.data))
                self.bias.grad.data += grad_bias
            
            # Reshape gradient: (N, C_out, out_h, out_w) -> (N * out_h * out_w, C_out)
            grad_reshaped = grad.transpose(0, 2, 3, 1).reshape(-1, C_out)
            
            # Gradient w.r.t. weights
            if self.weight.requires_grad:
                x_col = self._cache['x_col']
                grad_w = grad_reshaped.T @ x_col  # (C_out, C_in * kH * kW)
                grad_w = grad_w.reshape(self.weight.shape)
                if self.weight.grad is None:
                    self.weight.grad = Tensor(np.zeros_like(self.weight.data))
                self.weight.grad.data += grad_w
            
            # Gradient w.r.t. input
            grad_x = None
            if x.requires_grad:
                w_col = self.weight.data.reshape(self.out_channels, -1)
                grad_col = grad_reshaped @ w_col  # (N * out_h * out_w, C_in * kH * kW)
                grad_x = _col2im(grad_col, self._cache['x_shape'], kH, kW, 
                                self.stride, self.padding)
            
            if self.use_bias:
                return (grad_x, None, None)
            return (grad_x, None)
        
        out._backward_fn = _backward
        return out
    
    def parameters(self) -> List[Tensor]:
        """Get trainable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
    
    def zero_grad(self):
        """Reset gradients."""
        self.weight.grad = None
        if self.bias is not None:
            self.bias.grad = None
    
    def __repr__(self) -> str:
        return (f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")


class MaxPool2D:
    """
    2D Max Pooling Layer.
    
    Applies max pooling over an input signal composed of several input planes.
    
    Input shape: (N, C, H, W)
    Output shape: (N, C, H_out, W_out)
    """
    
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[int] = None,
        padding: int = 0
    ):
        """
        Initialize MaxPool2D layer.
        
        Args:
            kernel_size: Size of the pooling window
            stride: Stride of the pooling (defaults to kernel_size)
            padding: Zero-padding added to both sides
        """
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        self.stride = stride if stride is not None else self.kernel_size[0]
        self.padding = padding
        self._cache = {}
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass."""
        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        
        # Add padding if needed
        if self.padding > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
                constant_values=-np.inf  # For max pooling
            )
        else:
            x_padded = x.data
        
        H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]
        
        # Output dimensions
        out_h = (H_padded - kH) // self.stride + 1
        out_w = (W_padded - kW) // self.stride + 1
        
        # Reshape for pooling
        out_data = np.zeros((N, C, out_h, out_w), dtype=x.data.dtype)
        max_indices = np.zeros((N, C, out_h, out_w, 2), dtype=np.int32)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + kH
                w_start = j * self.stride
                w_end = w_start + kW
                
                window = x_padded[:, :, h_start:h_end, w_start:w_end]
                
                # Reshape to (N, C, kH*kW) for argmax
                window_flat = window.reshape(N, C, -1)
                max_idx_flat = np.argmax(window_flat, axis=2)
                
                # Convert flat index to 2D indices
                max_h = max_idx_flat // kW
                max_w = max_idx_flat % kW
                
                out_data[:, :, i, j] = np.max(window_flat, axis=2)
                max_indices[:, :, i, j, 0] = h_start + max_h
                max_indices[:, :, i, j, 1] = w_start + max_w
        
        # Cache for backward
        self._cache['x_shape'] = x.shape
        self._cache['max_indices'] = max_indices
        self._cache['x_padded_shape'] = x_padded.shape
        
        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _op='maxpool2d',
            _children=(x,)
        )
        
        def _backward(grad):
            if not x.requires_grad:
                return (None,)
            
            N, C, H, W = self._cache['x_shape']
            _, _, H_pad, W_pad = self._cache['x_padded_shape']
            max_indices = self._cache['max_indices']
            
            grad_x_padded = np.zeros((N, C, H_pad, W_pad), dtype=grad.dtype)
            
            out_h, out_w = grad.shape[2], grad.shape[3]
            for i in range(out_h):
                for j in range(out_w):
                    for n in range(N):
                        for c in range(C):
                            h_idx = max_indices[n, c, i, j, 0]
                            w_idx = max_indices[n, c, i, j, 1]
                            grad_x_padded[n, c, h_idx, w_idx] += grad[n, c, i, j]
            
            # Remove padding
            if self.padding > 0:
                grad_x = grad_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
            else:
                grad_x = grad_x_padded
            
            return (grad_x,)
        
        out._backward_fn = _backward
        return out
    
    def parameters(self) -> List[Tensor]:
        """No trainable parameters."""
        return []
    
    def zero_grad(self):
        """No gradients to reset."""
        pass
    
    def __repr__(self) -> str:
        return f"MaxPool2D(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AvgPool2D:
    """
    2D Average Pooling Layer.
    
    Applies average pooling over an input signal composed of several input planes.
    
    Input shape: (N, C, H, W)
    Output shape: (N, C, H_out, W_out)
    """
    
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[int] = None,
        padding: int = 0
    ):
        """
        Initialize AvgPool2D layer.
        
        Args:
            kernel_size: Size of the pooling window
            stride: Stride of the pooling (defaults to kernel_size)
            padding: Zero-padding added to both sides
        """
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        self.stride = stride if stride is not None else self.kernel_size[0]
        self.padding = padding
        self._cache = {}
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass."""
        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        
        # Add padding if needed
        if self.padding > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            x_padded = x.data
        
        H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]
        
        # Output dimensions
        out_h = (H_padded - kH) // self.stride + 1
        out_w = (W_padded - kW) // self.stride + 1
        
        # Compute average pooling
        out_data = np.zeros((N, C, out_h, out_w), dtype=x.data.dtype)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + kH
                w_start = j * self.stride
                w_end = w_start + kW
                
                window = x_padded[:, :, h_start:h_end, w_start:w_end]
                out_data[:, :, i, j] = np.mean(window, axis=(2, 3))
        
        # Cache for backward
        self._cache['x_shape'] = x.shape
        self._cache['x_padded_shape'] = x_padded.shape
        
        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _op='avgpool2d',
            _children=(x,)
        )
        
        def _backward(grad):
            if not x.requires_grad:
                return (None,)
            
            N, C, H, W = self._cache['x_shape']
            _, _, H_pad, W_pad = self._cache['x_padded_shape']
            kH, kW = self.kernel_size
            pool_size = kH * kW
            
            grad_x_padded = np.zeros((N, C, H_pad, W_pad), dtype=grad.dtype)
            
            out_h, out_w = grad.shape[2], grad.shape[3]
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + kH
                    w_start = j * self.stride
                    w_end = w_start + kW
                    
                    # Distribute gradient equally
                    grad_x_padded[:, :, h_start:h_end, w_start:w_end] += \
                        grad[:, :, i:i+1, j:j+1] / pool_size
            
            # Remove padding
            if self.padding > 0:
                grad_x = grad_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
            else:
                grad_x = grad_x_padded
            
            return (grad_x,)
        
        out._backward_fn = _backward
        return out
    
    def parameters(self) -> List[Tensor]:
        """No trainable parameters."""
        return []
    
    def zero_grad(self):
        """No gradients to reset."""
        pass
    
    def __repr__(self) -> str:
        return f"AvgPool2D(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class BatchNorm2D:
    """
    2D Batch Normalization.
    
    Normalizes the input over the batch dimension for each channel.
    Uses running statistics for evaluation mode.
    
    Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
    
    Input shape: (N, C, H, W)
    Output shape: (N, C, H, W)
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1
    ):
        """
        Initialize BatchNorm2D layer.
        
        Args:
            num_features: Number of channels (C)
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics update
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        
        # Running statistics (not trained)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        
        # Training mode
        self.training = True
        self._cache = {}
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass."""
        N, C, H, W = x.shape
        
        if self.training:
            # Compute batch statistics
            # Mean and var over (N, H, W) for each channel
            mean = np.mean(x.data, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
            var = np.var(x.data, axis=(0, 2, 3), keepdims=True)    # (1, C, 1, 1)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * var.squeeze()
        else:
            # Use running statistics
            mean = self.running_mean.reshape(1, C, 1, 1)
            var = self.running_var.reshape(1, C, 1, 1)
        
        # Normalize
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        gamma = self.gamma.data.reshape(1, C, 1, 1)
        beta = self.beta.data.reshape(1, C, 1, 1)
        out_data = gamma * x_norm + beta
        
        # Cache for backward
        self._cache['x'] = x
        self._cache['x_norm'] = x_norm
        self._cache['mean'] = mean
        self._cache['var'] = var
        self._cache['std'] = np.sqrt(var + self.eps)
        
        out = Tensor(
            out_data,
            requires_grad=x.requires_grad or self.gamma.requires_grad,
            _op='batchnorm2d',
            _children=(x, self.gamma, self.beta)
        )
        
        def _backward(grad):
            N, C, H, W = grad.shape
            x_norm = self._cache['x_norm']
            std = self._cache['std']
            
            # Gradient w.r.t. gamma and beta
            if self.gamma.requires_grad:
                grad_gamma = np.sum(grad * x_norm, axis=(0, 2, 3))
                if self.gamma.grad is None:
                    self.gamma.grad = Tensor(np.zeros_like(self.gamma.data))
                self.gamma.grad.data += grad_gamma
            
            if self.beta.requires_grad:
                grad_beta = np.sum(grad, axis=(0, 2, 3))
                if self.beta.grad is None:
                    self.beta.grad = Tensor(np.zeros_like(self.beta.data))
                self.beta.grad.data += grad_beta
            
            # Gradient w.r.t. input
            grad_x = None
            if x.requires_grad:
                gamma = self.gamma.data.reshape(1, C, 1, 1)
                m = N * H * W  # Number of elements per channel
                
                # Backprop through normalization
                grad_x_norm = grad * gamma
                grad_var = np.sum(grad_x_norm * (x.data - self._cache['mean']) * 
                                 -0.5 * (self._cache['var'] + self.eps) ** (-1.5),
                                 axis=(0, 2, 3), keepdims=True)
                grad_mean = np.sum(grad_x_norm * -1 / std, axis=(0, 2, 3), keepdims=True) + \
                           grad_var * np.sum(-2 * (x.data - self._cache['mean']), 
                                            axis=(0, 2, 3), keepdims=True) / m
                
                grad_x = grad_x_norm / std + \
                        grad_var * 2 * (x.data - self._cache['mean']) / m + \
                        grad_mean / m
            
            return (grad_x, None, None)
        
        out._backward_fn = _backward
        return out
    
    def parameters(self) -> List[Tensor]:
        """Get trainable parameters."""
        return [self.gamma, self.beta]
    
    def zero_grad(self):
        """Reset gradients."""
        self.gamma.grad = None
        self.beta.grad = None
    
    def __repr__(self) -> str:
        return f"BatchNorm2D(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"


class Dropout:
    """
    Dropout Layer for regularization.
    
    Randomly zeros elements with probability p during training.
    Scales output by 1/(1-p) to maintain expected value (inverted dropout).
    
    Input shape: Any
    Output shape: Same as input
    """
    
    def __init__(self, p: float = 0.5):
        """
        Initialize Dropout layer.
        
        Args:
            p: Probability of dropping an element (0 to 1)
        """
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.training = True
        self._cache = {}
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if not self.training or self.p == 0:
            return x
        
        # Generate dropout mask
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        
        # Scale by 1/(1-p) for inverted dropout
        scale = 1.0 / (1.0 - self.p)
        out_data = x.data * mask * scale
        
        # Cache for backward
        self._cache['mask'] = mask
        self._cache['scale'] = scale
        
        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _op='dropout',
            _children=(x,)
        )
        
        def _backward(grad):
            if not x.requires_grad:
                return (None,)
            
            mask = self._cache['mask']
            scale = self._cache['scale']
            grad_x = grad * mask * scale
            
            return (grad_x,)
        
        out._backward_fn = _backward
        return out
    
    def parameters(self) -> List[Tensor]:
        """No trainable parameters."""
        return []
    
    def zero_grad(self):
        """No gradients to reset."""
        pass
    
    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class Flatten:
    """
    Flatten Layer.
    
    Flattens input tensor while preserving batch dimension.
    
    Input shape: (N, C, H, W) or any shape
    Output shape: (N, C*H*W) or (N, product of remaining dims)
    """
    
    def __init__(self, start_dim: int = 1):
        """
        Initialize Flatten layer.
        
        Args:
            start_dim: First dimension to flatten (default=1, preserving batch)
        """
        self.start_dim = start_dim
        self._cache = {}
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass."""
        original_shape = x.shape
        
        # Compute new shape
        new_shape = list(original_shape[:self.start_dim])
        new_shape.append(np.prod(original_shape[self.start_dim:]))
        
        out_data = x.data.reshape(new_shape)
        
        # Cache for backward
        self._cache['original_shape'] = original_shape
        
        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _op='flatten',
            _children=(x,)
        )
        
        def _backward(grad):
            if not x.requires_grad:
                return (None,)
            
            grad_x = grad.reshape(self._cache['original_shape'])
            return (grad_x,)
        
        out._backward_fn = _backward
        return out
    
    def parameters(self) -> List[Tensor]:
        """No trainable parameters."""
        return []
    
    def zero_grad(self):
        """No gradients to reset."""
        pass
    
    def __repr__(self) -> str:
        return f"Flatten(start_dim={self.start_dim})"
