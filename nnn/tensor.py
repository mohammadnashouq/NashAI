"""
Automatic differentiation engine (Autograd).

This module implements a computational graph for automatic differentiation,
enabling automatic gradient computation through backpropagation.
"""

import numpy as np
from typing import Optional, List, Callable, Union
from collections import defaultdict


class Tensor:
    """
    A tensor with automatic differentiation support.
    
    This class implements a computational graph that tracks operations
    and enables automatic gradient computation via backpropagation.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, list, float, int],
        requires_grad: bool = False,
        _op: Optional[str] = None,
        _children: Optional[tuple] = None,
        _backward_fn: Optional[Callable] = None
    ):
        """
        Initialize a Tensor.
        
        Args:
            data: The tensor data (numpy array, list, or scalar).
            requires_grad: Whether to track gradients for this tensor.
            _op: Operation that created this tensor (internal use).
            _children: Child tensors in computational graph (internal use).
            _backward_fn: Function to compute gradients (internal use).
        """
        # Convert input to numpy array
        if isinstance(data, (int, float)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        else:
            self.data = np.asarray(data, dtype=np.float32)
        
        # Ensure at least 1D
        if self.data.ndim == 0:
            self.data = self.data.reshape(1)
        
        self.requires_grad = requires_grad
        self.grad = None
        
        # Computational graph
        self._op = _op
        self._children = _children or ()
        self._backward_fn = _backward_fn
        
        # Gradient accumulation
        self._grad_accumulated = False
    
    @property
    def shape(self) -> tuple:
        """Get the shape of the tensor."""
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return self.data.ndim
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
    
    def __str__(self) -> str:
        """String representation."""
        return f"Tensor({self.data})"
    
    def backward(self, grad: Optional[np.ndarray] = None):
        """
        Backpropagate gradients through the computational graph.
        
        Args:
            grad: Gradient from parent node. If None, assumes this is the output
                  and initializes with ones.
        """
        if not self.requires_grad:
            return
        
        # Initialize gradient if this is the output node
        if grad is None:
            grad = np.ones_like(self.data)
        
        # Accumulate gradients (for nodes used multiple times)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        
        self.grad += grad
        
        # Backpropagate to children
        if self._backward_fn is not None:
            child_grads = self._backward_fn(grad)
            
            if self._children:
                if isinstance(child_grads, tuple):
                    for child, child_grad in zip(self._children, child_grads):
                        # Skip if child_grad is None (gradient handled manually by layer)
                        if child is not None and child.requires_grad and child_grad is not None:
                            child.backward(child_grad)
                else:
                    # Single child
                    if self._children[0] is not None and self._children[0].requires_grad and child_grads is not None:
                        self._children[0].backward(child_grads)
    
    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad = None
        for child in self._children:
            if isinstance(child, Tensor):
                child.zero_grad()
    
    # Arithmetic operations
    def __add__(self, other):
        """Addition: self + other"""
        other = _ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op='add',
            _children=(self, other)
        )
        
        def _backward(grad):
            # Gradient flows to both operands
            grad_self = grad
            grad_other = grad
            
            # Handle broadcasting
            if self.shape != grad.shape:
                grad_self = _sum_to_shape(grad, self.shape)
            if other.shape != grad.shape:
                grad_other = _sum_to_shape(grad, other.shape)
            
            return grad_self, grad_other
        
        out._backward_fn = _backward
        return out
    
    def __radd__(self, other):
        """Right addition: other + self"""
        return self.__add__(other)
    
    def __sub__(self, other):
        """Subtraction: self - other"""
        other = _ensure_tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op='sub',
            _children=(self, other)
        )
        
        def _backward(grad):
            grad_self = grad
            grad_other = -grad
            
            if self.shape != grad.shape:
                grad_self = _sum_to_shape(grad, self.shape)
            if other.shape != grad.shape:
                grad_other = _sum_to_shape(-grad, other.shape)
            
            return grad_self, grad_other
        
        out._backward_fn = _backward
        return out
    
    def __rsub__(self, other):
        """Right subtraction: other - self"""
        return _ensure_tensor(other).__sub__(self)
    
    def __mul__(self, other):
        """Element-wise multiplication: self * other"""
        other = _ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op='mul',
            _children=(self, other)
        )
        
        def _backward(grad):
            # d/dx (x * y) = y
            # d/dy (x * y) = x
            grad_self = grad * other.data
            grad_other = grad * self.data
            
            if self.shape != grad.shape:
                grad_self = _sum_to_shape(grad_self, self.shape)
            if other.shape != grad.shape:
                grad_other = _sum_to_shape(grad_other, other.shape)
            
            return grad_self, grad_other
        
        out._backward_fn = _backward
        return out
    
    def __rmul__(self, other):
        """Right multiplication: other * self"""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Division: self / other"""
        other = _ensure_tensor(other)
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        """Right division: other / self"""
        return _ensure_tensor(other) / self
    
    def __pow__(self, power):
        """Power: self ** power"""
        power = _ensure_tensor(power)
        out = Tensor(
            self.data ** power.data,
            requires_grad=self.requires_grad,
            _op='pow',
            _children=(self, power)
        )
        
        def _backward(grad):
            # d/dx (x^n) = n * x^(n-1)
            if self.requires_grad:
                grad_self = grad * power.data * (self.data ** (power.data - 1))
                if self.shape != grad.shape:
                    grad_self = _sum_to_shape(grad_self, self.shape)
            else:
                grad_self = None
            
            # d/dn (x^n) = x^n * log(x) (usually not needed)
            if power.requires_grad:
                grad_power = grad * out.data * np.log(self.data + 1e-10)
                if power.shape != grad.shape:
                    grad_power = _sum_to_shape(grad_power, power.shape)
            else:
                grad_power = None
            
            return grad_self, grad_power
        
        out._backward_fn = _backward
        return out
    
    def __neg__(self):
        """Negation: -self"""
        return self * -1
    
    def __matmul__(self, other):
        """Matrix multiplication: self @ other"""
        other = _ensure_tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op='matmul',
            _children=(self, other)
        )
        
        def _backward(grad):
            # d/dA (A @ B) = grad @ B^T
            # d/dB (A @ B) = A^T @ grad
            if self.requires_grad:
                grad_self = grad @ other.data.T
            else:
                grad_self = None
            
            if other.requires_grad:
                grad_other = self.data.T @ grad
            else:
                grad_other = None
            
            return grad_self, grad_other
        
        out._backward_fn = _backward
        return out
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False):
        """Sum over axis."""
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _op='sum',
            _children=(self,)
        )
        
        def _backward(grad):
            # Gradient of sum is broadcasted ones
            if axis is None:
                # Sum over all dimensions
                grad_self = np.ones_like(self.data) * grad
            else:
                # Sum over specific axis
                grad_self = np.ones_like(self.data) * np.expand_dims(grad, axis=axis)
            
            return (grad_self,)
        
        out._backward_fn = _backward
        return out
    
    def mean(self, axis: Optional[int] = None, keepdims: bool = False):
        """Mean over axis."""
        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / n
    
    def reshape(self, *shape):
        """Reshape tensor."""
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _op='reshape',
            _children=(self,)
        )
        
        def _backward(grad):
            return (grad.reshape(self.shape),)
        
        out._backward_fn = _backward
        return out
    
    def transpose(self, axes: Optional[tuple] = None):
        """Transpose tensor."""
        out = Tensor(
            self.data.transpose(axes),
            requires_grad=self.requires_grad,
            _op='transpose',
            _children=(self,)
        )
        
        def _backward(grad):
            if axes is None:
                return (grad.transpose(),)
            else:
                # Reverse the transpose
                inv_axes = tuple(np.argsort(axes))
                return (grad.transpose(inv_axes),)
        
        out._backward_fn = _backward
        return out
    
    @property
    def T(self):
        """Transpose (2D matrix)."""
        return self.transpose()
    
    def log(self):
        """Natural logarithm."""
        out = Tensor(
            np.log(self.data + 1e-15),  # Add small epsilon to avoid log(0)
            requires_grad=self.requires_grad,
            _op='log',
            _children=(self,)
        )
        
        def _backward(grad):
            # d/dx log(x) = 1/x
            return (grad / (self.data + 1e-15),)
        
        out._backward_fn = _backward
        return out
    
    def exp(self):
        """Exponential function."""
        out = Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            _op='exp',
            _children=(self,)
        )
        
        def _backward(grad):
            # d/dx exp(x) = exp(x)
            return (grad * out.data,)
        
        out._backward_fn = _backward
        return out


def _ensure_tensor(x):
    """Convert input to Tensor if not already."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=False)


def _sum_to_shape(grad, target_shape):
    """Sum gradient to match target shape (for broadcasting)."""
    # Remove dimensions of size 1
    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0)
    
    # Sum over dimensions where target has size 1
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if g_dim != t_dim and t_dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad


# Convenience functions
def zeros(shape, requires_grad=False):
    """Create tensor of zeros."""
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def ones(shape, requires_grad=False):
    """Create tensor of ones."""
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def randn(*shape, requires_grad=False):
    """Create tensor with random values from standard normal distribution."""
    return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)


def rand(*shape, requires_grad=False):
    """Create tensor with random values from uniform [0, 1) distribution."""
    return Tensor(np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad)
