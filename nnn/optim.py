"""
Optimizers for neural network training.

This module implements various optimization algorithms for updating
neural network parameters during training.
"""

import numpy as np
from typing import List, Optional
from .tensor import Tensor


class Optimizer:
    """Base class for optimizers."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        """
        Initialize optimizer.
        
        Args:
            parameters: List of tensors to optimize.
            lr: Learning rate.
        """
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """Perform one optimization step."""
        raise NotImplementedError
    
    def zero_grad(self):
        """Reset gradients to zero."""
        for param in self.parameters:
            param.zero_grad()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Update rule: θ_{t+1} = θ_t - lr * ∇θ_t
    """
    
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        """
        Initialize SGD optimizer.
        
        Args:
            parameters: List of tensors to optimize.
            lr: Learning rate.
            momentum: Momentum factor (0 = no momentum).
            weight_decay: L2 regularization coefficient.
        """
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize velocity for momentum
        if momentum > 0:
            self.velocity = [np.zeros_like(p.data) for p in parameters]
        else:
            self.velocity = None
    
    def step(self):
        """Perform one SGD step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Ensure we're working with numpy arrays
            # Handle both Tensor.grad (numpy array) and layer.grad (Tensor)
            if isinstance(param.grad, Tensor):
                grad = np.asarray(param.grad.data)
            else:
                grad = np.asarray(param.grad)
            param_data = np.asarray(param.data)
            
            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param_data
            
            # Apply momentum if enabled
            if self.momentum > 0:
                # Ensure velocity is numpy array
                if self.velocity[i] is not None:
                    self.velocity[i] = np.asarray(self.velocity[i])
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grad
                update = self.velocity[i]
            else:
                update = -self.lr * grad
            
            # Update parameters
            param.data = param_data + update


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Combines advantages of AdaGrad and RMSProp.
    Update rule uses exponential moving averages of gradients and squared gradients.
    """
    
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            parameters: List of tensors to optimize.
            lr: Learning rate.
            betas: Coefficients for computing running averages (beta1, beta2).
            eps: Small constant for numerical stability.
            weight_decay: L2 regularization coefficient.
        """
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates as empty lists (lazy initialization)
        self.m = []  # First moment
        self.v = []  # Second moment
        self.t = 0  # Time step
    
    def step(self):
        """Perform one Adam step."""
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            # Ensure we're working with numpy arrays
            # Handle both Tensor.grad (numpy array) and layer.grad (Tensor)
            if isinstance(param.grad, Tensor):
                grad = np.asarray(param.grad.data)
            else:
                grad = np.asarray(param.grad)
            param_data = np.asarray(param.data)
            
            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param_data
            
            # Lazy initialization of moment estimates for new parameters
            while len(self.m) <= i:
                self.m.append(None)
            while len(self.v) <= i:
                self.v.append(None)
            
            if self.m[i] is None:
                self.m[i] = np.zeros_like(param_data)
            if self.v[i] is None:
                self.v[i] = np.zeros_like(param_data)
            
            # Ensure state arrays are numpy arrays
            self.m[i] = np.asarray(self.m[i])
            self.v[i] = np.asarray(self.v[i])
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data = param_data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Maintains a moving average of squared gradients for adaptive learning rates.
    """
    
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize RMSprop optimizer.
        
        Args:
            parameters: List of tensors to optimize.
            lr: Learning rate.
            alpha: Smoothing constant for moving average.
            eps: Small constant for numerical stability.
            weight_decay: L2 regularization coefficient.
        """
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize squared gradient moving average (lazy initialization)
        self.sq_grad_avg = []
    
    def step(self):
        """Perform one RMSprop step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Ensure we're working with numpy arrays
            # Handle both Tensor.grad (numpy array) and layer.grad (Tensor)
            if isinstance(param.grad, Tensor):
                grad = np.asarray(param.grad.data)
            else:
                grad = np.asarray(param.grad)
            param_data = np.asarray(param.data)
            
            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param_data
            
            # Lazy initialization of squared gradient average for new parameters
            while len(self.sq_grad_avg) <= i:
                self.sq_grad_avg.append(None)
            
            if self.sq_grad_avg[i] is None:
                self.sq_grad_avg[i] = np.zeros_like(param_data)
            
            # Ensure state array is numpy array
            self.sq_grad_avg[i] = np.asarray(self.sq_grad_avg[i])
            
            # Update squared gradient moving average
            self.sq_grad_avg[i] = self.alpha * self.sq_grad_avg[i] + (1 - self.alpha) * (grad ** 2)
            
            # Update parameters
            param.data = param_data - self.lr * grad / (np.sqrt(self.sq_grad_avg[i]) + self.eps)
