"""
Optimization algorithms and utilities.

This module provides implementations of optimization algorithms including
gradient descent, numerical differentiation, and other optimization techniques.
"""

from typing import Callable, Union, List, Tuple, Optional
import numpy as np


def numerical_derivative(
    func: Callable,
    x: Union[float, np.ndarray],
    h: float = 1e-5,
    method: str = 'central'
) -> Union[float, np.ndarray]:
    """
    Compute the numerical derivative of a function at a point using finite differences.
    
    This is implemented from scratch using finite difference approximations.
    
    Args:
        func: The function to differentiate. Should take a scalar or array and return a scalar.
        x: The point(s) at which to compute the derivative.
        h: Step size for finite differences. Default is 1e-5.
        method: Method to use for finite differences:
            - 'forward': Forward difference (f(x+h) - f(x)) / h
            - 'backward': Backward difference (f(x) - f(x-h)) / h
            - 'central': Central difference (f(x+h) - f(x-h)) / (2h) [default, most accurate]
            
    Returns:
        The numerical derivative. Same shape as input x.
        
    Examples:
        >>> def f(x):
        ...     return x**2
        >>> numerical_derivative(f, 2.0)
        4.0  # derivative of x^2 at x=2 is 2x = 4
    """
    x = np.asarray(x, dtype=float)
    is_scalar = x.ndim == 0
    x = np.atleast_1d(x)
    
    if method == 'forward':
        # Forward difference: f'(x) ≈ (f(x+h) - f(x)) / h
        result = (func(x + h) - func(x)) / h
    elif method == 'backward':
        # Backward difference: f'(x) ≈ (f(x) - f(x-h)) / h
        result = (func(x) - func(x - h)) / h
    elif method == 'central':
        # Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        # This is more accurate (O(h^2) error vs O(h))
        result = (func(x + h) - func(x - h)) / (2 * h)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'forward', 'backward', or 'central'")
    
    # Convert result to scalar if input was scalar
    if is_scalar:
        result = np.asarray(result, dtype=float)
        # Extract scalar value from result (handles both scalar and array results)
        return float(result.flat[0])
    return result


def numerical_gradient(
    func: Callable,
    x: np.ndarray,
    h: float = 1e-5
) -> np.ndarray:
    """
    Compute the numerical gradient of a multivariate function.
    
    Computes the gradient vector using central differences for each dimension.
    This is implemented from scratch.
    
    Args:
        func: The function to differentiate. Should take an array and return a scalar.
        x: The point at which to compute the gradient. Should be a 1D array.
        h: Step size for finite differences. Default is 1e-5.
        
    Returns:
        The gradient vector as a 1D numpy array.
        
    Examples:
        >>> def f(x):
        ...     return x[0]**2 + x[1]**2
        >>> numerical_gradient(f, np.array([1.0, 2.0]))
        array([2., 4.])  # gradient is [2x, 2y] = [2, 4]
    """
    x = np.asarray(x, dtype=float).flatten()
    gradient = np.zeros_like(x)
    
    # Compute partial derivative with respect to each dimension
    for i in range(len(x)):
        # Create perturbed vector
        x_forward = x.copy()
        x_forward[i] += h
        
        x_backward = x.copy()
        x_backward[i] -= h
        
        # Central difference for this dimension
        gradient[i] = (func(x_forward) - func(x_backward)) / (2 * h)
    
    return gradient


def numerical_hessian(
    func: Callable,
    x: np.ndarray,
    h: float = 1e-5
) -> np.ndarray:
    """
    Compute the numerical Hessian matrix (second derivatives) of a function.
    
    The Hessian is the matrix of second partial derivatives.
    This is implemented from scratch using finite differences.
    
    Args:
        func: The function to differentiate. Should take an array and return a scalar.
        x: The point at which to compute the Hessian. Should be a 1D array.
        h: Step size for finite differences. Default is 1e-5.
        
    Returns:
        The Hessian matrix as a 2D numpy array.
        
    Examples:
        >>> def f(x):
        ...     return x[0]**2 + x[1]**2
        >>> numerical_hessian(f, np.array([1.0, 2.0]))
        array([[2., 0.],
               [0., 2.]])  # Hessian of x^2 + y^2 is [[2, 0], [0, 2]]
    """
    x = np.asarray(x, dtype=float).flatten()
    n = len(x)
    hessian = np.zeros((n, n))
    
    # Compute second derivatives
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal: second derivative with respect to x[i]
                x_forward = x.copy()
                x_forward[i] += h
                x_backward = x.copy()
                x_backward[i] -= h
                hessian[i, j] = (func(x_forward) - 2 * func(x) + func(x_backward)) / (h ** 2)
            else:
                # Off-diagonal: mixed partial derivative
                x_pp = x.copy()
                x_pp[i] += h
                x_pp[j] += h
                
                x_pm = x.copy()
                x_pm[i] += h
                x_pm[j] -= h
                
                x_mp = x.copy()
                x_mp[i] -= h
                x_mp[j] += h
                
                x_mm = x.copy()
                x_mm[i] -= h
                x_mm[j] -= h
                
                hessian[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h ** 2)
    
    return hessian


class GradientDescent:
    """
    Gradient descent optimizer implemented from scratch.
    
    This is a basic gradient descent optimizer that can be used to minimize
    functions. Supports both analytical gradients and numerical gradients.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        use_numerical_gradient: bool = False,
        h: float = 1e-5
    ):
        """
        Initialize the gradient descent optimizer.
        
        Args:
            learning_rate: Step size for each iteration. Also called 'alpha' or 'eta'.
            max_iterations: Maximum number of iterations to run.
            tolerance: Convergence tolerance. Stops if change in function value is less than this.
            use_numerical_gradient: If True, uses numerical differentiation instead of analytical gradient.
            h: Step size for numerical differentiation (if use_numerical_gradient is True).
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_numerical_gradient = use_numerical_gradient
        self.h = h
        
        # History tracking
        self.history = {
            'iterations': [],
            'values': [],
            'gradients': [],
            'positions': []
        }
    
    def minimize(
        self,
        func: Callable,
        x0: np.ndarray,
        gradient: Optional[Callable] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Minimize a function using gradient descent.
        
        Args:
            func: The objective function to minimize. Should take an array and return a scalar.
            x0: Initial guess for the minimum. Should be a 1D array.
            gradient: Optional gradient function. If None and use_numerical_gradient is False,
                     will use numerical_gradient. Should take an array and return an array.
        
        Returns:
            A tuple containing:
            - x_opt: The optimized parameters (numpy array)
            - info: Dictionary with optimization information:
                - 'converged': Whether the optimizer converged
                - 'iterations': Number of iterations performed
                - 'final_value': Final function value
                - 'history': Full optimization history
                
        Examples:
            >>> def f(x):
            ...     return x[0]**2 + x[1]**2
            >>> def grad_f(x):
            ...     return np.array([2*x[0], 2*x[1]])
            >>> optimizer = GradientDescent(learning_rate=0.1, max_iterations=100)
            >>> x_opt, info = optimizer.minimize(f, np.array([3.0, 4.0]), gradient=grad_f)
            >>> print(x_opt)  # Should be close to [0, 0]
        """
        x = np.asarray(x0, dtype=float).flatten()
        self.history = {
            'iterations': [],
            'values': [],
            'gradients': [],
            'positions': []
        }
        
        prev_value = func(x)
        self.history['iterations'].append(0)
        self.history['values'].append(prev_value)
        self.history['positions'].append(x.copy())
        
        for iteration in range(1, self.max_iterations + 1):
            # Compute gradient
            if gradient is not None:
                grad = np.asarray(gradient(x), dtype=float).flatten()
            elif self.use_numerical_gradient:
                grad = numerical_gradient(func, x, h=self.h)
            else:
                # Default to numerical gradient if no gradient function provided
                grad = numerical_gradient(func, x, h=self.h)
            
            # Gradient descent update: x_new = x_old - learning_rate * gradient
            x = x - self.learning_rate * grad
            
            # Evaluate function at new position
            current_value = func(x)
            
            # Store history
            self.history['iterations'].append(iteration)
            self.history['values'].append(current_value)
            self.history['gradients'].append(grad.copy())
            self.history['positions'].append(x.copy())
            
            # Check for convergence
            if abs(prev_value - current_value) < self.tolerance:
                info = {
                    'converged': True,
                    'iterations': iteration,
                    'final_value': current_value,
                    'history': self.history
                }
                return x, info
            
            prev_value = current_value
        
        # Did not converge within max_iterations
        info = {
            'converged': False,
            'iterations': self.max_iterations,
            'final_value': current_value,
            'history': self.history
        }
        return x, info
    
    def reset(self):
        """Reset the optimizer's history."""
        self.history = {
            'iterations': [],
            'values': [],
            'gradients': [],
            'positions': []
        }


class StochasticGradientDescent(GradientDescent):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    
    A variant of gradient descent that uses a random subset of data points
    (mini-batch) for each iteration, making it more efficient for large datasets.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize the SGD optimizer.
        
        Args:
            learning_rate: Step size for each iteration.
            max_iterations: Maximum number of iterations to run.
            tolerance: Convergence tolerance.
            batch_size: Size of mini-batches to use.
            **kwargs: Additional arguments passed to parent GradientDescent class.
        """
        super().__init__(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            **kwargs
        )
        self.batch_size = batch_size
    
    def minimize_batch(
        self,
        func: Callable,
        x0: np.ndarray,
        data: np.ndarray,
        gradient: Optional[Callable] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Minimize a function using stochastic gradient descent with mini-batches.
        
        Args:
            func: The objective function. Should take (params, batch_data) and return a scalar.
            x0: Initial guess for the minimum.
            data: The full dataset. Will be split into mini-batches.
            gradient: Optional gradient function. Should take (params, batch_data) and return gradient.
        
        Returns:
            A tuple containing optimized parameters and info dictionary.
        """
        x = np.asarray(x0, dtype=float).flatten()
        n_samples = len(data)
        self.history = {
            'iterations': [],
            'values': [],
            'gradients': [],
            'positions': []
        }
        
        prev_value = np.mean([func(x, data[i:i+1]) for i in range(min(10, n_samples))])
        
        for iteration in range(1, self.max_iterations + 1):
            # Randomly sample a mini-batch
            batch_indices = np.random.choice(n_samples, size=min(self.batch_size, n_samples), replace=False)
            batch_data = data[batch_indices]
            
            # Compute gradient on mini-batch
            if gradient is not None:
                grad = np.asarray(gradient(x, batch_data), dtype=float).flatten()
            else:
                # Use numerical gradient on mini-batch
                def batch_func(params):
                    return func(params, batch_data)
                grad = numerical_gradient(batch_func, x, h=self.h)
            
            # SGD update
            x = x - self.learning_rate * grad
            
            # Evaluate on full dataset (for monitoring)
            if iteration % 10 == 0:
                current_value = np.mean([func(x, data[i:i+1]) for i in range(min(10, n_samples))])
                self.history['values'].append(current_value)
                
                if abs(prev_value - current_value) < self.tolerance:
                    info = {
                        'converged': True,
                        'iterations': iteration,
                        'final_value': current_value,
                        'history': self.history
                    }
                    return x, info
                prev_value = current_value
            
            self.history['iterations'].append(iteration)
            self.history['positions'].append(x.copy())
        
        info = {
            'converged': False,
            'iterations': self.max_iterations,
            'final_value': prev_value,
            'history': self.history
        }
        return x, info
