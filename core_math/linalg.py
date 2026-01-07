"""
Linear algebra operations and utilities.

This module provides implementations of fundamental linear algebra operations
including vector and matrix operations, decompositions, and transformations.
"""

from typing import Union, List, Tuple, NamedTuple
import numpy as np


class EigenDecomposition(NamedTuple):
    """Container for eigenvalue decomposition results."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


class Vector:
    """
    A vector class for linear algebra operations.
    
    Supports both 1D vectors and 2D matrices, with operations for
    addition, multiplication, dot product, and outer product.
    """
    
    def __init__(self, data: Union[List, np.ndarray]):
        """
        Initialize a Vector from a list or numpy array.
        
        Args:
            data: Input data as a list or numpy array. Can be 1D (vector) or 2D (matrix).
        """
        if isinstance(data, list):
            self.data = np.array(data, dtype=float)
        else:
            self.data = np.array(data, dtype=float)
        
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        
        # Ensure we have at least 1D
        if self.ndim == 0:
            raise ValueError("Input must be at least 1-dimensional")
    
    def __repr__(self) -> str:
        """String representation of the Vector."""
        return f"Vector(shape={self.shape})\n{self.data}"
    
    def __str__(self) -> str:
        """String representation of the Vector."""
        return str(self.data)
    
    def add(self, other: 'Vector') -> 'Vector':
        """
        Element-wise addition of two vectors or matrices.
        
        Args:
            other: Another Vector to add to this one.
            
        Returns:
            A new Vector containing the element-wise sum.
            
        Raises:
            ValueError: If shapes are incompatible for addition.
        """
        if self.shape != other.shape:
            raise ValueError(f"Shapes {self.shape} and {other.shape} are incompatible for addition")
        
        return Vector(self.data + other.data)
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """Support for + operator."""
        return self.add(other)
    
    def multiply(self, other: 'Vector') -> 'Vector':
        """
        Element-wise multiplication (Hadamard product) of two vectors or matrices.
        
        Args:
            other: Another Vector to multiply with this one.
            
        Returns:
            A new Vector containing the element-wise product.
            
        Raises:
            ValueError: If shapes are incompatible for element-wise multiplication.
        """
        if self.shape != other.shape:
            raise ValueError(f"Shapes {self.shape} and {other.shape} are incompatible for element-wise multiplication")
        
        return Vector(self.data * other.data)
    
    def __mul__(self, other: Union['Vector', float, int]) -> 'Vector':
        """
        Support for * operator.
        
        If other is a Vector, performs element-wise multiplication.
        If other is a scalar, performs scalar multiplication.
        """
        if isinstance(other, (int, float)):
            return Vector(self.data * other)
        return self.multiply(other)
    
    def dot(self, other: 'Vector') -> Union['Vector', float]:
        """
        Dot product (for vectors) or matrix multiplication (for matrices).
        
        For 1D vectors: returns a scalar (dot product).
        For 2D matrices: returns a matrix (matrix multiplication).
        
        Args:
            other: Another Vector to compute the dot product with.
            
        Returns:
            A scalar for vector dot product, or a Vector for matrix multiplication.
            
        Raises:
            ValueError: If shapes are incompatible for dot product.
        """
        # Vector dot product (1D)
        if self.ndim == 1 and other.ndim == 1:
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Vector lengths {self.shape[0]} and {other.shape[0]} are incompatible for dot product")
            return float(np.dot(self.data, other.data))
        
        # Matrix multiplication (2D)
        elif self.ndim == 2 and other.ndim == 2:
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    f"Matrix shapes {self.shape} and {other.shape} are incompatible for matrix multiplication. "
                    f"Number of columns of first matrix ({self.shape[1]}) must equal number of rows of second matrix ({other.shape[0]})"
                )
            return Vector(np.dot(self.data, other.data))
        
        # Vector-matrix or matrix-vector multiplication
        elif self.ndim == 1 and other.ndim == 2:
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Vector length {self.shape[0]} must equal matrix rows {other.shape[0]}")
            return Vector(np.dot(self.data, other.data))
        
        elif self.ndim == 2 and other.ndim == 1:
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Matrix columns {self.shape[1]} must equal vector length {other.shape[0]}")
            return Vector(np.dot(self.data, other.data))
        
        else:
            raise ValueError(f"Dot product not supported for shapes {self.shape} and {other.shape}")
    
    def external_multiply(self, other: 'Vector') -> 'Vector':
        """
        Outer product (external product) of two vectors.
        
        For vectors v (m,) and w (n,), returns a matrix (m, n).
        Also known as the tensor product or dyadic product.
        
        Args:
            other: Another Vector to compute the outer product with.
            
        Returns:
            A new Vector (matrix) containing the outer product.
            
        Raises:
            ValueError: If either vector is not 1D.
        """
        if self.ndim != 1 or other.ndim != 1:
            raise ValueError("Outer product is only defined for 1D vectors")
        
        return Vector(np.outer(self.data, other.data))
    
    def norm(self, p: Union[int, float, str] = 2) -> float:
        """
        Compute the p-norm of a vector or matrix.
        
        For vectors:
        - p=1: L1 norm (Manhattan norm) - sum of absolute values
        - p=2: L2 norm (Euclidean norm) - default
        - p='inf' or 'infinity': Infinity norm - maximum absolute value
        - p='fro': Frobenius norm (same as L2 for vectors)
        
        For matrices:
        - p='fro': Frobenius norm (default for matrices)
        - p=1: Maximum absolute column sum
        - p='inf' or 'infinity': Maximum absolute row sum
        
        Args:
            p: Order of the norm. Can be an integer, float, 'inf', 'infinity', or 'fro'.
            
        Returns:
            The computed norm as a float.
        """
        if self.ndim == 1:
            # Vector norms
            if p == 1:
                return float(np.linalg.norm(self.data, ord=1))
            elif p == 2:
                return float(np.linalg.norm(self.data, ord=2))
            elif p in ('inf', 'infinity'):
                return float(np.linalg.norm(self.data, ord=np.inf))
            elif p == 'fro':
                # For vectors, Frobenius norm is equivalent to L2 norm
                return float(np.linalg.norm(self.data, ord=2))
            else:
                # General p-norm
                return float(np.linalg.norm(self.data, ord=p))
        else:
            # Matrix norms
            if p == 'fro':
                return float(np.linalg.norm(self.data, ord='fro'))
            elif p == 1:
                return float(np.linalg.norm(self.data, ord=1))
            elif p in ('inf', 'infinity'):
                return float(np.linalg.norm(self.data, ord=np.inf))
            else:
                return float(np.linalg.norm(self.data, ord=p))
    
    def norm_l1(self) -> float:
        """
        Compute the L1 norm (Manhattan norm).
        
        For vectors: sum of absolute values.
        For matrices: maximum absolute column sum.
        
        Returns:
            The L1 norm as a float.
        """
        return self.norm(p=1)
    
    def norm_l2(self) -> float:
        """
        Compute the L2 norm (Euclidean norm).
        
        For vectors: sqrt(sum of squares).
        For matrices: same as Frobenius norm.
        
        Returns:
            The L2 norm as a float.
        """
        return self.norm(p=2)
    
    def norm_frobenius(self) -> float:
        """
        Compute the Frobenius norm.
        
        For vectors: same as L2 norm.
        For matrices: sqrt(sum of squares of all elements).
        
        Returns:
            The Frobenius norm as a float.
        """
        return self.norm(p='fro')
    
    def norm_inf(self) -> float:
        """
        Compute the infinity norm (maximum norm).
        
        For vectors: maximum absolute value.
        For matrices: maximum absolute row sum.
        
        Returns:
            The infinity norm as a float.
        """
        return self.norm(p='inf')
    
    def eig(self) -> EigenDecomposition:
        """
        Compute the eigenvalue decomposition of a square matrix.
        
        For a matrix A, finds eigenvalues λ and eigenvectors v such that:
        A @ v = λ * v
        
        Returns:
            EigenDecomposition named tuple containing:
            - eigenvalues: 1D array of eigenvalues
            - eigenvectors: 2D array where each column is an eigenvector
            
        Raises:
            ValueError: If the matrix is not square.
        """
        if self.ndim != 2:
            raise ValueError("Eigenvalue decomposition is only defined for 2D matrices")
        
        if self.shape[0] != self.shape[1]:
            raise ValueError(f"Matrix must be square for eigendecomposition. Got shape {self.shape}")
        
        eigenvalues, eigenvectors = np.linalg.eig(self.data)
        
        return EigenDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors
        )
    
    def eigvals(self) -> np.ndarray:
        """
        Compute only the eigenvalues of a square matrix.
        
        More efficient than eig() if you only need eigenvalues.
        
        Returns:
            1D array of eigenvalues.
            
        Raises:
            ValueError: If the matrix is not square.
        """
        if self.ndim != 2:
            raise ValueError("Eigenvalue computation is only defined for 2D matrices")
        
        if self.shape[0] != self.shape[1]:
            raise ValueError(f"Matrix must be square for eigendecomposition. Got shape {self.shape}")
        
        return np.linalg.eigvals(self.data)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data.copy()
    
    @classmethod
    def zeros(cls, shape: tuple) -> 'Vector':
        """Create a Vector filled with zeros."""
        return cls(np.zeros(shape))
    
    @classmethod
    def ones(cls, shape: tuple) -> 'Vector':
        """Create a Vector filled with ones."""
        return cls(np.ones(shape))
    
    @classmethod
    def eye(cls, n: int) -> 'Vector':
        """Create an n x n identity matrix."""
        return cls(np.eye(n))
    
    @classmethod
    def random(cls, shape: tuple, seed: int = None) -> 'Vector':
        """Create a Vector with random values between 0 and 1."""
        if seed is not None:
            np.random.seed(seed)
        return cls(np.random.rand(*shape))


# Alias for convenience
Matrix = Vector
