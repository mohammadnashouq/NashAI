"""
Dimensionality reduction algorithms implemented from scratch.

This module implements Principal Component Analysis (PCA) and
Independent Component Analysis (ICA) without using sklearn.
"""

import numpy as np
from typing import Optional, Tuple


class PCA:
    """
    Principal Component Analysis (PCA) implemented from scratch.
    
    Reduces dimensionality by projecting data onto principal components
    (directions of maximum variance).
    """
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of components to keep. If None, keeps all components.
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X: np.ndarray):
        """
        Fit PCA by computing principal components.
        
        Args:
            X: Training data of shape (n_samples, n_features).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        # Using (1/(n-1)) normalization for sample covariance
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending order)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select number of components
        if self.n_components is None:
            n_components = n_features
        else:
            n_components = min(self.n_components, n_features)
        
        # Store components (transpose so each row is a component)
        self.components_ = eigenvectors[:, :n_components].T
        
        # Store explained variance
        self.explained_variance_ = eigenvalues[:n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component space.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        if self.components_ is None:
            raise ValueError("Model must be fitted before transformation")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Center the data
        X_centered = X - self.mean_
        
        # Project onto principal components
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Args:
            X_transformed: Transformed data of shape (n_samples, n_components).
            
        Returns:
            Data in original space of shape (n_samples, n_features).
        """
        if self.components_ is None:
            raise ValueError("Model must be fitted before inverse transformation")
        
        X_transformed = np.asarray(X_transformed, dtype=float)
        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(-1, 1)
        
        # Project back to original space
        X_reconstructed = X_transformed @ self.components_
        
        # Add back the mean
        return X_reconstructed + self.mean_


class ICA:
    """
    Independent Component Analysis (ICA) implemented from scratch.
    
    Separates mixed signals into independent components using FastICA algorithm.
    """
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        max_iterations: int = 200,
        tolerance: float = 1e-4,
        random_state: Optional[int] = None
    ):
        """
        Initialize ICA.
        
        Args:
            n_components: Number of components to extract. If None, uses n_features.
            max_iterations: Maximum number of iterations for FastICA.
            tolerance: Convergence tolerance.
            random_state: Random seed for initialization.
        """
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.components_ = None
        self.mixing_ = None
        self.mean_ = None
    
    def _g(self, x: np.ndarray) -> np.ndarray:
        """Nonlinearity function: tanh."""
        return np.tanh(x)
    
    def _g_prime(self, x: np.ndarray) -> np.ndarray:
        """Derivative of nonlinearity function."""
        return 1 - np.tanh(x) ** 2
    
    def _whiten(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Whiten the data using PCA.
        
        Returns:
            Tuple of (whitened_data, whitening_matrix).
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Whitening matrix: D^(-1/2) @ E^T
        # where D is diagonal matrix of eigenvalues, E is eigenvectors
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
        whitening_matrix = D_inv_sqrt @ eigenvectors.T
        
        # Whiten the data
        X_whitened = X_centered @ whitening_matrix.T
        
        return X_whitened, whitening_matrix
    
    def _fastica_one_component(
        self,
        X: np.ndarray,
        W: np.ndarray,
        w: np.ndarray
    ) -> np.ndarray:
        """
        FastICA algorithm for one component.
        
        Args:
            X: Whitened data of shape (n_samples, n_features).
            W: Current unmixing matrix (for decorrelation).
            w: Initial weight vector for this component.
            
        Returns:
            Updated weight vector.
        """
        n_samples = X.shape[0]
        
        for iteration in range(self.max_iterations):
            # Compute w^T @ X
            wTx = X @ w
            
            # Update w
            w_new = np.mean(X * self._g(wTx)[:, np.newaxis], axis=0) - \
                   np.mean(self._g_prime(wTx)) * w
            
            # Normalize
            w_new = w_new / np.linalg.norm(w_new)
            
            # Decorrelate with previous components
            if W.shape[0] > 0:
                w_new = w_new - W.T @ (W @ w_new)
                w_new = w_new / np.linalg.norm(w_new)
            
            # Check convergence
            if np.abs(np.abs(w @ w_new) - 1) < self.tolerance:
                break
            
            w = w_new
        
        return w
    
    def fit(self, X: np.ndarray):
        """
        Fit ICA model.
        
        Args:
            X: Training data of shape (n_samples, n_features).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Determine number of components
        if self.n_components is None:
            n_components = n_features
        else:
            n_components = min(self.n_components, n_features)
        
        # Whiten the data
        X_whitened, whitening_matrix = self._whiten(X)
        
        # Initialize unmixing matrix
        W = np.zeros((n_components, n_features))
        
        for i in range(n_components):
            # Random initialization
            w = np.random.randn(n_features)
            w = w / np.linalg.norm(w)
            
            # FastICA for this component
            w = self._fastica_one_component(X_whitened, W[:i], w)
            
            W[i] = w
        
        # Store components (unmixing matrix)
        self.components_ = W
        
        # Compute mixing matrix (pseudo-inverse of unmixing matrix)
        # This is the matrix that mixes the independent components
        self.mixing_ = np.linalg.pinv(W) @ np.linalg.pinv(whitening_matrix)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to independent component space.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Independent components of shape (n_samples, n_components).
        """
        if self.components_ is None:
            raise ValueError("Model must be fitted before transformation")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Center the data
        X_centered = X - self.mean_
        
        # Project onto independent components
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit ICA and transform data.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Independent components of shape (n_samples, n_components).
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform independent components back to original space.
        
        Args:
            X_transformed: Independent components of shape (n_samples, n_components).
            
        Returns:
            Reconstructed data of shape (n_samples, n_features).
        """
        if self.mixing_ is None:
            raise ValueError("Model must be fitted before inverse transformation")
        
        X_transformed = np.asarray(X_transformed, dtype=float)
        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(-1, 1)
        
        # Reconstruct using mixing matrix
        X_reconstructed = X_transformed @ self.mixing_
        
        # Add back the mean
        return X_reconstructed + self.mean_
