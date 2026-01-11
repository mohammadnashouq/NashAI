"""
Clustering algorithms implemented from scratch.

This module implements k-Means and Gaussian Mixture Model (GMM)
without using sklearn.
"""

import numpy as np
from typing import Optional, Tuple


class KMeans:
    """
    k-Means Clustering implemented from scratch.
    
    Partitions data into k clusters by minimizing within-cluster sum of squares.
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        max_iterations: int = 300,
        tolerance: float = 1e-4,
        random_state: Optional[int] = None,
        init: str = 'random'
    ):
        """
        Initialize k-Means.
        
        Args:
            n_clusters: Number of clusters to form.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance (change in centroids).
            random_state: Random seed for initialization.
            init: Initialization method ('random' or 'kmeans++').
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.init = init
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize cluster centroids."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Random initialization
            indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()
        
        elif self.init == 'kmeans++':
            # k-means++ initialization
            centroids = np.zeros((self.n_clusters, n_features))
            
            # First centroid: random point
            centroids[0] = X[np.random.randint(n_samples)]
            
            for i in range(1, self.n_clusters):
                # Compute distances to nearest centroid
                distances = np.array([
                    min([np.sum((x - centroids[j]) ** 2) for j in range(i)])
                    for x in X
                ])
                
                # Probability proportional to distance squared
                probabilities = distances / distances.sum()
                
                # Choose next centroid
                centroids[i] = X[np.random.choice(n_samples, p=probabilities)]
            
            return centroids
        
        else:
            raise ValueError(f"Unknown init method: {self.init}")
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid."""
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            distances = [np.sum((X[i] - centroid) ** 2) for centroid in centroids]
            labels[i] = np.argmin(distances)
        
        return labels
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids to be the mean of points in each cluster."""
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, keep old centroid or reinitialize
                centroids[k] = X[np.random.randint(len(X))]
        
        return centroids
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Compute within-cluster sum of squares (inertia)."""
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia
    
    def fit(self, X: np.ndarray):
        """
        Fit k-Means clustering.
        
        Args:
            X: Training data of shape (n_samples, n_features).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize centroids
        centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iterations):
            # Assign points to nearest centroids
            labels = self._assign_clusters(X, centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = np.max([
                np.sum((centroids[k] - new_centroids[k]) ** 2)
                for k in range(self.n_clusters)
            ])
            
            if centroid_shift < self.tolerance:
                break
            
            centroids = new_centroids
        
        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels, centroids)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Cluster labels of shape (n_samples,).
        """
        if self.centroids_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self._assign_clusters(X, self.centroids_)


class GaussianMixtureModel:
    """
    Gaussian Mixture Model (GMM) implemented from scratch.
    
    Uses Expectation-Maximization (EM) algorithm to fit a mixture of Gaussians.
    """
    
    def __init__(
        self,
        n_components: int = 1,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        random_state: Optional[int] = None
    ):
        """
        Initialize Gaussian Mixture Model.
        
        Args:
            n_components: Number of mixture components.
            max_iterations: Maximum number of EM iterations.
            tolerance: Convergence tolerance (change in log-likelihood).
            random_state: Random seed for initialization.
        """
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
    
    def _multivariate_gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        """Compute multivariate Gaussian probability density."""
        n_features = len(x)
        
        # Add small value to diagonal for numerical stability
        cov = cov + np.eye(n_features) * 1e-6
        
        try:
            cov_inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            cov_inv = np.linalg.pinv(cov)
            det = np.linalg.det(cov + np.eye(n_features) * 1e-5)
        
        diff = x - mean
        exponent = -0.5 * diff.T @ cov_inv @ diff
        
        normalization = 1.0 / np.sqrt((2 * np.pi) ** n_features * det)
        
        return normalization * np.exp(exponent)
    
    def _initialize_parameters(self, X: np.ndarray):
        """Initialize GMM parameters using k-means-like approach."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Initialize means using random samples
        indices = np.random.choice(n_samples, size=self.n_components, replace=False)
        self.means_ = X[indices].copy()
        
        # Initialize covariances as identity matrices scaled by data variance
        overall_cov = np.cov(X.T)
        # Ensure overall_cov is 2D (for 1D data, np.cov returns scalar)
        if overall_cov.ndim == 0:
            overall_cov = np.array([[overall_cov]])
        elif overall_cov.ndim == 1:
            overall_cov = np.diag(overall_cov)
        self.covariances_ = np.array([
            overall_cov.copy() for _ in range(self.n_components)
        ])
    
    def _expectation_step(self, X: np.ndarray) -> np.ndarray:
        """
        E-step: Compute responsibilities (posterior probabilities).
        
        Returns:
            Responsibilities matrix of shape (n_samples, n_components).
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for i in range(n_samples):
            for k in range(self.n_components):
                # Compute likelihood: weight * Gaussian PDF
                likelihood = self.weights_[k] * self._multivariate_gaussian_pdf(
                    X[i], self.means_[k], self.covariances_[k]
                )
                responsibilities[i, k] = likelihood
        
        # Normalize responsibilities
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10  # Avoid division by zero
        responsibilities = responsibilities / row_sums
        
        return responsibilities
    
    def _maximization_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """M-step: Update parameters to maximize likelihood."""
        n_samples, n_features = X.shape
        
        # Effective number of points assigned to each component
        Nk = responsibilities.sum(axis=0)
        
        # Update weights
        self.weights_ = Nk / n_samples
        
        # Update means
        for k in range(self.n_components):
            self.means_[k] = np.sum(
                responsibilities[:, k:k+1] * X, axis=0
            ) / Nk[k]
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            cov_update = (weighted_diff.T @ diff) / Nk[k]
            
            # Ensure cov_update is 2D array
            if cov_update.ndim == 0:
                cov_update = np.array([[cov_update]])
            elif cov_update.ndim == 1:
                cov_update = np.diag(cov_update)
            
            self.covariances_[k] = cov_update
            
            # Ensure positive definite
            self.covariances_[k] += np.eye(n_features) * 1e-6
    
    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """Compute log-likelihood of data under current model."""
        n_samples = X.shape[0]
        log_likelihood = 0.0
        
        for i in range(n_samples):
            sample_likelihood = 0.0
            for k in range(self.n_components):
                sample_likelihood += self.weights_[k] * self._multivariate_gaussian_pdf(
                    X[i], self.means_[k], self.covariances_[k]
                )
            log_likelihood += np.log(sample_likelihood + 1e-10)
        
        return log_likelihood
    
    def fit(self, X: np.ndarray):
        """
        Fit the Gaussian Mixture Model using EM algorithm.
        
        Args:
            X: Training data of shape (n_samples, n_features).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        prev_log_likelihood = -float('inf')
        
        for iteration in range(self.max_iterations):
            # E-step
            responsibilities = self._expectation_step(X)
            
            # M-step
            self._maximization_step(X, responsibilities)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                break
            
            prev_log_likelihood = log_likelihood
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict posterior probabilities for each component.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Probabilities of shape (n_samples, n_components).
        """
        if self.means_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self._expectation_step(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments (hard clustering).
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Cluster labels of shape (n_samples,).
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
