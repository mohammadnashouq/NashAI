"""
Linear models and related supervised learning algorithms.

This module implements Linear Regression, Logistic Regression, k-NN,
Naive Bayes, and SVM from scratch without using sklearn.
"""

import numpy as np
from typing import Optional, Union, Tuple
import sys
import os

# Add parent directory to path for core_math imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from core_math.optimization import GradientDescent
except ImportError:
    # Fallback if core_math is not available
    GradientDescent = None


class LinearRegression:
    """
    Linear Regression implemented from scratch.
    
    Fits a linear model: y = X @ w + b
    Uses least squares solution (normal equation) or gradient descent.
    """
    
    def __init__(self, fit_intercept: bool = True, method: str = 'normal'):
        """
        Initialize Linear Regression.
        
        Args:
            fit_intercept: Whether to fit the intercept term (bias).
            method: 'normal' for normal equation, 'gradient' for gradient descent.
        """
        self.fit_intercept = fit_intercept
        self.method = method
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the linear regression model.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            # Add column of ones for intercept
            X_aug = np.column_stack([np.ones(n_samples), X])
        else:
            X_aug = X
        
        if self.method == 'normal':
            # Normal equation: w = (X^T @ X)^(-1) @ X^T @ y
            # Using pseudo-inverse for numerical stability
            try:
                coef = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                coef = np.linalg.pinv(X_aug) @ y
            
            if self.fit_intercept:
                self.intercept_ = coef[0]
                self.coef_ = coef[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = coef
                
        elif self.method == 'gradient':
            # Gradient descent approach
            if GradientDescent is None:
                raise ImportError("GradientDescent not available. Use method='normal'")
            
            # Initialize weights
            n_params = n_features + (1 if self.fit_intercept else 0)
            w0 = np.random.randn(n_params) * 0.01
            
            # Cost function: MSE
            def cost(params):
                if self.fit_intercept:
                    predictions = X @ params[1:] + params[0]
                else:
                    predictions = X @ params
                return np.mean((predictions - y) ** 2)
            
            # Gradient
            def gradient(params):
                if self.fit_intercept:
                    predictions = X @ params[1:] + params[0]
                    errors = predictions - y
                    grad_b = 2 * np.mean(errors)
                    grad_w = 2 * np.mean(errors[:, np.newaxis] * X, axis=0)
                    return np.concatenate([[grad_b], grad_w])
                else:
                    predictions = X @ params
                    errors = predictions - y
                    return 2 * np.mean(errors[:, np.newaxis] * X, axis=0)
            
            optimizer = GradientDescent(learning_rate=0.01, max_iterations=1000)
            w_opt, _ = optimizer.minimize(cost, w0, gradient=gradient)
            
            if self.fit_intercept:
                self.intercept_ = w_opt[0]
                self.coef_ = w_opt[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = w_opt
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Predictions of shape (n_samples,).
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X @ self.coef_ + self.intercept_


class LogisticRegression:
    """
    Logistic Regression implemented from scratch.
    
    Uses sigmoid activation: p = 1 / (1 + exp(-(X @ w + b)))
    Trained with gradient descent to minimize cross-entropy loss.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        fit_intercept: bool = True
    ):
        """
        Initialize Logistic Regression.
        
        Args:
            learning_rate: Learning rate for gradient descent.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance.
            fit_intercept: Whether to fit intercept term.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the logistic regression model.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,). Should be 0 or 1.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize weights
        if self.fit_intercept:
            w = np.random.randn(n_features) * 0.01
            b = 0.0
        else:
            w = np.random.randn(n_features) * 0.01
            b = 0.0
        
        prev_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward pass
            z = X @ w + b
            predictions = self._sigmoid(z)
            
            # Compute loss (cross-entropy)
            loss = -np.mean(y * np.log(predictions + 1e-15) + 
                          (1 - y) * np.log(1 - predictions + 1e-15))
            
            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                break
            
            prev_loss = loss
            
            # Compute gradients
            errors = predictions - y
            grad_w = np.mean(errors[:, np.newaxis] * X, axis=0)
            grad_b = np.mean(errors) if self.fit_intercept else 0.0
            
            # Update weights
            w -= self.learning_rate * grad_w
            if self.fit_intercept:
                b -= self.learning_rate * grad_b
        
        self.coef_ = w
        self.intercept_ = b if self.fit_intercept else 0.0
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Probabilities of shape (n_samples,).
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        z = X @ self.coef_ + self.intercept_
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Binary predictions (0 or 1) of shape (n_samples,).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


class KNeighborsClassifier:
    """
    k-Nearest Neighbors Classifier implemented from scratch.
    
    Predicts class based on majority vote of k nearest neighbors.
    """
    
    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean'):
        """
        Initialize k-NN Classifier.
        
        Args:
            n_neighbors: Number of neighbors to consider.
            metric: Distance metric ('euclidean' or 'manhattan').
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None
    
    def _distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute distance between two points."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the k-NN model (just stores training data).
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
        """
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        
        if self.X_train.ndim == 1:
            self.X_train = self.X_train.reshape(-1, 1)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Predicted labels of shape (n_samples,).
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        predictions = []
        
        for x in X:
            # Compute distances to all training points
            distances = [self._distance(x, x_train) for x_train in self.X_train]
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_labels = self.y_train[k_indices]
            
            # Majority vote
            unique_labels, counts = np.unique(k_labels, return_counts=True)
            prediction = unique_labels[np.argmax(counts)]
            predictions.append(prediction)
        
        return np.array(predictions)


class NaiveBayes:
    """
    Naive Bayes Classifier implemented from scratch.
    
    Assumes features are conditionally independent given the class.
    Uses Gaussian distribution for continuous features.
    """
    
    def __init__(self):
        """Initialize Naive Bayes Classifier."""
        self.classes_ = None
        self.class_priors_ = None
        self.means_ = None
        self.stds_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Naive Bayes model.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize storage
        self.class_priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.stds_ = np.zeros((n_classes, n_features))
        
        # Compute statistics for each class
        for i, c in enumerate(self.classes_):
            mask = y == c
            X_class = X[mask]
            
            # Prior probability
            self.class_priors_[i] = np.mean(mask)
            
            # Mean and std for each feature
            self.means_[i] = np.mean(X_class, axis=0)
            self.stds_[i] = np.std(X_class, axis=0) + 1e-10  # Add small value for stability
        
        return self
    
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        """Compute Gaussian probability density."""
        exponent = -0.5 * np.sum(((x - mean) / std) ** 2)
        normalization = 1.0 / (np.sqrt(2 * np.pi) * np.prod(std))
        return normalization * np.exp(exponent)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Probabilities of shape (n_samples, n_classes).
        """
        if self.classes_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X):
            for j, c in enumerate(self.classes_):
                # Compute likelihood using Naive Bayes assumption
                likelihood = self._gaussian_pdf(x, self.means_[j], self.stds_[j])
                # Posterior = prior * likelihood (unnormalized)
                probabilities[i, j] = self.class_priors_[j] * likelihood
        
        # Normalize probabilities
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Predicted labels of shape (n_samples,).
        """
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]


class SVM:
    """
    Support Vector Machine (SVM) implemented from scratch.
    
    Uses gradient descent to optimize the hinge loss with L2 regularization.
    Simplified version without kernel trick.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ):
        """
        Initialize SVM.
        
        Args:
            C: Regularization parameter (inverse of regularization strength).
            learning_rate: Learning rate for gradient descent.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance.
        """
        self.C = C
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM model.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,). Should be -1 or 1.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        
        # Convert labels to -1 and 1 if needed
        unique_labels = np.unique(y)
        if len(unique_labels) == 2:
            y = np.where(y == unique_labels[0], -1, 1)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize weights
        w = np.random.randn(n_features) * 0.01
        b = 0.0
        
        prev_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            # Compute margin: y * (X @ w + b)
            margins = y * (X @ w + b)
            
            # Hinge loss: max(0, 1 - margin)
            hinge_loss = np.maximum(0, 1 - margins)
            loss = np.mean(hinge_loss) + (1 / (2 * self.C)) * np.sum(w ** 2)
            
            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                break
            
            prev_loss = loss
            
            # Compute gradients
            # For samples with margin < 1, gradient is -y * X
            mask = margins < 1
            grad_w = np.zeros(n_features)
            grad_b = 0.0
            
            if np.any(mask):
                grad_w = -np.mean(y[mask, np.newaxis] * X[mask], axis=0)
                grad_b = -np.mean(y[mask])
            
            # Add regularization gradient
            grad_w += (1 / self.C) * w
            
            # Update weights
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
        
        self.coef_ = w
        self.intercept_ = b
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Predicted labels of shape (n_samples,).
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Decision function: X @ w + b
        decisions = X @ self.coef_ + self.intercept_
        
        # Return -1 or 1
        return np.sign(decisions)
