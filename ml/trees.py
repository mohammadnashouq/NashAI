"""
Decision Trees and Random Forest implemented from scratch.

This module implements Decision Trees (classification and regression)
and Random Forest without using sklearn.
"""

import numpy as np
from typing import Optional, Union, Tuple, List
from collections import Counter


class DecisionTreeClassifier:
    """
    Decision Tree Classifier implemented from scratch.
    
    Uses recursive binary splitting based on information gain (entropy).
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini'
    ):
        """
        Initialize Decision Tree Classifier.
        
        Args:
            max_depth: Maximum depth of the tree. None for no limit.
            min_samples_split: Minimum samples required to split a node.
            min_samples_leaf: Minimum samples required in a leaf node.
            criterion: Splitting criterion ('gini' or 'entropy').
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None
    
    def _gini(self, y: np.ndarray) -> float:
        """Compute Gini impurity."""
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y)
        proportions = counts / len(y)
        return 1.0 - np.sum(proportions ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """Compute entropy."""
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y)
        proportions = counts[counts > 0] / len(y)
        return -np.sum(proportions * np.log2(proportions))
    
    def _impurity(self, y: np.ndarray) -> float:
        """Compute impurity based on criterion."""
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """
        Find the best feature and threshold to split on.
        
        Returns:
            Tuple of (feature_index, threshold).
        """
        best_feature = None
        best_threshold = None
        best_gain = -float('inf')
        
        n_features = X.shape[1]
        parent_impurity = self._impurity(y)
        
        for feature_idx in range(n_features):
            # Get unique values for this feature
            feature_values = np.unique(X[:, feature_idx])
            
            # Try each threshold (midpoint between consecutive values)
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Compute weighted impurity
                left_impurity = self._impurity(y[left_mask])
                right_impurity = self._impurity(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(y)
                
                weighted_impurity = (n_left / n_total) * left_impurity + \
                                  (n_right / n_total) * right_impurity
                
                # Information gain
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> dict:
        """
        Recursively build the decision tree.
        
        Returns:
            Dictionary representing a node in the tree.
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            # Leaf node: return majority class
            counts = np.bincount(y)
            return {
                'leaf': True,
                'class': np.argmax(counts),
                'samples': n_samples
            }
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            # No valid split found
            counts = np.bincount(y)
            return {
                'leaf': True,
                'class': np.argmax(counts),
                'samples': n_samples
            }
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
            'samples': n_samples
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the decision tree.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).flatten()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: dict) -> int:
        """Predict a single sample."""
        if node['leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Predicted labels of shape (n_samples,).
        """
        if self.tree is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        predictions = [self._predict_sample(x, self.tree) for x in X]
        return np.array(predictions)


class DecisionTreeRegressor:
    """
    Decision Tree Regressor implemented from scratch.
    
    Uses recursive binary splitting based on variance reduction (MSE).
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ):
        """
        Initialize Decision Tree Regressor.
        
        Args:
            max_depth: Maximum depth of the tree. None for no limit.
            min_samples_split: Minimum samples required to split a node.
            min_samples_leaf: Minimum samples required in a leaf node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def _mse(self, y: np.ndarray) -> float:
        """Compute mean squared error (variance)."""
        if len(y) == 0:
            return 0.0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """
        Find the best feature and threshold to split on.
        
        Returns:
            Tuple of (feature_index, threshold).
        """
        best_feature = None
        best_threshold = None
        best_gain = -float('inf')
        
        n_features = X.shape[1]
        parent_mse = self._mse(y)
        
        for feature_idx in range(n_features):
            feature_values = np.unique(X[:, feature_idx])
            
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(y)
                
                weighted_mse = (n_left / n_total) * left_mse + \
                              (n_right / n_total) * right_mse
                
                gain = parent_mse - weighted_mse
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> dict:
        """Recursively build the decision tree."""
        n_samples = len(y)
        
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            # Leaf node: return mean value
            return {
                'leaf': True,
                'value': np.mean(y),
                'samples': n_samples
            }
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return {
                'leaf': True,
                'value': np.mean(y),
                'samples': n_samples
            }
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
            'samples': n_samples
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the decision tree.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: dict) -> float:
        """Predict a single sample."""
        if node['leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Predicted values of shape (n_samples,).
        """
        if self.tree is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        predictions = [self._predict_sample(x, self.tree) for x in X]
        return np.array(predictions)


class RandomForestClassifier:
    """
    Random Forest Classifier implemented from scratch.
    
    Ensemble of decision trees trained on bootstrap samples with random feature selection.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = 'sqrt',
        random_state: Optional[int] = None
    ):
        """
        Initialize Random Forest Classifier.
        
        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each tree.
            min_samples_split: Minimum samples to split a node.
            min_samples_leaf: Minimum samples in a leaf node.
            max_features: Number of features to consider for each split.
                          Can be int, float (fraction), 'sqrt', or 'log2'.
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.n_features_ = None
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a bootstrap sample."""
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def _get_max_features(self, n_features: int) -> int:
        """Determine number of features to use for each split."""
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features)) + 1
        else:
            return n_features
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the random forest.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).flatten()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_ = X.shape[1]
        self.trees = []
        self.feature_indices_list = []
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            X_boot, y_boot = self._bootstrap_sample(X, y)
            
            # Select random feature subset
            max_features = self._get_max_features(self.n_features_)
            feature_indices = np.random.choice(
                self.n_features_,
                size=max_features,
                replace=False
            )
            self.feature_indices_list.append(feature_indices)
            
            # Create and train tree on subset of features
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            
            # Train on subset of features
            tree.fit(X_boot[:, feature_indices], y_boot)
            
            self.trees.append(tree)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using majority voting.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Predicted labels of shape (n_samples,).
        """
        if len(self.trees) == 0:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        all_predictions = []
        
        for tree, feature_indices in zip(self.trees, self.feature_indices_list):
            # Use only the features this tree was trained on
            X_subset = X[:, feature_indices]
            predictions = tree.predict(X_subset)
            all_predictions.append(predictions)
        
        # Majority vote
        all_predictions = np.array(all_predictions)
        final_predictions = []
        
        for i in range(n_samples):
            votes = all_predictions[:, i]
            counts = np.bincount(votes)
            final_predictions.append(np.argmax(counts))
        
        return np.array(final_predictions)


class RandomForestRegressor:
    """
    Random Forest Regressor implemented from scratch.
    
    Ensemble of regression trees trained on bootstrap samples.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = 'sqrt',
        random_state: Optional[int] = None
    ):
        """
        Initialize Random Forest Regressor.
        
        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each tree.
            min_samples_split: Minimum samples to split a node.
            min_samples_leaf: Minimum samples in a leaf node.
            max_features: Number of features to consider for each split.
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.n_features_ = None
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a bootstrap sample."""
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def _get_max_features(self, n_features: int) -> int:
        """Determine number of features to use for each split."""
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features)) + 1
        else:
            return n_features
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the random forest.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_ = X.shape[1]
        self.trees = []
        self.feature_indices_list = []
        
        for i in range(self.n_estimators):
            X_boot, y_boot = self._bootstrap_sample(X, y)
            
            # Select random feature subset
            max_features = self._get_max_features(self.n_features_)
            feature_indices = np.random.choice(
                self.n_features_,
                size=max_features,
                replace=False
            )
            self.feature_indices_list.append(feature_indices)
            
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            
            tree.fit(X_boot[:, feature_indices], y_boot)
            
            self.trees.append(tree)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using average of all trees.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            Predicted values of shape (n_samples,).
        """
        if len(self.trees) == 0:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        all_predictions = []
        
        for tree, feature_indices in zip(self.trees, self.feature_indices_list):
            X_subset = X[:, feature_indices]
            predictions = tree.predict(X_subset)
            all_predictions.append(predictions)
        
        # Average predictions
        all_predictions = np.array(all_predictions)
        return np.mean(all_predictions, axis=0)
