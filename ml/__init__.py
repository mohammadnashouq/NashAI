"""
Machine learning algorithms implemented from scratch.

This module contains implementations of supervised and unsupervised
learning algorithms without using sklearn.
"""

from .linear_models import (
    LinearRegression,
    LogisticRegression,
    KNeighborsClassifier,
    NaiveBayes,
    SVM
)

from .trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)

from .clustering import (
    KMeans,
    GaussianMixtureModel
)

from .decomposition import (
    PCA,
    ICA
)

__all__ = [
    # Linear models
    'LinearRegression',
    'LogisticRegression',
    'KNeighborsClassifier',
    'NaiveBayes',
    'SVM',
    # Trees
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    'RandomForestClassifier',
    'RandomForestRegressor',
    # Clustering
    'KMeans',
    'GaussianMixtureModel',
    # Decomposition
    'PCA',
    'ICA',
]
