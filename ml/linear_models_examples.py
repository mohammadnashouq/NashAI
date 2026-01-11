"""
Examples demonstrating all functions in the linear_models module.

This script shows how to use Linear Regression, Logistic Regression,
k-NN, Naive Bayes, and SVM with clear examples and outputs.
"""

import numpy as np
import sys
import os

# Add parent directory to path for core_math imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.linear_models import (
    LinearRegression,
    LogisticRegression,
    KNeighborsClassifier,
    NaiveBayes,
    SVM
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    """Run all examples."""
    
    print_section("1. LINEAR REGRESSION")
    
    # Example 1: Simple linear relationship
    print("Example 1: y = 2x + 1 + noise")
    np.random.seed(42)
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
    y = 2 * X.flatten() + 1 + 0.1 * np.random.randn(8)
    
    print(f"Training data:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  First 3 samples: X={X[:3].flatten()}, y={y[:3]}")
    
    # Fit using normal equation
    lr_normal = LinearRegression(fit_intercept=True, method='normal')
    lr_normal.fit(X, y)
    
    print(f"\nFitted model (normal equation):")
    print(f"  Coefficients: {lr_normal.coef_}")
    print(f"  Intercept: {lr_normal.intercept_:.4f}")
    print(f"  Expected: coef~2.0, intercept~1.0")
    
    # Predictions
    X_test = np.array([[9.0], [10.0]])
    y_pred = lr_normal.predict(X_test)
    print(f"\nPredictions for X_test = {X_test.flatten()}:")
    print(f"  Predictions: {y_pred}")
    print(f"  Expected: ~[19, 21]")
    
    # Example 2: Multiple features
    print("\n" + "-" * 70)
    print("Example 2: Multiple features y = 1.5*x1 + 2.5*x2 - 0.5")
    np.random.seed(42)
    X_multi = np.random.randn(20, 2)
    y_multi = 1.5 * X_multi[:, 0] + 2.5 * X_multi[:, 1] - 0.5 + 0.1 * np.random.randn(20)
    
    lr_multi = LinearRegression()
    lr_multi.fit(X_multi, y_multi)
    
    print(f"Fitted model:")
    print(f"  Coefficients: {lr_multi.coef_}")
    print(f"  Intercept: {lr_multi.intercept_:.4f}")
    print(f"  Expected: coef~[1.5, 2.5], intercept~-0.5")
    
    
    print_section("2. LOGISTIC REGRESSION")
    
    # Example: Binary classification
    print("Example: Binary classification (two classes)")
    np.random.seed(42)
    
    # Generate two classes
    n_samples = 100
    X_class1 = np.random.randn(n_samples, 2) + [2, 2]
    X_class2 = np.random.randn(n_samples, 2) + [-2, -2]
    X_log = np.vstack([X_class1, X_class2])
    y_log = np.hstack([np.ones(n_samples), np.zeros(n_samples)])
    
    print(f"Training data:")
    print(f"  X shape: {X_log.shape}")
    print(f"  y shape: {y_log.shape}")
    print(f"  Class distribution: {np.bincount(y_log.astype(int))}")
    
    # Fit model
    log_reg = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    log_reg.fit(X_log, y_log)
    
    print(f"\nFitted model:")
    print(f"  Coefficients: {log_reg.coef_}")
    print(f"  Intercept: {log_reg.intercept_:.4f}")
    
    # Predictions
    X_test_log = np.array([[2.0, 2.0], [-2.0, -2.0], [0.0, 0.0]])
    y_pred_log = log_reg.predict(X_test_log)
    y_proba_log = log_reg.predict_proba(X_test_log)
    
    print(f"\nPredictions for test points:")
    for i, (x, pred, proba) in enumerate(zip(X_test_log, y_pred_log, y_proba_log)):
        print(f"  Point {x}: prediction={pred}, probability={proba:.4f}")
    
    # Accuracy on training data
    train_pred = log_reg.predict(X_log)
    accuracy = np.mean(train_pred == y_log)
    print(f"\nTraining accuracy: {accuracy:.4f}")
    
    
    print_section("3. k-NEAREST NEIGHBORS (k-NN)")
    
    # Example: Classification
    print("Example: k-NN Classification")
    np.random.seed(42)
    
    # Generate three classes
    X_knn = np.array([
        [1, 1], [1, 2], [2, 1], [2, 2],  # Class 0
        [5, 5], [5, 6], [6, 5], [6, 6],  # Class 1
        [1, 5], [2, 5], [1, 6], [2, 6]   # Class 2
    ])
    y_knn = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    
    print(f"Training data:")
    print(f"  X shape: {X_knn.shape}")
    print(f"  Classes: {np.unique(y_knn)}")
    
    # Fit with different k values
    for k in [1, 3, 5]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_knn, y_knn)
        
        X_test_knn = np.array([[3, 3], [4, 4], [1.5, 5.5]])
        y_pred_knn = knn.predict(X_test_knn)
        
        print(f"\nk={k} predictions for test points:")
        for x, pred in zip(X_test_knn, y_pred_knn):
            print(f"  Point {x}: predicted class={pred}")
    
    # Example with different distance metric
    print("\n" + "-" * 70)
    print("Example: k-NN with Manhattan distance")
    knn_manhattan = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
    knn_manhattan.fit(X_knn, y_knn)
    y_pred_man = knn_manhattan.predict(X_test_knn)
    print(f"Predictions (Manhattan): {y_pred_man}")
    
    
    print_section("4. NAIVE BAYES")
    
    # Example: Classification
    print("Example: Gaussian Naive Bayes Classification")
    np.random.seed(42)
    
    # Generate two classes with different means
    X_nb = np.vstack([
        np.random.randn(50, 2) + [1, 1],  # Class 0
        np.random.randn(50, 2) + [-1, -1]  # Class 1
    ])
    y_nb = np.hstack([np.zeros(50), np.ones(50)])
    
    print(f"Training data:")
    print(f"  X shape: {X_nb.shape}")
    print(f"  Class distribution: {np.bincount(y_nb.astype(int))}")
    
    # Fit model
    nb = NaiveBayes()
    nb.fit(X_nb, y_nb)
    
    print(f"\nFitted model:")
    print(f"  Classes: {nb.classes_}")
    print(f"  Class priors: {nb.class_priors_}")
    print(f"  Means for each class:")
    for i, c in enumerate(nb.classes_):
        print(f"    Class {c}: {nb.means_[i]}")
    print(f"  Standard deviations for each class:")
    for i, c in enumerate(nb.classes_):
        print(f"    Class {c}: {nb.stds_[i]}")
    
    # Predictions
    X_test_nb = np.array([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])
    y_pred_nb = nb.predict(X_test_nb)
    y_proba_nb = nb.predict_proba(X_test_nb)
    
    print(f"\nPredictions for test points:")
    for i, (x, pred, proba) in enumerate(zip(X_test_nb, y_pred_nb, y_proba_nb)):
        print(f"  Point {x}: prediction={pred}, probabilities={proba}")
    
    # Accuracy
    train_pred_nb = nb.predict(X_nb)
    accuracy_nb = np.mean(train_pred_nb == y_nb)
    print(f"\nTraining accuracy: {accuracy_nb:.4f}")
    
    
    print_section("5. SUPPORT VECTOR MACHINE (SVM)")
    
    # Example: Binary classification
    print("Example: SVM Binary Classification")
    np.random.seed(42)
    
    # Generate linearly separable data
    X_svm = np.vstack([
        np.random.randn(30, 2) + [2, 2],  # Class 1
        np.random.randn(30, 2) + [-2, -2]  # Class -1
    ])
    y_svm = np.hstack([np.ones(30), -np.ones(30)])
    
    print(f"Training data:")
    print(f"  X shape: {X_svm.shape}")
    print(f"  Classes: {np.unique(y_svm)}")
    
    # Fit model
    svm = SVM(C=1.0, learning_rate=0.01, max_iterations=1000)
    svm.fit(X_svm, y_svm)
    
    print(f"\nFitted model:")
    print(f"  Coefficients: {svm.coef_}")
    print(f"  Intercept: {svm.intercept_:.4f}")
    
    # Predictions
    X_test_svm = np.array([[2.0, 2.0], [-2.0, -2.0], [0.0, 0.0]])
    y_pred_svm = svm.predict(X_test_svm)
    
    print(f"\nPredictions for test points:")
    for x, pred in zip(X_test_svm, y_pred_svm):
        print(f"  Point {x}: prediction={pred}")
    
    # Accuracy
    train_pred_svm = svm.predict(X_svm)
    accuracy_svm = np.mean(train_pred_svm == y_svm)
    print(f"\nTraining accuracy: {accuracy_svm:.4f}")
    
    # Example with different C values
    print("\n" + "-" * 70)
    print("Example: Effect of regularization parameter C")
    for C in [0.1, 1.0, 10.0]:
        svm_c = SVM(C=C, learning_rate=0.01, max_iterations=1000)
        svm_c.fit(X_svm, y_svm)
        train_pred_c = svm_c.predict(X_svm)
        acc_c = np.mean(train_pred_c == y_svm)
        print(f"  C={C}: Training accuracy={acc_c:.4f}")
    
    
    print_section("EXAMPLES COMPLETE!")
    print("All linear models have been demonstrated.")
    print("You can use these examples as a reference for your own tasks.")


if __name__ == "__main__":
    main()
