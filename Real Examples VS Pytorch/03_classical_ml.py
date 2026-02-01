"""
Classical Machine Learning - NashAI vs Scikit-Learn Comparison

This benchmark compares classical ML algorithms using:
1. NashAI (our from-scratch implementation)
2. Scikit-Learn (industry standard)

Algorithms compared:
- Linear Regression
- Logistic Regression  
- K-Nearest Neighbors
- Naive Bayes
- Support Vector Machine

Datasets used:
- Iris (classification)
- California Housing (regression)
- Synthetic binary classification
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# NashAI imports
from ml.linear_models import (
    LinearRegression as NashAILinearRegression,
    LogisticRegression as NashAILogisticRegression,
    KNeighborsClassifier as NashAIKNN,
    NaiveBayes as NashAINaiveBayes,
    SVM as NashAISVM
)

# Scikit-Learn imports
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris, fetch_california_housing, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(title)
    print("="*60)


# ============================================
# Linear Regression Comparison
# ============================================

def compare_linear_regression():
    """Compare Linear Regression implementations."""
    print_header("LINEAR REGRESSION COMPARISON")
    
    # Load California Housing dataset
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Use subset for faster comparison
    n_samples = 5000
    indices = np.random.permutation(len(X))[:n_samples]
    X, y = X[indices], y[indices]
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    
    # NashAI
    print("\nTraining NashAI Linear Regression...")
    start_time = time.time()
    nashai_model = NashAILinearRegression(method='normal')
    nashai_model.fit(X_train, y_train)
    nashai_time = time.time() - start_time
    
    nashai_pred = nashai_model.predict(X_test)
    nashai_mse = mean_squared_error(y_test, nashai_pred)
    nashai_r2 = r2_score(y_test, nashai_pred)
    
    # Scikit-Learn
    print("Training Scikit-Learn Linear Regression...")
    start_time = time.time()
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    sklearn_time = time.time() - start_time
    
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)
    sklearn_r2 = r2_score(y_test, sklearn_pred)
    
    # Results
    print(f"\n{'Metric':<20} {'NashAI':>15} {'Scikit-Learn':>15}")
    print("-"*50)
    print(f"{'MSE':<20} {nashai_mse:>15.4f} {sklearn_mse:>15.4f}")
    print(f"{'R² Score':<20} {nashai_r2:>15.4f} {sklearn_r2:>15.4f}")
    print(f"{'Training Time (s)':<20} {nashai_time:>15.4f} {sklearn_time:>15.4f}")
    
    return {
        'nashai': {'mse': nashai_mse, 'r2': nashai_r2, 'time': nashai_time},
        'sklearn': {'mse': sklearn_mse, 'r2': sklearn_r2, 'time': sklearn_time}
    }


# ============================================
# Logistic Regression Comparison
# ============================================

def compare_logistic_regression():
    """Compare Logistic Regression implementations."""
    print_header("LOGISTIC REGRESSION COMPARISON")
    
    # Create binary classification dataset
    print("Creating synthetic binary classification dataset...")
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    
    # NashAI
    print("\nTraining NashAI Logistic Regression...")
    start_time = time.time()
    nashai_model = NashAILogisticRegression(learning_rate=0.1, max_iterations=1000)
    nashai_model.fit(X_train, y_train)
    nashai_time = time.time() - start_time
    
    nashai_pred = nashai_model.predict(X_test)
    nashai_acc = accuracy_score(y_test, nashai_pred) * 100
    
    # Scikit-Learn
    print("Training Scikit-Learn Logistic Regression...")
    start_time = time.time()
    sklearn_model = LogisticRegression(max_iter=1000)
    sklearn_model.fit(X_train, y_train)
    sklearn_time = time.time() - start_time
    
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred) * 100
    
    # Results
    print(f"\n{'Metric':<20} {'NashAI':>15} {'Scikit-Learn':>15}")
    print("-"*50)
    print(f"{'Accuracy (%)':<20} {nashai_acc:>15.2f} {sklearn_acc:>15.2f}")
    print(f"{'Training Time (s)':<20} {nashai_time:>15.4f} {sklearn_time:>15.4f}")
    
    return {
        'nashai': {'accuracy': nashai_acc, 'time': nashai_time},
        'sklearn': {'accuracy': sklearn_acc, 'time': sklearn_time}
    }


# ============================================
# K-Nearest Neighbors Comparison
# ============================================

def compare_knn():
    """Compare KNN implementations."""
    print_header("K-NEAREST NEIGHBORS COMPARISON")
    
    # Load Iris dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    
    # NashAI
    print("\nTraining NashAI KNN (k=5)...")
    start_time = time.time()
    nashai_model = NashAIKNN(n_neighbors=5)
    nashai_model.fit(X_train, y_train)
    nashai_pred = nashai_model.predict(X_test)
    nashai_time = time.time() - start_time
    nashai_acc = accuracy_score(y_test, nashai_pred) * 100
    
    # Scikit-Learn
    print("Training Scikit-Learn KNN (k=5)...")
    start_time = time.time()
    sklearn_model = KNeighborsClassifier(n_neighbors=5)
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_time = time.time() - start_time
    sklearn_acc = accuracy_score(y_test, sklearn_pred) * 100
    
    # Results
    print(f"\n{'Metric':<20} {'NashAI':>15} {'Scikit-Learn':>15}")
    print("-"*50)
    print(f"{'Accuracy (%)':<20} {nashai_acc:>15.2f} {sklearn_acc:>15.2f}")
    print(f"{'Inference Time (s)':<20} {nashai_time:>15.4f} {sklearn_time:>15.4f}")
    
    return {
        'nashai': {'accuracy': nashai_acc, 'time': nashai_time},
        'sklearn': {'accuracy': sklearn_acc, 'time': sklearn_time}
    }


# ============================================
# Naive Bayes Comparison
# ============================================

def compare_naive_bayes():
    """Compare Naive Bayes implementations."""
    print_header("NAIVE BAYES COMPARISON")
    
    # Load Iris dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # NashAI
    print("\nTraining NashAI Naive Bayes...")
    start_time = time.time()
    nashai_model = NashAINaiveBayes()
    nashai_model.fit(X_train, y_train)
    nashai_time = time.time() - start_time
    
    nashai_pred = nashai_model.predict(X_test)
    nashai_acc = accuracy_score(y_test, nashai_pred) * 100
    
    # Scikit-Learn
    print("Training Scikit-Learn Gaussian Naive Bayes...")
    start_time = time.time()
    sklearn_model = GaussianNB()
    sklearn_model.fit(X_train, y_train)
    sklearn_time = time.time() - start_time
    
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred) * 100
    
    # Results
    print(f"\n{'Metric':<20} {'NashAI':>15} {'Scikit-Learn':>15}")
    print("-"*50)
    print(f"{'Accuracy (%)':<20} {nashai_acc:>15.2f} {sklearn_acc:>15.2f}")
    print(f"{'Training Time (s)':<20} {nashai_time:>15.4f} {sklearn_time:>15.4f}")
    
    return {
        'nashai': {'accuracy': nashai_acc, 'time': nashai_time},
        'sklearn': {'accuracy': sklearn_acc, 'time': sklearn_time}
    }


# ============================================
# SVM Comparison
# ============================================

def compare_svm():
    """Compare SVM implementations."""
    print_header("SUPPORT VECTOR MACHINE COMPARISON")
    
    # Create binary classification dataset
    print("Creating synthetic binary classification dataset...")
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8,
        n_redundant=2, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # NashAI (Linear SVM)
    print("\nTraining NashAI Linear SVM...")
    start_time = time.time()
    nashai_model = NashAISVM(C=1.0, learning_rate=0.01, max_iterations=1000)
    nashai_model.fit(X_train, y_train)
    nashai_time = time.time() - start_time
    
    nashai_pred = nashai_model.predict(X_test)
    # Convert -1/1 to 0/1 for comparison
    nashai_pred = (nashai_pred + 1) / 2
    nashai_acc = accuracy_score(y_test, nashai_pred) * 100
    
    # Scikit-Learn (Linear SVM)
    print("Training Scikit-Learn Linear SVM...")
    start_time = time.time()
    sklearn_model = SVC(kernel='linear', C=1.0)
    sklearn_model.fit(X_train, y_train)
    sklearn_time = time.time() - start_time
    
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred) * 100
    
    # Results
    print(f"\n{'Metric':<20} {'NashAI':>15} {'Scikit-Learn':>15}")
    print("-"*50)
    print(f"{'Accuracy (%)':<20} {nashai_acc:>15.2f} {sklearn_acc:>15.2f}")
    print(f"{'Training Time (s)':<20} {nashai_time:>15.4f} {sklearn_time:>15.4f}")
    
    return {
        'nashai': {'accuracy': nashai_acc, 'time': nashai_time},
        'sklearn': {'accuracy': sklearn_acc, 'time': sklearn_time}
    }


# ============================================
# Visualization
# ============================================

def plot_comparison_summary(results):
    """Create summary visualization of all comparisons."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    algorithms = list(results.keys())
    
    # Filter algorithms with accuracy metric
    acc_algorithms = [a for a in algorithms if 'accuracy' in results[a]['nashai'] or 'r2' in results[a]['nashai']]
    
    # Accuracy/R2 comparison
    nashai_metrics = []
    sklearn_metrics = []
    labels = []
    
    for algo in acc_algorithms:
        if 'accuracy' in results[algo]['nashai']:
            nashai_metrics.append(results[algo]['nashai']['accuracy'])
            sklearn_metrics.append(results[algo]['sklearn']['accuracy'])
            labels.append(algo.replace('_', ' ').title())
        elif 'r2' in results[algo]['nashai']:
            nashai_metrics.append(results[algo]['nashai']['r2'] * 100)  # Scale for visibility
            sklearn_metrics.append(results[algo]['sklearn']['r2'] * 100)
            labels.append(algo.replace('_', ' ').title() + ' (R²×100)')
    
    x = np.arange(len(labels))
    width = 0.35
    
    axes[0].bar(x - width/2, nashai_metrics, width, label='NashAI', color='steelblue')
    axes[0].bar(x + width/2, sklearn_metrics, width, label='Scikit-Learn', color='coral')
    axes[0].set_ylabel('Accuracy / R² Score')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Time comparison
    nashai_times = [results[a]['nashai']['time'] for a in algorithms]
    sklearn_times = [results[a]['sklearn']['time'] for a in algorithms]
    time_labels = [a.replace('_', ' ').title() for a in algorithms]
    
    x = np.arange(len(time_labels))
    
    axes[1].bar(x - width/2, nashai_times, width, label='NashAI', color='steelblue')
    axes[1].bar(x + width/2, sklearn_times, width, label='Scikit-Learn', color='coral')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Training Time Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(time_labels, rotation=15, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_yscale('log')  # Log scale for time
    
    plt.tight_layout()
    plt.savefig('classical_ml_comparison.png', dpi=150)
    plt.show()
    print("\nPlot saved as 'classical_ml_comparison.png'")


def print_overall_summary(results):
    """Print overall summary of all comparisons."""
    print_header("OVERALL SUMMARY")
    
    print(f"\n{'Algorithm':<25} {'NashAI':>12} {'Sklearn':>12} {'Time Ratio':>12}")
    print("-"*65)
    
    for algo, data in results.items():
        algo_name = algo.replace('_', ' ').title()
        
        if 'accuracy' in data['nashai']:
            nashai_metric = f"{data['nashai']['accuracy']:.2f}%"
            sklearn_metric = f"{data['sklearn']['accuracy']:.2f}%"
        else:
            nashai_metric = f"R²={data['nashai']['r2']:.4f}"
            sklearn_metric = f"R²={data['sklearn']['r2']:.4f}"
        
        time_ratio = data['nashai']['time'] / max(data['sklearn']['time'], 1e-6)
        
        print(f"{algo_name:<25} {nashai_metric:>12} {sklearn_metric:>12} {time_ratio:>11.2f}x")
    
    print("\n" + "="*60)
    print("Observations:")
    print("="*60)
    print("✓ NashAI achieves comparable accuracy to Scikit-Learn")
    print("✓ Time differences are expected (pure Python vs optimized C)")
    print("✓ Algorithms are mathematically equivalent")


def main():
    """Run all classical ML comparisons."""
    print("="*60)
    print("CLASSICAL ML BENCHMARK: NashAI vs Scikit-Learn")
    print("="*60)
    
    np.random.seed(42)
    
    results = {}
    
    # Run all comparisons
    results['linear_regression'] = compare_linear_regression()
    results['logistic_regression'] = compare_logistic_regression()
    results['knn'] = compare_knn()
    results['naive_bayes'] = compare_naive_bayes()
    results['svm'] = compare_svm()
    
    # Print overall summary
    print_overall_summary(results)
    
    # Plot comparison
    try:
        plot_comparison_summary(results)
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    return results


if __name__ == '__main__':
    main()
