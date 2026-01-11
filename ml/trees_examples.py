"""
Examples demonstrating all functions in the trees module.

This script shows how to use Decision Trees and Random Forest
(classification and regression) with clear examples and outputs.
"""

import numpy as np
import sys
import os

# Add parent directory to path for core_math imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    """Run all examples."""
    
    print_section("1. DECISION TREE CLASSIFIER")
    
    # Example 1: Simple classification
    print("Example 1: Binary classification with clear decision boundaries")
    np.random.seed(42)
    
    # Create dataset: if x1 + x2 > 2, class 1, else class 0
    X_tree = np.random.rand(100, 2) * 4 - 2  # Range [-2, 2]
    y_tree = ((X_tree[:, 0] + X_tree[:, 1]) > 2).astype(int)
    
    print(f"Training data:")
    print(f"  X shape: {X_tree.shape}")
    print(f"  Class distribution: {np.bincount(y_tree)}")
    print(f"  First 5 samples:")
    for i in range(5):
        print(f"    X={X_tree[i]}, y={y_tree[i]}")
    
    # Fit with different criteria
    for criterion in ['gini', 'entropy']:
        dt_clf = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            criterion=criterion
        )
        dt_clf.fit(X_tree, y_tree)
        
        # Predictions
        X_test_tree = np.array([[1.5, 1.5], [-1.0, -1.0], [2.0, 2.0]])
        y_pred_tree = dt_clf.predict(X_test_tree)
        
        # Accuracy
        train_pred = dt_clf.predict(X_tree)
        accuracy = np.mean(train_pred == y_tree)
        
        print(f"\n{criterion.capitalize()} criterion:")
        print(f"  Test predictions: {y_pred_tree}")
        print(f"  Training accuracy: {accuracy:.4f}")
    
    # Example 2: Multi-class classification
    print("\n" + "-" * 70)
    print("Example 2: Multi-class classification (3 classes)")
    np.random.seed(42)
    
    # Create three classes in different regions
    X_multi = np.vstack([
        np.random.randn(30, 2) + [2, 2],   # Class 0
        np.random.randn(30, 2) + [-2, -2], # Class 1
        np.random.randn(30, 2) + [0, 0]    # Class 2
    ])
    y_multi = np.hstack([np.zeros(30), np.ones(30), np.full(30, 2)])
    
    dt_multi = DecisionTreeClassifier(max_depth=10, min_samples_split=3)
    dt_multi.fit(X_multi, y_multi)
    
    X_test_multi = np.array([[2.0, 2.0], [-2.0, -2.0], [0.0, 0.0], [1.0, 1.0]])
    y_pred_multi = dt_multi.predict(X_test_multi)
    
    print(f"Test predictions: {y_pred_multi}")
    train_pred_multi = dt_multi.predict(X_multi)
    accuracy_multi = np.mean(train_pred_multi == y_multi)
    print(f"Training accuracy: {accuracy_multi:.4f}")
    
    
    print_section("2. DECISION TREE REGRESSOR")
    
    # Example: Regression
    print("Example: Regression tree for y = x1^2 + x2^2 + noise")
    np.random.seed(42)
    
    X_reg = np.random.rand(100, 2) * 4 - 2  # Range [-2, 2]
    y_reg = X_reg[:, 0]**2 + X_reg[:, 1]**2 + 0.1 * np.random.randn(100)
    
    print(f"Training data:")
    print(f"  X shape: {X_reg.shape}")
    print(f"  y range: [{y_reg.min():.2f}, {y_reg.max():.2f}]")
    print(f"  First 5 samples:")
    for i in range(5):
        print(f"    X={X_reg[i]}, y={y_reg[i]:.4f}")
    
    # Fit model
    dt_reg = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2
    )
    dt_reg.fit(X_reg, y_reg)
    
    # Predictions
    X_test_reg = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [-1.0, -1.0]])
    y_pred_reg = dt_reg.predict(X_test_reg)
    y_true_reg = X_test_reg[:, 0]**2 + X_test_reg[:, 1]**2
    
    print(f"\nTest predictions:")
    for i, (x, pred, true_val) in enumerate(zip(X_test_reg, y_pred_reg, y_true_reg)):
        print(f"  X={x}: predicted={pred:.4f}, actual={true_val:.4f}, error={abs(pred-true_val):.4f}")
    
    # Mean squared error
    train_pred_reg = dt_reg.predict(X_reg)
    mse = np.mean((train_pred_reg - y_reg)**2)
    print(f"\nTraining MSE: {mse:.4f}")
    
    # Example with different max_depth
    print("\n" + "-" * 70)
    print("Example: Effect of max_depth on regression")
    for depth in [3, 5, 10, None]:
        dt_depth = DecisionTreeRegressor(max_depth=depth, min_samples_split=5)
        dt_depth.fit(X_reg, y_reg)
        train_pred_depth = dt_depth.predict(X_reg)
        mse_depth = np.mean((train_pred_depth - y_reg)**2)
        print(f"  max_depth={depth}: Training MSE={mse_depth:.4f}")
    
    
    print_section("3. RANDOM FOREST CLASSIFIER")
    
    # Example: Classification with ensemble
    print("Example: Random Forest for binary classification")
    np.random.seed(42)
    
    # Use same data as decision tree example
    X_rf = X_tree
    y_rf = y_tree
    
    print(f"Training data:")
    print(f"  X shape: {X_rf.shape}")
    print(f"  Class distribution: {np.bincount(y_rf)}")
    
    # Fit with different number of trees
    for n_trees in [10, 50, 100]:
        rf_clf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=5,
            max_features='sqrt',
            random_state=42
        )
        rf_clf.fit(X_rf, y_rf)
        
        # Predictions
        X_test_rf = np.array([[1.5, 1.5], [-1.0, -1.0], [2.0, 2.0]])
        y_pred_rf = rf_clf.predict(X_test_rf)
        
        # Accuracy
        train_pred_rf = rf_clf.predict(X_rf)
        accuracy_rf = np.mean(train_pred_rf == y_rf)
        
        print(f"\n{n_trees} trees:")
        print(f"  Test predictions: {y_pred_rf}")
        print(f"  Training accuracy: {accuracy_rf:.4f}")
    
    # Example: Multi-class
    print("\n" + "-" * 70)
    print("Example: Random Forest for multi-class classification")
    rf_multi = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        max_features='sqrt',
        random_state=42
    )
    rf_multi.fit(X_multi, y_multi)
    
    y_pred_rf_multi = rf_multi.predict(X_test_multi)
    print(f"Test predictions: {y_pred_rf_multi}")
    train_pred_rf_multi = rf_multi.predict(X_multi)
    accuracy_rf_multi = np.mean(train_pred_rf_multi == y_multi)
    print(f"Training accuracy: {accuracy_rf_multi:.4f}")
    
    # Compare with single decision tree
    print(f"\nComparison:")
    print(f"  Single Decision Tree accuracy: {accuracy_multi:.4f}")
    print(f"  Random Forest accuracy: {accuracy_rf_multi:.4f}")
    print(f"  Improvement: {accuracy_rf_multi - accuracy_multi:.4f}")
    
    
    print_section("4. RANDOM FOREST REGRESSOR")
    
    # Example: Regression with ensemble
    print("Example: Random Forest for regression")
    np.random.seed(42)
    
    # Use same data as decision tree regressor example
    X_rf_reg = X_reg
    y_rf_reg = y_reg
    
    print(f"Training data:")
    print(f"  X shape: {X_rf_reg.shape}")
    print(f"  y range: [{y_rf_reg.min():.2f}, {y_rf_reg.max():.2f}]")
    
    # Fit with different number of trees
    for n_trees in [10, 50, 100]:
        rf_reg = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=5,
            max_features='sqrt',
            random_state=42
        )
        rf_reg.fit(X_rf_reg, y_rf_reg)
        
        # Predictions
        y_pred_rf_reg = rf_reg.predict(X_test_reg)
        
        # MSE
        train_pred_rf_reg = rf_reg.predict(X_rf_reg)
        mse_rf = np.mean((train_pred_rf_reg - y_rf_reg)**2)
        
        # Test MSE
        test_mse_rf = np.mean((y_pred_rf_reg - y_true_reg)**2)
        
        print(f"\n{n_trees} trees:")
        print(f"  Training MSE: {mse_rf:.4f}")
        print(f"  Test MSE: {test_mse_rf:.4f}")
    
    # Compare with single decision tree
    print(f"\nComparison:")
    print(f"  Single Decision Tree MSE: {mse:.4f}")
    print(f"  Random Forest (100 trees) MSE: {mse_rf:.4f}")
    print(f"  Improvement: {mse - mse_rf:.4f}")
    
    # Example: Effect of max_features
    print("\n" + "-" * 70)
    print("Example: Effect of max_features parameter")
    for max_feat in ['sqrt', 1, 2]:
        rf_feat = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            max_features=max_feat,
            random_state=42
        )
        rf_feat.fit(X_rf_reg, y_rf_reg)
        train_pred_feat = rf_feat.predict(X_rf_reg)
        mse_feat = np.mean((train_pred_feat - y_rf_reg)**2)
        print(f"  max_features={max_feat}: Training MSE={mse_feat:.4f}")
    
    
    print_section("EXAMPLES COMPLETE!")
    print("All tree-based models have been demonstrated.")
    print("Key observations:")
    print("  - Decision Trees can overfit with high depth")
    print("  - Random Forest reduces overfitting through ensemble averaging")
    print("  - More trees generally improve performance (with diminishing returns)")
    print("  - max_features controls the diversity of trees in the forest")


if __name__ == "__main__":
    main()
