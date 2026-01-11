"""
Examples demonstrating all functions in the clustering module.

This script shows how to use k-Means and Gaussian Mixture Model (GMM)
with clear examples and outputs.
"""

import numpy as np
import sys
import os

# Add parent directory to path for core_math imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.clustering import KMeans, GaussianMixtureModel


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    """Run all examples."""
    
    print_section("1. k-MEANS CLUSTERING")
    
    # Example 1: Simple 2D clustering
    print("Example 1: Clustering 2D data into 3 clusters")
    np.random.seed(42)
    
    # Generate three distinct clusters
    cluster1 = np.random.randn(50, 2) + [2, 2]
    cluster2 = np.random.randn(50, 2) + [-2, -2]
    cluster3 = np.random.randn(50, 2) + [0, 3]
    X_kmeans = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"Data:")
    print(f"  X shape: {X_kmeans.shape}")
    print(f"  Data range: X1=[{X_kmeans[:, 0].min():.2f}, {X_kmeans[:, 0].max():.2f}], "
          f"X2=[{X_kmeans[:, 1].min():.2f}, {X_kmeans[:, 1].max():.2f}]")
    
    # Fit k-means
    kmeans = KMeans(n_clusters=3, max_iterations=100, random_state=42, init='random')
    kmeans.fit(X_kmeans)
    
    print(f"\nFitted model:")
    print(f"  Centroids:")
    for i, centroid in enumerate(kmeans.centroids_):
        print(f"    Cluster {i}: {centroid}")
    print(f"  Inertia (within-cluster sum of squares): {kmeans.inertia_:.4f}")
    
    # Cluster assignments
    labels = kmeans.labels_
    print(f"\nCluster assignments:")
    for i in range(3):
        count = np.sum(labels == i)
        print(f"  Cluster {i}: {count} points")
    
    # Predict new points
    X_test_kmeans = np.array([[2.0, 2.0], [-2.0, -2.0], [0.0, 3.0], [0.0, 0.0]])
    test_labels = kmeans.predict(X_test_kmeans)
    print(f"\nPredictions for test points:")
    for x, label in zip(X_test_kmeans, test_labels):
        print(f"  Point {x}: assigned to cluster {label}")
    
    # Example 2: Different k values
    print("\n" + "-" * 70)
    print("Example 2: Effect of different k values")
    for k in [2, 3, 4, 5]:
        kmeans_k = KMeans(n_clusters=k, max_iterations=100, random_state=42)
        kmeans_k.fit(X_kmeans)
        print(f"  k={k}: Inertia={kmeans_k.inertia_:.4f}")
    
    # Example 3: k-means++ initialization
    print("\n" + "-" * 70)
    print("Example 3: k-means++ initialization (better than random)")
    kmeans_plus = KMeans(n_clusters=3, max_iterations=100, random_state=42, init='kmeans++')
    kmeans_plus.fit(X_kmeans)
    print(f"  k-means++ inertia: {kmeans_plus.inertia_:.4f}")
    print(f"  Random init inertia: {kmeans.inertia_:.4f}")
    print(f"  Improvement: {kmeans.inertia_ - kmeans_plus.inertia_:.4f}")
    
    # Example 4: Higher dimensional data
    print("\n" + "-" * 70)
    print("Example 4: Clustering 3D data")
    np.random.seed(42)
    X_3d = np.vstack([
        np.random.randn(30, 3) + [1, 1, 1],
        np.random.randn(30, 3) + [-1, -1, -1],
        np.random.randn(30, 3) + [0, 2, -1]
    ])
    
    kmeans_3d = KMeans(n_clusters=3, random_state=42)
    kmeans_3d.fit(X_3d)
    
    print(f"  3D data shape: {X_3d.shape}")
    print(f"  Centroids shape: {kmeans_3d.centroids_.shape}")
    print(f"  Inertia: {kmeans_3d.inertia_:.4f}")
    
    
    print_section("2. GAUSSIAN MIXTURE MODEL (GMM)")
    
    # Example 1: 2D GMM
    print("Example 1: GMM clustering on 2D data")
    np.random.seed(42)
    
    # Use same data as k-means example
    X_gmm = X_kmeans
    
    print(f"Data:")
    print(f"  X shape: {X_gmm.shape}")
    
    # Fit GMM
    gmm = GaussianMixtureModel(
        n_components=3,
        max_iterations=100,
        tolerance=1e-3,
        random_state=42
    )
    gmm.fit(X_gmm)
    
    print(f"\nFitted model:")
    print(f"  Number of components: {gmm.n_components}")
    print(f"  Component weights: {gmm.weights_}")
    print(f"  Component means:")
    for i, mean in enumerate(gmm.means_):
        print(f"    Component {i}: {mean}")
    print(f"  Component covariances (diagonal elements):")
    for i, cov in enumerate(gmm.covariances_):
        print(f"    Component {i}: diag={np.diag(cov)}")
    
    # Hard clustering (assign to most likely component)
    labels_gmm = gmm.predict(X_gmm)
    print(f"\nHard cluster assignments:")
    for i in range(3):
        count = np.sum(labels_gmm == i)
        print(f"  Component {i}: {count} points")
    
    # Soft clustering (probabilities)
    probabilities = gmm.predict_proba(X_gmm)
    print(f"\nSoft clustering (probabilities) for first 5 points:")
    for i in range(5):
        print(f"  Point {X_gmm[i]}: probabilities={probabilities[i]}")
    
    # Predict new points
    X_test_gmm = np.array([[2.0, 2.0], [-2.0, -2.0], [0.0, 3.0], [0.0, 0.0]])
    test_labels_gmm = gmm.predict(X_test_gmm)
    test_proba_gmm = gmm.predict_proba(X_test_gmm)
    
    print(f"\nPredictions for test points:")
    for x, label, proba in zip(X_test_gmm, test_labels_gmm, test_proba_gmm):
        print(f"  Point {x}: component={label}, probabilities={proba}")
    
    # Example 2: Different number of components
    print("\n" + "-" * 70)
    print("Example 2: Effect of different number of components")
    for n_comp in [2, 3, 4, 5]:
        gmm_n = GaussianMixtureModel(
            n_components=n_comp,
            max_iterations=50,
            random_state=42
        )
        gmm_n.fit(X_gmm)
        print(f"  n_components={n_comp}: weights={gmm_n.weights_}")
    
    # Example 3: Compare k-means vs GMM
    print("\n" + "-" * 70)
    print("Example 3: Comparison of k-Means and GMM")
    
    # Check how many points are assigned to same cluster
    labels_kmeans = kmeans.labels_
    labels_gmm_hard = gmm.predict(X_gmm)
    
    # Create confusion matrix
    confusion = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            confusion[i, j] = np.sum((labels_kmeans == i) & (labels_gmm_hard == j))
    
    print(f"  Confusion matrix (k-means rows, GMM columns):")
    print(f"    {confusion}")
    
    # Simple agreement (without scipy)
    # Count how many points are in the same cluster (with best matching)
    max_matched = 0
    for perm in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
        matched = sum(confusion[i, perm[i]] for i in range(3))
        max_matched = max(max_matched, matched)
    
    agreement = max_matched / len(X_gmm)
    print(f"  Agreement (best matching): {agreement:.4f}")
    
    # Example 4: GMM on 1D data (easier to visualize concept)
    print("\n" + "-" * 70)
    print("Example 4: GMM on 1D data (two overlapping Gaussians)")
    np.random.seed(42)
    X_1d = np.hstack([
        np.random.randn(100) * 0.5 + 0,  # Mean 0
        np.random.randn(100) * 0.5 + 2   # Mean 2
    ]).reshape(-1, 1)
    
    gmm_1d = GaussianMixtureModel(n_components=2, random_state=42)
    gmm_1d.fit(X_1d)
    
    print(f"  1D data shape: {X_1d.shape}")
    print(f"  Component weights: {gmm_1d.weights_}")
    print(f"  Component means: {gmm_1d.means_.flatten()}")
    print(f"  Component stds: {np.sqrt(np.diag(gmm_1d.covariances_[0]))[0]:.4f}, "
          f"{np.sqrt(np.diag(gmm_1d.covariances_[1]))[0]:.4f}")
    
    
    print_section("3. COMPARISON: k-MEANS vs GMM")
    
    print("Key differences:")
    print("  k-Means:")
    print("    - Hard clustering (each point belongs to one cluster)")
    print("    - Assumes spherical clusters of equal size")
    print("    - Faster and simpler")
    print("    - Works well when clusters are well-separated")
    print()
    print("  GMM:")
    print("    - Soft clustering (probabilistic assignments)")
    print("    - Can model elliptical clusters of different sizes")
    print("    - More flexible but slower")
    print("    - Better for overlapping clusters")
    print()
    print("  When to use:")
    print("    - Use k-Means for: speed, well-separated clusters, hard assignments")
    print("    - Use GMM for: overlapping clusters, need probabilities, elliptical clusters")
    
    
    print_section("EXAMPLES COMPLETE!")
    print("All clustering algorithms have been demonstrated.")
    print("You can use these examples as a reference for your own clustering tasks.")


if __name__ == "__main__":
    main()
