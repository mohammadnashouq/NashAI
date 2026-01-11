"""
Examples demonstrating all functions in the decomposition module.

This script shows how to use Principal Component Analysis (PCA) and
Independent Component Analysis (ICA) with clear examples and outputs.
"""

import numpy as np
import sys
import os

# Add parent directory to path for core_math imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.decomposition import PCA, ICA


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    """Run all examples."""
    
    print_section("1. PRINCIPAL COMPONENT ANALYSIS (PCA)")
    
    # Example 1: Simple 2D to 1D reduction
    print("Example 1: Reducing 2D data to 1D")
    np.random.seed(42)
    
    # Create data with clear principal direction
    X_2d = np.random.randn(100, 2)
    # Make second dimension correlated with first
    X_2d[:, 1] = 0.8 * X_2d[:, 0] + 0.2 * X_2d[:, 1]
    
    print(f"Original data:")
    print(f"  Shape: {X_2d.shape}")
    print(f"  Variance along each axis: {np.var(X_2d, axis=0)}")
    print(f"  Correlation: {np.corrcoef(X_2d.T)[0, 1]:.4f}")
    
    # Fit PCA
    pca_2d = PCA(n_components=1)
    pca_2d.fit(X_2d)
    
    print(f"\nFitted PCA:")
    print(f"  Components shape: {pca_2d.components_.shape}")
    print(f"  First principal component: {pca_2d.components_[0]}")
    print(f"  Explained variance: {pca_2d.explained_variance_[0]:.4f}")
    print(f"  Explained variance ratio: {pca_2d.explained_variance_ratio_[0]:.4f}")
    
    # Transform data
    X_transformed = pca_2d.transform(X_2d)
    print(f"\nTransformed data:")
    print(f"  Shape: {X_transformed.shape}")
    print(f"  Variance: {np.var(X_transformed):.4f}")
    print(f"  First 5 values: {X_transformed[:5].flatten()}")
    
    # Reconstruct
    X_reconstructed = pca_2d.inverse_transform(X_transformed)
    reconstruction_error = np.mean((X_2d - X_reconstructed)**2)
    print(f"\nReconstruction:")
    print(f"  Reconstruction error (MSE): {reconstruction_error:.4f}")
    print(f"  Original variance: {np.var(X_2d):.4f}")
    
    # Example 2: Higher dimensional data
    print("\n" + "-" * 70)
    print("Example 2: Reducing 5D data to 2D")
    np.random.seed(42)
    
    # Create 5D data with 2 main directions
    X_5d = np.random.randn(200, 5)
    # Make some dimensions correlated
    X_5d[:, 2] = 0.7 * X_5d[:, 0] + 0.3 * X_5d[:, 2]
    X_5d[:, 3] = 0.6 * X_5d[:, 1] + 0.4 * X_5d[:, 3]
    
    print(f"Original data:")
    print(f"  Shape: {X_5d.shape}")
    print(f"  Variance along each axis: {np.var(X_5d, axis=0)}")
    
    # Fit PCA with 2 components
    pca_5d = PCA(n_components=2)
    pca_5d.fit(X_5d)
    
    print(f"\nFitted PCA (2 components):")
    print(f"  Components shape: {pca_5d.components_.shape}")
    print(f"  Explained variance: {pca_5d.explained_variance_}")
    print(f"  Explained variance ratio: {pca_5d.explained_variance_ratio_}")
    print(f"  Total variance explained: {np.sum(pca_5d.explained_variance_ratio_):.4f}")
    
    # Transform
    X_transformed_5d = pca_5d.transform(X_5d)
    print(f"\nTransformed data:")
    print(f"  Shape: {X_transformed_5d.shape}")
    print(f"  Variance along each PC: {np.var(X_transformed_5d, axis=0)}")
    
    # Example 3: Choosing number of components
    print("\n" + "-" * 70)
    print("Example 3: Explained variance for different numbers of components")
    pca_full = PCA()  # Keep all components
    pca_full.fit(X_5d)
    
    print(f"Explained variance ratio for all components:")
    for i, ratio in enumerate(pca_full.explained_variance_ratio_):
        print(f"  PC {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
    
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    print(f"\nCumulative explained variance:")
    for i, cum in enumerate(cumulative):
        print(f"  First {i+1} components: {cum:.4f} ({cum*100:.2f}%)")
    
    # Example 4: Visualization concept (showing how PCA finds directions)
    print("\n" + "-" * 70)
    print("Example 4: PCA finds directions of maximum variance")
    np.random.seed(42)
    
    # Create data with clear structure
    angle = np.pi / 4  # 45 degrees
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    X_rotated = np.random.randn(100, 2) @ rotation.T
    X_rotated[:, 0] *= 3  # Stretch along first axis
    
    pca_rot = PCA(n_components=2)
    pca_rot.fit(X_rotated)
    
    print(f"Data variance along original axes: {np.var(X_rotated, axis=0)}")
    print(f"Principal components:")
    for i, pc in enumerate(pca_rot.components_):
        print(f"  PC{i+1}: {pc} (explains {pca_rot.explained_variance_ratio_[i]:.4f} of variance)")
    
    
    print_section("2. INDEPENDENT COMPONENT ANALYSIS (ICA)")
    
    # Example 1: Simple signal separation
    print("Example 1: Separating mixed signals (cocktail party problem)")
    np.random.seed(42)
    
    # Create two independent source signals
    t = np.linspace(0, 10, 1000)
    source1 = np.sin(2 * np.pi * 0.5 * t)  # Low frequency
    source2 = np.sign(np.sin(2 * np.pi * 2 * t))  # Square wave
    
    sources = np.column_stack([source1, source2])
    
    print(f"Source signals:")
    print(f"  Shape: {sources.shape}")
    print(f"  Source 1: sin wave, range=[{source1.min():.2f}, {source1.max():.2f}]")
    print(f"  Source 2: square wave, range=[{source2.min():.2f}, {source2.max():.2f}]")
    
    # Mix the signals
    mixing_matrix = np.array([[0.6, 0.4],
                              [0.4, 0.6]])
    mixed_signals = sources @ mixing_matrix.T
    
    print(f"\nMixed signals:")
    print(f"  Shape: {mixed_signals.shape}")
    print(f"  Mixing matrix:\n{mixing_matrix}")
    
    # Apply ICA
    ica = ICA(n_components=2, max_iterations=200, random_state=42)
    ica.fit(mixed_signals)
    
    print(f"\nFitted ICA:")
    print(f"  Components shape: {ica.components_.shape}")
    print(f"  Unmixing matrix (components):")
    print(f"    {ica.components_}")
    print(f"  Mixing matrix (estimated):")
    print(f"    {ica.mixing_}")
    
    # Separate signals
    separated = ica.transform(mixed_signals)
    print(f"\nSeparated signals:")
    print(f"  Shape: {separated.shape}")
    print(f"  Separated signal 1: range=[{separated[:, 0].min():.2f}, {separated[:, 0].max():.2f}]")
    print(f"  Separated signal 2: range=[{separated[:, 1].min():.2f}, {separated[:, 1].max():.2f}]")
    
    # Note: ICA results may be scaled/flipped compared to original
    print(f"\nNote: ICA results may be scaled or flipped compared to original sources")
    print(f"  This is because ICA finds independent components up to scaling and sign")
    
    # Example 2: Higher dimensional separation
    print("\n" + "-" * 70)
    print("Example 2: Separating 3 independent sources")
    np.random.seed(42)
    
    # Create 3 independent sources
    source1_3d = np.sin(2 * np.pi * 0.3 * t[:500])
    source2_3d = np.random.randn(500) * 0.5  # Noise
    source3_3d = np.sin(2 * np.pi * 1.5 * t[:500])
    
    sources_3d = np.column_stack([source1_3d, source2_3d, source3_3d])
    
    # Mix them
    mixing_3d = np.random.rand(3, 3)
    mixed_3d = sources_3d @ mixing_3d.T
    
    print(f"Source signals shape: {sources_3d.shape}")
    print(f"Mixed signals shape: {mixed_3d.shape}")
    
    # Apply ICA
    ica_3d = ICA(n_components=3, max_iterations=200, random_state=42)
    ica_3d.fit(mixed_3d)
    
    separated_3d = ica_3d.transform(mixed_3d)
    print(f"Separated signals shape: {separated_3d.shape}")
    print(f"ICA found {ica_3d.n_components} independent components")
    
    # Example 3: ICA vs PCA comparison
    print("\n" + "-" * 70)
    print("Example 3: ICA vs PCA - Key Differences")
    
    # Use simple 2D data
    X_ica_comp = mixed_signals[:100]  # Use subset for speed
    
    # PCA
    pca_comp = PCA(n_components=2)
    pca_comp.fit(X_ica_comp)
    X_pca = pca_comp.transform(X_ica_comp)
    
    # ICA
    ica_comp = ICA(n_components=2, max_iterations=100, random_state=42)
    ica_comp.fit(X_ica_comp)
    X_ica = ica_comp.transform(X_ica_comp)
    
    print(f"PCA:")
    print(f"  Finds directions of maximum variance")
    print(f"  Components are orthogonal")
    print(f"  Components are uncorrelated (but may be dependent)")
    print(f"  Explained variance: {pca_comp.explained_variance_ratio_}")
    
    print(f"\nICA:")
    print(f"  Finds statistically independent components")
    print(f"  Components are not necessarily orthogonal")
    print(f"  Components are independent (stronger than uncorrelated)")
    print(f"  Order of components is arbitrary")
    
    # Check independence (correlation should be low for both)
    corr_pca = np.corrcoef(X_pca.T)[0, 1]
    corr_ica = np.corrcoef(X_ica.T)[0, 1]
    print(f"\nCorrelation between components:")
    print(f"  PCA components: {corr_pca:.4f}")
    print(f"  ICA components: {corr_ica:.4f}")
    
    # Example 4: Reconstruction
    print("\n" + "-" * 70)
    print("Example 4: Reconstruction with ICA")
    
    # Transform and inverse transform
    X_ica_transformed = ica_comp.transform(X_ica_comp)
    X_ica_reconstructed = ica_comp.inverse_transform(X_ica_transformed)
    
    reconstruction_error_ica = np.mean((X_ica_comp - X_ica_reconstructed)**2)
    print(f"Reconstruction error (MSE): {reconstruction_error_ica:.4f}")
    print(f"Original variance: {np.var(X_ica_comp):.4f}")
    
    
    print_section("3. COMPARISON: PCA vs ICA")
    
    print("When to use PCA:")
    print("  - Dimensionality reduction")
    print("  - Data compression")
    print("  - Visualization (reduce to 2D/3D)")
    print("  - Noise reduction")
    print("  - When you want orthogonal components")
    print()
    print("When to use ICA:")
    print("  - Blind source separation")
    print("  - Signal processing (separating mixed signals)")
    print("  - Feature extraction (finding independent features)")
    print("  - When statistical independence is important")
    print("  - When sources are non-Gaussian")
    print()
    print("Key differences:")
    print("  - PCA: orthogonal, uncorrelated, maximizes variance")
    print("  - ICA: independent, not necessarily orthogonal, maximizes independence")
    print("  - PCA: deterministic ordering (by variance)")
    print("  - ICA: arbitrary ordering")
    
    
    print_section("EXAMPLES COMPLETE!")
    print("All decomposition algorithms have been demonstrated.")
    print("You can use these examples as a reference for your own dimensionality reduction tasks.")


if __name__ == "__main__":
    main()
