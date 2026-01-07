"""
Examples demonstrating all functions in the linalg module.

This script shows how to use the Vector/Matrix class and all its methods
with clear examples and outputs.
"""

import sys
import os

# Add the math directory to the path so we can import linalg
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'math'))

from math.linalg import Vector, Matrix, EigenDecomposition


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    """Run all examples."""
    
    print_section("1. CREATING VECTORS AND MATRICES")
    
    # Create vectors from lists
    print("Creating vectors from lists:")
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1.shape = {v1.shape}")
    print(f"v2.shape = {v2.shape}")
    
    # Create matrices
    print("\nCreating matrices:")
    m1 = Vector([[1, 2], [3, 4]])
    m2 = Vector([[5, 6], [7, 8]])
    print(f"m1 = \n{m1}")
    print(f"m2 = \n{m2}")
    print(f"m1.shape = {m1.shape}")
    
    # Using class methods
    print("\nUsing class methods to create special matrices:")
    zeros = Vector.zeros((3, 3))
    ones = Vector.ones((2, 2))
    identity = Vector.eye(3)
    random_vec = Vector.random((3,), seed=42)
    
    print(f"zeros(3x3) = \n{zeros}")
    print(f"ones(2x2) = \n{ones}")
    print(f"identity(3x3) = \n{identity}")
    print(f"random vector (seed=42) = {random_vec}")
    
    
    print_section("2. ADDITION OPERATIONS")
    
    # Vector addition
    print("Vector addition:")
    v3 = v1.add(v2)
    print(f"v1 + v2 = {v1} + {v2} = {v3}")
    
    # Using + operator
    v4 = v1 + v2
    print(f"Using + operator: v1 + v2 = {v4}")
    
    # Matrix addition
    print("\nMatrix addition:")
    m3 = m1.add(m2)
    print(f"m1 + m2 = \n{m1}\n+\n{m2}\n=\n{m3}")
    
    
    print_section("3. MULTIPLICATION OPERATIONS")
    
    # Element-wise multiplication
    print("Element-wise multiplication (Hadamard product):")
    v5 = v1.multiply(v2)
    print(f"v1 * v2 (element-wise) = {v1} * {v2} = {v5}")
    
    # Using * operator for element-wise
    v6 = v1 * v2
    print(f"Using * operator: v1 * v2 = {v6}")
    
    # Scalar multiplication
    print("\nScalar multiplication:")
    v7 = v1 * 3
    print(f"v1 * 3 = {v1} * 3 = {v7}")
    
    # Matrix element-wise multiplication
    print("\nMatrix element-wise multiplication:")
    m4 = m1.multiply(m2)
    print(f"m1 * m2 (element-wise) = \n{m1}\n*\n{m2}\n=\n{m4}")
    
    
    print_section("4. DOT PRODUCT OPERATIONS")
    
    # Vector dot product
    print("Vector dot product (returns scalar):")
    dot_result = v1.dot(v2)
    print(f"v1 · v2 = {v1} · {v2} = {dot_result}")
    print(f"Type: {type(dot_result)}")
    
    # Matrix multiplication
    print("\nMatrix multiplication:")
    m5 = m1.dot(m2)
    print(f"m1 @ m2 = \n{m1}\n@\n{m2}\n=\n{m5}")
    
    # Vector-matrix multiplication
    print("\nVector-matrix multiplication:")
    v_vec = Vector([1, 2])
    m_small = Vector([[1, 2, 3], [4, 5, 6]])
    result = v_vec.dot(m_small)
    print(f"v = {v_vec}")
    print(f"m = \n{m_small}")
    print(f"v @ m = {result}")
    
    # Matrix-vector multiplication
    print("\nMatrix-vector multiplication:")
    m_small2 = Vector([[1, 2], [3, 4], [5, 6]])
    v_vec2 = Vector([1, 2])
    result2 = m_small2.dot(v_vec2)
    print(f"m = \n{m_small2}")
    print(f"v = {v_vec2}")
    print(f"m @ v = {result2}")
    
    
    print_section("5. OUTER PRODUCT (EXTERNAL MULTIPLY)")
    
    # Outer product
    print("Outer product of two vectors:")
    v_a = Vector([1, 2, 3])
    v_b = Vector([4, 5])
    outer = v_a.external_multiply(v_b)
    print(f"v_a = {v_a}")
    print(f"v_b = {v_b}")
    print(f"Outer product (v_a ⊗ v_b) = \n{outer}")
    print(f"Result shape: {outer.shape}")
    
    
    print_section("6. NORM CALCULATIONS")
    
    # Vector norms
    print("Vector norms:")
    v_norm = Vector([3, 4])
    print(f"Vector: {v_norm}")
    print(f"L1 norm (Manhattan): ||v||₁ = {v_norm.norm_l1()}")
    print(f"L2 norm (Euclidean): ||v||₂ = {v_norm.norm_l2()}")
    print(f"Infinity norm: ||v||∞ = {v_norm.norm_inf()}")
    print(f"Frobenius norm (same as L2 for vectors): ||v||F = {v_norm.norm_frobenius()}")
    
    # General p-norm
    print(f"\nGeneral p-norm examples:")
    print(f"p=1: {v_norm.norm(p=1)}")
    print(f"p=2: {v_norm.norm(p=2)}")
    print(f"p=3: {v_norm.norm(p=3)}")
    
    # Matrix norms
    print("\nMatrix norms:")
    m_norm = Vector([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Matrix:\n{m_norm}")
    print(f"Frobenius norm: ||M||F = {m_norm.norm_frobenius():.4f}")
    print(f"L1 norm (max column sum): ||M||₁ = {m_norm.norm_l1()}")
    print(f"Infinity norm (max row sum): ||M||∞ = {m_norm.norm_inf()}")
    
    
    print_section("7. EIGENVALUE DECOMPOSITION")
    
    # Create a symmetric matrix for eigendecomposition
    print("Eigenvalue decomposition of a symmetric matrix:")
    A = Vector([[2, 1], [1, 2]])
    print(f"Matrix A:\n{A}")
    
    # Full eigendecomposition
    eigen_decomp = A.eig()
    print(f"\nEigenvalues: {eigen_decomp.eigenvalues}")
    print(f"Eigenvectors (each column is an eigenvector):\n{eigen_decomp.eigenvectors}")
    
    # Verify: A @ v = λ * v
    print("\nVerification (A @ v = λ * v):")
    for i, (eigenval, eigenvec) in enumerate(zip(eigen_decomp.eigenvalues, eigen_decomp.eigenvectors.T)):
        eigenvec_vec = Vector(eigenvec)
        left_side = A.dot(eigenvec_vec)
        right_side = eigenvec_vec * eigenval
        print(f"Eigenvalue {i+1} (λ={eigenval:.4f}):")
        print(f"  A @ v = {left_side}")
        print(f"  λ * v = {right_side}")
        print(f"  Match: {np.allclose(left_side.to_numpy(), right_side.to_numpy())}")
    
    # Eigenvalues only
    print("\nEigenvalues only (more efficient):")
    eigenvals = A.eigvals()
    print(f"Eigenvalues: {eigenvals}")
    
    # Another example with a 3x3 matrix
    print("\nEigenvalue decomposition of a 3x3 matrix:")
    B = Vector([[4, 2, 1], [2, 4, 2], [1, 2, 4]])
    print(f"Matrix B:\n{B}")
    eigen_decomp_B = B.eig()
    print(f"Eigenvalues: {eigen_decomp_B.eigenvalues}")
    print(f"Eigenvectors:\n{eigen_decomp_B.eigenvectors}")
    
    
    print_section("8. COMPREHENSIVE EXAMPLE: SOLVING A LINEAR SYSTEM")
    
    print("Example: Using matrix operations to understand a linear transformation")
    # Create a transformation matrix
    T = Vector([[1, 1], [0, 1]])  # Shear transformation
    v_input = Vector([1, 0])
    
    print(f"Transformation matrix T:\n{T}")
    print(f"Input vector: {v_input}")
    
    # Apply transformation
    v_output = T.dot(v_input)
    print(f"Output vector (T @ v): {v_output}")
    
    # Check eigenvalues to understand the transformation
    eigen_decomp_T = T.eig()
    print(f"\nEigenvalues of T: {eigen_decomp_T.eigenvalues}")
    print(f"Eigenvectors of T:\n{eigen_decomp_T.eigenvectors}")
    
    # Compute norms
    print(f"\nNorms:")
    print(f"Input vector L2 norm: {v_input.norm_l2():.4f}")
    print(f"Output vector L2 norm: {v_output.norm_l2():.4f}")
    print(f"Transformation matrix Frobenius norm: {T.norm_frobenius():.4f}")
    
    
    print_section("9. UTILITY METHODS")
    
    # Convert to numpy
    print("Converting to numpy array:")
    v_util = Vector([1, 2, 3])
    numpy_array = v_util.to_numpy()
    print(f"Vector: {v_util}")
    print(f"As numpy array: {numpy_array}")
    print(f"Type: {type(numpy_array)}")
    
    # String representations
    print("\nString representations:")
    print(f"__repr__: {repr(v_util)}")
    print(f"__str__: {str(v_util)}")
    
    
    print_section("10. ERROR HANDLING EXAMPLES")
    
    print("Attempting operations with incompatible shapes:")
    
    # Incompatible addition
    try:
        v_err1 = Vector([1, 2])
        v_err2 = Vector([1, 2, 3])
        result = v_err1 + v_err2
    except ValueError as e:
        print(f"✓ Caught error for incompatible addition: {e}")
    
    # Incompatible dot product
    try:
        m_err1 = Vector([[1, 2], [3, 4]])
        m_err2 = Vector([[1, 2, 3], [4, 5, 6]])
        result = m_err1.dot(m_err2)
    except ValueError as e:
        print(f"✓ Caught error for incompatible matrix multiplication: {e}")
    
    # Eigendecomposition on non-square matrix
    try:
        m_non_square = Vector([[1, 2, 3], [4, 5, 6]])
        eigen = m_non_square.eig()
    except ValueError as e:
        print(f"✓ Caught error for eigendecomposition on non-square matrix: {e}")
    
    # Outer product on matrices
    try:
        m1_err = Vector([[1, 2], [3, 4]])
        m2_err = Vector([[5, 6], [7, 8]])
        result = m1_err.external_multiply(m2_err)
    except ValueError as e:
        print(f"✓ Caught error for outer product on matrices: {e}")
    
    
    print_section("EXAMPLES COMPLETE!")
    print("All functions in the linalg module have been demonstrated.")
    print("You can use these examples as a reference for your own code.")


if __name__ == "__main__":
    main()

