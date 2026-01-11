"""
Examples demonstrating all functions in the optimization module.

This script shows how to use numerical differentiation and gradient descent
optimizers with clear examples and outputs.
"""

import sys
import os
import numpy as np

# Add the current directory to the path so we can import core_math
sys.path.insert(0, os.path.dirname(__file__))

from core_math.optimization import (
    numerical_derivative,
    numerical_gradient,
    numerical_hessian,
    GradientDescent,
    StochasticGradientDescent
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    """Run all examples."""
    
    print_section("1. NUMERICAL DERIVATIVE - SCALAR FUNCTIONS")
    
    # Example 1: Simple polynomial
    print("Example 1: f(x) = x²")
    print("Analytical derivative: f'(x) = 2x")
    
    def f1(x):
        return x ** 2
    
    x_val = 3.0
    analytical_derivative = 2 * x_val
    numerical_forward = numerical_derivative(f1, x_val, h=1e-5, method='forward')
    numerical_backward = numerical_derivative(f1, x_val, h=1e-5, method='backward')
    numerical_central = numerical_derivative(f1, x_val, h=1e-5, method='central')
    
    print(f"At x = {x_val}:")
    print(f"  Analytical: f'({x_val}) = {analytical_derivative}")
    print(f"  Forward difference: {numerical_forward:.10f} (error: {abs(numerical_forward - analytical_derivative):.2e})")
    print(f"  Backward difference: {numerical_backward:.10f} (error: {abs(numerical_backward - analytical_derivative):.2e})")
    print(f"  Central difference: {numerical_central:.10f} (error: {abs(numerical_central - analytical_derivative):.2e})")
    print(f"  → Central difference is most accurate!")
    
    # Example 2: Exponential function
    print("\nExample 2: f(x) = e^x")
    print("Analytical derivative: f'(x) = e^x")
    
    def f2(x):
        return np.exp(x)
    
    x_val = 1.0
    analytical_derivative = np.exp(x_val)
    numerical_central = numerical_derivative(f2, x_val, method='central')
    
    print(f"At x = {x_val}:")
    print(f"  Analytical: f'({x_val}) = {analytical_derivative:.10f}")
    print(f"  Numerical (central): {numerical_central:.10f}")
    print(f"  Error: {abs(numerical_central - analytical_derivative):.2e}")
    
    # Example 3: Trigonometric function
    print("\nExample 3: f(x) = sin(x)")
    print("Analytical derivative: f'(x) = cos(x)")
    
    def f3(x):
        return np.sin(x)
    
    x_val = np.pi / 4
    analytical_derivative = np.cos(x_val)
    numerical_central = numerical_derivative(f3, x_val, method='central')
    
    print(f"At x = π/4 = {x_val:.4f}:")
    print(f"  Analytical: f'({x_val:.4f}) = {analytical_derivative:.10f}")
    print(f"  Numerical (central): {numerical_central:.10f}")
    print(f"  Error: {abs(numerical_central - analytical_derivative):.2e}")
    
    # Example 4: Multiple points
    print("\nExample 4: Computing derivative at multiple points")
    x_points = np.array([0.0, 1.0, 2.0, 3.0])
    derivatives = numerical_derivative(f1, x_points, method='central')
    print(f"f(x) = x² at points {x_points}:")
    for x, deriv in zip(x_points, derivatives):
        print(f"  f'({x}) = {deriv:.6f} (analytical: {2*x:.6f})")
    
    
    print_section("2. NUMERICAL GRADIENT - MULTIVARIATE FUNCTIONS")
    
    # Example 1: Simple quadratic function
    print("Example 1: f(x, y) = x² + y²")
    print("Analytical gradient: ∇f = [2x, 2y]")
    
    def f4(x):
        return x[0]**2 + x[1]**2
    
    x_point = np.array([3.0, 4.0])
    analytical_grad = np.array([2 * x_point[0], 2 * x_point[1]])
    numerical_grad = numerical_gradient(f4, x_point)
    
    print(f"At point x = {x_point}:")
    print(f"  Analytical gradient: {analytical_grad}")
    print(f"  Numerical gradient: {numerical_grad}")
    print(f"  Error: {np.linalg.norm(numerical_grad - analytical_grad):.2e}")
    
    # Example 2: More complex function
    print("\nExample 2: f(x, y) = x²y + y²x + xy")
    print("Analytical gradient: ∇f = [2xy + y² + y, x² + 2yx + x]")
    
    def f5(x):
        return x[0]**2 * x[1] + x[1]**2 * x[0] + x[0] * x[1]
    
    x_point = np.array([2.0, 3.0])
    analytical_grad = np.array([
        2 * x_point[0] * x_point[1] + x_point[1]**2 + x_point[1],
        x_point[0]**2 + 2 * x_point[1] * x_point[0] + x_point[0]
    ])
    numerical_grad = numerical_gradient(f5, x_point)
    
    print(f"At point x = {x_point}:")
    print(f"  Analytical gradient: {analytical_grad}")
    print(f"  Numerical gradient: {numerical_grad}")
    print(f"  Error: {np.linalg.norm(numerical_grad - analytical_grad):.2e}")
    
    # Example 3: 3D function
    print("\nExample 3: f(x, y, z) = x² + y² + z²")
    print("Analytical gradient: ∇f = [2x, 2y, 2z]")
    
    def f6(x):
        return x[0]**2 + x[1]**2 + x[2]**2
    
    x_point = np.array([1.0, 2.0, 3.0])
    analytical_grad = 2 * x_point
    numerical_grad = numerical_gradient(f6, x_point)
    
    print(f"At point x = {x_point}:")
    print(f"  Analytical gradient: {analytical_grad}")
    print(f"  Numerical gradient: {numerical_grad}")
    print(f"  Error: {np.linalg.norm(numerical_grad - analytical_grad):.2e}")
    
    
    print_section("3. NUMERICAL HESSIAN - SECOND DERIVATIVES")
    
    # Example 1: Simple quadratic
    print("Example 1: f(x, y) = x² + y²")
    print("Analytical Hessian: H = [[2, 0], [0, 2]]")
    
    x_point = np.array([1.0, 2.0])
    analytical_hessian = np.array([[2.0, 0.0], [0.0, 2.0]])
    numerical_hess = numerical_hessian(f4, x_point)
    
    print(f"At point x = {x_point}:")
    print(f"  Analytical Hessian:\n{analytical_hessian}")
    print(f"  Numerical Hessian:\n{numerical_hess}")
    print(f"  Error (Frobenius norm): {np.linalg.norm(numerical_hess - analytical_hessian):.2e}")
    
    # Example 2: More complex function
    print("\nExample 2: f(x, y) = x²y + y²x")
    print("Analytical Hessian: H = [[2y, 2x+2y], [2x+2y, 2x]]")
    
    x_point = np.array([2.0, 3.0])
    analytical_hessian = np.array([
        [2 * x_point[1], 2 * x_point[0] + 2 * x_point[1]],
        [2 * x_point[0] + 2 * x_point[1], 2 * x_point[0]]
    ])
    numerical_hess = numerical_hessian(f5, x_point)
    
    print(f"At point x = {x_point}:")
    print(f"  Analytical Hessian:\n{analytical_hessian}")
    print(f"  Numerical Hessian:\n{numerical_hess}")
    print(f"  Error (Frobenius norm): {np.linalg.norm(numerical_hess - analytical_hessian):.2e}")
    
    
    print_section("4. GRADIENT DESCENT - BASIC OPTIMIZATION")
    
    # Example 1: Simple quadratic (convex function)
    print("Example 1: Minimizing f(x, y) = x² + y²")
    print("Minimum is at (0, 0) with value 0")
    
    def objective1(x):
        return x[0]**2 + x[1]**2
    
    def grad_objective1(x):
        return np.array([2 * x[0], 2 * x[1]])
    
    optimizer1 = GradientDescent(learning_rate=0.1, max_iterations=100, tolerance=1e-6)
    x0 = np.array([3.0, 4.0])
    
    print(f"Starting point: {x0}")
    print(f"Initial function value: {objective1(x0):.6f}")
    
    x_opt, info = optimizer1.minimize(objective1, x0, gradient=grad_objective1)
    
    print(f"\nOptimization results:")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Optimal point: [{x_opt[0]:.6f}, {x_opt[1]:.6f}]")
    print(f"  Optimal value: {info['final_value']:.6f}")
    print(f"  Expected: [0.000000, 0.000000] with value 0.000000")
    
    # Show convergence history
    print(f"\nConvergence history (first 5 and last 5 iterations):")
    history = info['history']
    for i in [0, 1, 2, 3, 4] + list(range(max(0, len(history['iterations'])-5), len(history['iterations']))):
        if i < len(history['iterations']):
            print(f"  Iter {history['iterations'][i]}: value = {history['values'][i]:.6f}, "
                  f"x = [{history['positions'][i][0]:.4f}, {history['positions'][i][1]:.4f}]")
    
    # Example 2: Using numerical gradient
    print("\n" + "-" * 70)
    print("Example 2: Same function, but using numerical gradient (no analytical gradient provided)")
    
    optimizer2 = GradientDescent(learning_rate=0.1, max_iterations=100, tolerance=1e-6)
    x_opt2, info2 = optimizer2.minimize(objective1, x0)
    
    print(f"  Optimal point: [{x_opt2[0]:.6f}, {x_opt2[1]:.6f}]")
    print(f"  Optimal value: {info2['final_value']:.6f}")
    print(f"  Iterations: {info2['iterations']}")
    
    # Example 3: Different learning rates
    print("\n" + "-" * 70)
    print("Example 3: Effect of different learning rates")
    
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    for lr in learning_rates:
        optimizer = GradientDescent(learning_rate=lr, max_iterations=50, tolerance=1e-6)
        x_opt, info = optimizer.minimize(objective1, x0, gradient=grad_objective1)
        print(f"  Learning rate {lr:4.2f}: {info['iterations']:3d} iterations, "
              f"final value = {info['final_value']:.6f}, converged = {info['converged']}")
    
    
    print_section("5. GRADIENT DESCENT - MORE COMPLEX FUNCTIONS")
    
    # Example: Rosenbrock function (famous test function)
    print("Example: Minimizing Rosenbrock function f(x, y) = (1-x)² + 100(y-x²)²")
    print("Global minimum is at (1, 1) with value 0")
    
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def grad_rosenbrock(x):
        return np.array([
            -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
            200 * (x[1] - x[0]**2)
        ])
    
    optimizer3 = GradientDescent(learning_rate=0.001, max_iterations=1000, tolerance=1e-6)
    x0_rosen = np.array([-1.0, 1.0])
    
    print(f"Starting point: {x0_rosen}")
    print(f"Initial function value: {rosenbrock(x0_rosen):.6f}")
    
    x_opt, info = optimizer3.minimize(rosenbrock, x0_rosen, gradient=grad_rosenbrock)
    
    print(f"\nOptimization results:")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Optimal point: [{x_opt[0]:.6f}, {x_opt[1]:.6f}]")
    print(f"  Optimal value: {info['final_value']:.6f}")
    print(f"  Expected: [1.000000, 1.000000] with value 0.000000")
    
    # Example: Linear regression cost function
    print("\n" + "-" * 70)
    print("Example: Minimizing linear regression cost function")
    print("f(w, b) = (w·x + b - y)² for a simple dataset")
    
    # Simple dataset: y = 2x + 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])  # y = 2x + 1
    
    def linear_regression_cost(params):
        w, b = params[0], params[1]
        predictions = w * X.flatten() + b
        return np.mean((predictions - y)**2)
    
    def grad_linear_regression(params):
        w, b = params[0], params[1]
        predictions = w * X.flatten() + b
        errors = predictions - y
        dw = 2 * np.mean(errors * X.flatten())
        db = 2 * np.mean(errors)
        return np.array([dw, db])
    
    optimizer4 = GradientDescent(learning_rate=0.1, max_iterations=100, tolerance=1e-6)
    x0_lr = np.array([0.0, 0.0])
    
    print(f"Starting point: w={x0_lr[0]}, b={x0_lr[1]}")
    print(f"Initial cost: {linear_regression_cost(x0_lr):.6f}")
    
    x_opt, info = optimizer4.minimize(linear_regression_cost, x0_lr, gradient=grad_linear_regression)
    
    print(f"\nOptimization results:")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Optimal: w={x_opt[0]:.6f}, b={x_opt[1]:.6f}")
    print(f"  Optimal cost: {info['final_value']:.6f}")
    print(f"  Expected: w≈2.0, b≈1.0")
    
    
    print_section("6. STOCHASTIC GRADIENT DESCENT (SGD)")
    
    print("Example: Linear regression with SGD using mini-batches")
    print("This is useful for large datasets where computing full gradient is expensive")
    
    # Larger dataset
    np.random.seed(42)
    n_samples = 100
    X_large = np.random.randn(n_samples, 1)
    y_large = 2 * X_large.flatten() + 1 + 0.1 * np.random.randn(n_samples)
    
    def linear_regression_batch(params, batch_data):
        """Cost function for a batch of data."""
        w, b = params[0], params[1]
        X_batch, y_batch = batch_data[:, 0], batch_data[:, 1]
        predictions = w * X_batch + b
        return np.mean((predictions - y_batch)**2)
    
    def grad_linear_regression_batch(params, batch_data):
        """Gradient for a batch of data."""
        w, b = params[0], params[1]
        X_batch, y_batch = batch_data[:, 0], batch_data[:, 1]
        predictions = w * X_batch + b
        errors = predictions - y_batch
        dw = 2 * np.mean(errors * X_batch)
        db = 2 * np.mean(errors)
        return np.array([dw, db])
    
    # Prepare data: combine X and y
    data = np.column_stack([X_large.flatten(), y_large])
    
    sgd_optimizer = StochasticGradientDescent(
        learning_rate=0.01,
        max_iterations=200,
        batch_size=10,
        tolerance=1e-4
    )
    
    x0_sgd = np.array([0.0, 0.0])
    print(f"Starting point: w={x0_sgd[0]}, b={x0_sgd[1]}")
    print(f"Dataset size: {n_samples} samples")
    print(f"Batch size: {sgd_optimizer.batch_size}")
    
    x_opt, info = sgd_optimizer.minimize_batch(
        linear_regression_batch,
        x0_sgd,
        data,
        gradient=grad_linear_regression_batch
    )
    
    print(f"\nSGD Optimization results:")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Optimal: w={x_opt[0]:.6f}, b={x_opt[1]:.6f}")
    print(f"  Final cost: {info['final_value']:.6f}")
    print(f"  Expected: w≈2.0, b≈1.0")
    
    
    print_section("7. COMPARING DIFFERENT STEP SIZES")
    
    print("Comparing numerical differentiation with different step sizes")
    print("Function: f(x) = x³, f'(x) = 3x²")
    
    def f_cubic(x):
        return x ** 3
    
    x_val = 2.0
    analytical = 3 * x_val**2
    
    step_sizes = [1e-3, 1e-5, 1e-7, 1e-9]
    print(f"\nAt x = {x_val}, analytical derivative = {analytical}")
    print(f"{'Step size':<15} {'Numerical':<15} {'Error':<15}")
    print("-" * 45)
    
    for h in step_sizes:
        numerical = numerical_derivative(f_cubic, x_val, h=h, method='central')
        error = abs(numerical - analytical)
        print(f"{h:<15.2e} {numerical:<15.10f} {error:<15.2e}")
    
    print("\nNote: Very small step sizes can lead to numerical errors due to floating-point precision!")
    
    
    print_section("8. ERROR HANDLING EXAMPLES")
    
    print("Demonstrating error handling in optimization:")
    
    # Invalid method for numerical_derivative
    try:
        result = numerical_derivative(f1, 1.0, method='invalid')
    except ValueError as e:
        print(f"✓ Caught error for invalid method: {e}")
    
    # Very large learning rate (may not converge)
    print("\nTesting with very large learning rate (may diverge):")
    optimizer_large_lr = GradientDescent(learning_rate=10.0, max_iterations=10, tolerance=1e-6)
    x_opt, info = optimizer_large_lr.minimize(objective1, np.array([1.0, 1.0]), gradient=grad_objective1)
    print(f"  Converged: {info['converged']}")
    print(f"  Final value: {info['final_value']:.6f}")
    print(f"  Note: Large learning rates can cause divergence!")
    
    # Very small learning rate (slow convergence)
    print("\nTesting with very small learning rate (slow convergence):")
    optimizer_small_lr = GradientDescent(learning_rate=0.0001, max_iterations=50, tolerance=1e-6)
    x_opt, info = optimizer_small_lr.minimize(objective1, np.array([3.0, 4.0]), gradient=grad_objective1)
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Final value: {info['final_value']:.6f}")
    print(f"  Note: Small learning rates converge slowly but are more stable!")
    
    
    print_section("9. PRACTICAL TIPS AND BEST PRACTICES")
    
    print("Tips for using numerical differentiation:")
    print("  1. Use central differences when possible (most accurate)")
    print("  2. Step size h ≈ 1e-5 to 1e-7 is usually good")
    print("  3. Too small h can cause numerical errors")
    print("  4. Too large h reduces accuracy")
    
    print("\nTips for using gradient descent:")
    print("  1. Learning rate is crucial - start with 0.01 or 0.1")
    print("  2. If diverging, reduce learning rate")
    print("  3. If converging slowly, increase learning rate (carefully)")
    print("  4. Use analytical gradients when available (faster, more accurate)")
    print("  5. Monitor convergence history to tune hyperparameters")
    print("  6. For large datasets, use SGD with mini-batches")
    
    print("\nWhen to use numerical vs analytical gradients:")
    print("  - Numerical: Quick prototyping, complex functions, checking analytical gradients")
    print("  - Analytical: Production code, faster optimization, more accurate")
    
    
    print_section("EXAMPLES COMPLETE!")
    print("All functions in the optimization module have been demonstrated.")
    print("You can use these examples as a reference for your own optimization tasks.")


if __name__ == "__main__":
    main()


