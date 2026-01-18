"""
Example training script demonstrating the neural network framework.

This script shows how to:
1. Create a neural network
2. Define loss function
3. Set up optimizer
4. Train the model
"""

import numpy as np
from nnn import Tensor, Dense, Sequential, relu, softmax
from nnn.losses import mse_loss, cross_entropy_loss
from nnn.optim import SGD, Adam


def example_regression():
    """Example: Simple regression task."""
    print("=" * 50)
    print("Regression Example")
    print("=" * 50)
    
    # Generate synthetic data: y = 2x + 1 + noise
    np.random.seed(42)
    X = np.random.randn(100, 1).astype(np.float32)
    y = (2 * X + 1 + 0.1 * np.random.randn(100, 1)).astype(np.float32)
    
    # Create model: single layer
    model = Dense(in_features=1, out_features=1, use_bias=True, activation=None)
    
    # Loss and optimizer
    criterion = mse_loss
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        X_tensor = Tensor(X, requires_grad=False)
        y_tensor = Tensor(y, requires_grad=False)
        
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.data.item():.4f}")
    
    # Check learned parameters
    print(f"\nLearned weight: {model.weight.data[0, 0]:.4f} (expected ~2.0)")
    print(f"Learned bias: {model.bias.data[0]:.4f} (expected ~1.0)")
    print()


def example_classification():
    """Example: Binary classification task."""
    print("=" * 50)
    print("Classification Example")
    print("=" * 50)
    
    # Generate synthetic data: two classes
    np.random.seed(42)
    n_samples = 200
    
    # Class 0: centered at (-1, -1)
    X0 = np.random.randn(n_samples // 2, 2).astype(np.float32) + np.array([-1, -1])
    y0 = np.zeros((n_samples // 2,), dtype=np.int32)
    
    # Class 1: centered at (1, 1)
    X1 = np.random.randn(n_samples // 2, 2).astype(np.float32) + np.array([1, 1])
    y1 = np.ones((n_samples // 2,), dtype=np.int32)
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Create model: 2 -> 4 -> 2 -> 1
    model = Sequential(
        Dense(2, 4, use_bias=True, activation=relu),
        Dense(4, 2, use_bias=True, activation=relu),
        Dense(2, 1, use_bias=True, activation=None)  # No activation for binary classification
    )
    
    # Loss and optimizer
    def binary_ce_loss(pred, target):
        # Apply sigmoid for binary classification
        from nnn.activations import sigmoid
        probs = sigmoid(pred)
        # Simple binary cross-entropy
        eps = 1e-15
        # Handle both Tensor and numpy array inputs
        target_data = target.data if isinstance(target, Tensor) else target
        target_tensor = Tensor(target_data.reshape(-1, 1).astype(np.float32), requires_grad=False)
        loss = -(target_tensor * Tensor(np.log(probs.data + eps), requires_grad=False) + 
                 (1 - target_tensor) * Tensor(np.log(1 - probs.data + eps), requires_grad=False))
        return loss.mean()
    
    criterion = binary_ce_loss
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 200
    batch_size = 32
    
    for epoch in range(num_epochs):
        # Mini-batch training
        indices = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Forward pass
            X_tensor = Tensor(X_batch, requires_grad=False)
            y_tensor = Tensor(y_batch, requires_grad=False)
            
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.data.item()
            n_batches += 1
        
        if (epoch + 1) % 40 == 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate
    X_tensor = Tensor(X, requires_grad=False)
    pred = model(X_tensor)
    from nnn.activations import sigmoid
    probs = sigmoid(pred)
    predictions = (probs.data > 0.5).astype(int).flatten()
    accuracy = np.mean(predictions == y)
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    print()


def example_multi_class():
    """Example: Multi-class classification."""
    print("=" * 50)
    print("Multi-class Classification Example")
    print("=" * 50)
    
    # Generate synthetic data: 3 classes
    np.random.seed(42)
    n_samples_per_class = 100
    n_classes = 3
    
    X_list = []
    y_list = []
    
    for class_id in range(n_classes):
        angle = 2 * np.pi * class_id / n_classes
        center = np.array([np.cos(angle), np.sin(angle)]) * 2
        X_class = np.random.randn(n_samples_per_class, 2).astype(np.float32) + center
        y_class = np.full((n_samples_per_class,), class_id, dtype=np.int32)
        X_list.append(X_class)
        y_list.append(y_class)
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # Create model: 2 -> 8 -> 3
    model = Sequential(
        Dense(2, 8, use_bias=True, activation=relu),
        Dense(8, 3, use_bias=True, activation=None)  # No activation, softmax in loss
    )
    
    # Loss and optimizer
    criterion = cross_entropy_loss
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 300
    batch_size = 32
    n_samples = len(X)
    
    for epoch in range(num_epochs):
        # Mini-batch training
        indices = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Forward pass
            X_tensor = Tensor(X_batch, requires_grad=False)
            y_tensor = Tensor(y_batch, requires_grad=False)
            
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.data.item()
            n_batches += 1
        
        if (epoch + 1) % 60 == 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate
    X_tensor = Tensor(X, requires_grad=False)
    pred = model(X_tensor)
    probs = softmax(pred, axis=-1)
    predictions = np.argmax(probs.data, axis=-1)
    accuracy = np.mean(predictions == y)
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    print()


if __name__ == "__main__":
    # Run examples
    example_regression()
    example_classification()
    example_multi_class()
    
    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)
