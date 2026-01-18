"""
Examples demonstrating optimizers.

This file shows how to use different optimizers for training neural networks.
"""

import numpy as np
from nnn import Tensor, Dense, Sequential
from nnn.activations import relu
from nnn.losses import mse_loss, cross_entropy_loss
from nnn.optim import SGD, Adam, RMSprop


def example_sgd_basic():
    """Example: Basic SGD optimizer."""
    print("=" * 60)
    print("SGD Optimizer - Basic Usage")
    print("=" * 60)
    
    # Simple model
    model = Sequential(
        Dense(5, 3, activation=relu),
        Dense(3, 1, activation=None)
    )
    
    # Create optimizer
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Generate data
    X = Tensor(np.random.randn(10, 5).astype(np.float32), requires_grad=False)
    y = Tensor(np.random.randn(10, 1).astype(np.float32), requires_grad=False)
    
    # Training step
    predictions = model(X)
    loss = mse_loss(predictions, y)
    
    print(f"Loss before optimization: {loss.data.item():.4f}")
    
    # Backward and optimize
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Forward again
    predictions_after = model(X)
    loss_after = mse_loss(predictions_after, y)
    
    print(f"Loss after one step: {loss_after.data.item():.4f}")
    print()


def example_sgd_momentum():
    """Example: SGD with momentum."""
    print("=" * 60)
    print("SGD Optimizer - With Momentum")
    print("=" * 60)
    
    model = Sequential(
        Dense(10, 5, activation=relu),
        Dense(5, 1, activation=None)
    )
    
    # SGD without momentum
    optimizer_no_momentum = SGD(model.parameters(), lr=0.01, momentum=0.0)
    
    # SGD with momentum
    model_momentum = Sequential(
        Dense(10, 5, activation=relu),
        Dense(5, 1, activation=None)
    )
    optimizer_momentum = SGD(model_momentum.parameters(), lr=0.01, momentum=0.9)
    
    X = Tensor(np.random.randn(20, 10).astype(np.float32), requires_grad=False)
    y = Tensor(np.random.randn(20, 1).astype(np.float32), requires_grad=False)
    
    print("Training with and without momentum:")
    for epoch in range(5):
        # Without momentum
        pred1 = model(X)
        loss1 = mse_loss(pred1, y)
        model.zero_grad()
        loss1.backward()
        optimizer_no_momentum.step()
        
        # With momentum
        pred2 = model_momentum(X)
        loss2 = mse_loss(pred2, y)
        model_momentum.zero_grad()
        loss2.backward()
        optimizer_momentum.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: No momentum={loss1.data.item():.4f}, "
                  f"With momentum={loss2.data.item():.4f}")
    print()


def example_sgd_weight_decay():
    """Example: SGD with weight decay (L2 regularization)."""
    print("=" * 60)
    print("SGD Optimizer - With Weight Decay")
    print("=" * 60)
    
    model = Sequential(
        Dense(8, 4, activation=relu),
        Dense(4, 1, activation=None)
    )
    
    # Get initial weights
    initial_weights = [p.data.copy() for p in model.parameters()]
    
    optimizer = SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    
    X = Tensor(np.random.randn(10, 8).astype(np.float32), requires_grad=False)
    y = Tensor(np.random.randn(10, 1).astype(np.float32), requires_grad=False)
    
    # Training step
    predictions = model(X)
    loss = mse_loss(predictions, y)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check weight change
    final_weights = [p.data for p in model.parameters()]
    weight_changes = [np.linalg.norm(fw - iw) for fw, iw in zip(final_weights, initial_weights)]
    
    print("Weight changes after one step:")
    for i, change in enumerate(weight_changes):
        print(f"  Parameter {i}: {change:.6f}")
    print("Note: Weight decay adds L2 regularization to gradients")
    print()


def example_adam_basic():
    """Example: Adam optimizer."""
    print("=" * 60)
    print("Adam Optimizer - Basic Usage")
    print("=" * 60)
    
    model = Sequential(
        Dense(10, 8, activation=relu),
        Dense(8, 1, activation=None)
    )
    
    optimizer = Adam(model.parameters(), lr=0.001)
    
    X = Tensor(np.random.randn(16, 10).astype(np.float32), requires_grad=False)
    y = Tensor(np.random.randn(16, 1).astype(np.float32), requires_grad=False)
    
    print("Training with Adam:")
    for epoch in range(10):
        predictions = model(X)
        loss = mse_loss(predictions, y)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data.item():.4f}")
    print()


def example_adam_parameters():
    """Example: Adam with different parameters."""
    print("=" * 60)
    print("Adam Optimizer - Parameter Tuning")
    print("=" * 60)
    
    models = {
        'Default': (Sequential(Dense(5, 1, activation=None)), 
                   Adam([], lr=0.001, betas=(0.9, 0.999))),
        'High LR': (Sequential(Dense(5, 1, activation=None)),
                   Adam([], lr=0.01, betas=(0.9, 0.999))),
        'Low Beta1': (Sequential(Dense(5, 1, activation=None)),
                     Adam([], lr=0.001, betas=(0.5, 0.999)))
    }
    
    X = Tensor(np.random.randn(10, 5).astype(np.float32), requires_grad=False)
    y = Tensor(np.random.randn(10, 1).astype(np.float32), requires_grad=False)
    
    for name, (model, opt) in models.items():
        opt.parameters = model.parameters()
        # Initialize with same weights
        for p in model.parameters():
            p.data = np.random.randn(*p.shape).astype(np.float32) * 0.1
        
        # Train for a few steps
        for _ in range(5):
            pred = model(X)
            loss = mse_loss(pred, y)
            model.zero_grad()
            loss.backward()
            opt.step()
        
        final_loss = mse_loss(model(X), y)
        print(f"{name:15s}: Final loss = {final_loss.data.item():.4f}")
    print()


def example_rmsprop():
    """Example: RMSprop optimizer."""
    print("=" * 60)
    print("RMSprop Optimizer")
    print("=" * 60)
    
    model = Sequential(
        Dense(10, 5, activation=relu),
        Dense(5, 1, activation=None)
    )
    
    optimizer = RMSprop(model.parameters(), lr=0.01, alpha=0.99)
    
    X = Tensor(np.random.randn(20, 10).astype(np.float32), requires_grad=False)
    y = Tensor(np.random.randn(20, 1).astype(np.float32), requires_grad=False)
    
    print("Training with RMSprop:")
    for epoch in range(10):
        predictions = model(X)
        loss = mse_loss(predictions, y)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data.item():.4f}")
    print()


def example_optimizer_comparison():
    """Example: Compare different optimizers."""
    print("=" * 60)
    print("Optimizer Comparison")
    print("=" * 60)
    
    # Create identical models
    models = {
        'SGD': Sequential(Dense(10, 5, activation=relu), Dense(5, 1, activation=None)),
        'SGD+Momentum': Sequential(Dense(10, 5, activation=relu), Dense(5, 1, activation=None)),
        'Adam': Sequential(Dense(10, 5, activation=relu), Dense(5, 1, activation=None)),
        'RMSprop': Sequential(Dense(10, 5, activation=relu), Dense(5, 1, activation=None))
    }
    
    optimizers = {
        'SGD': SGD(models['SGD'].parameters(), lr=0.01),
        'SGD+Momentum': SGD(models['SGD+Momentum'].parameters(), lr=0.01, momentum=0.9),
        'Adam': Adam(models['Adam'].parameters(), lr=0.001),
        'RMSprop': RMSprop(models['RMSprop'].parameters(), lr=0.01)
    }
    
    # Initialize with same weights
    for model in models.values():
        for p in model.parameters():
            p.data = np.random.randn(*p.shape).astype(np.float32) * 0.1
    
    X = Tensor(np.random.randn(16, 10).astype(np.float32), requires_grad=False)
    y = Tensor(np.random.randn(16, 1).astype(np.float32), requires_grad=False)
    
    print("Training progress:")
    for epoch in range(15):
        losses = {}
        for name, model in models.items():
            pred = model(X)
            loss = mse_loss(pred, y)
            losses[name] = loss.data.item()
            
            model.zero_grad()
            loss.backward()
            optimizers[name].step()
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch:2d}: ", end="")
            for name, loss_val in losses.items():
                print(f"{name:15s}={loss_val:.4f}  ", end="")
            print()
    print()


def example_training_loop():
    """Example: Complete training loop with optimizer."""
    print("=" * 60)
    print("Complete Training Loop")
    print("=" * 60)
    
    # Create model
    model = Sequential(
        Dense(8, 16, activation=relu),
        Dense(16, 8, activation=relu),
        Dense(8, 1, activation=None)
    )
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Generate training data
    np.random.seed(42)
    X_train = Tensor(np.random.randn(100, 8).astype(np.float32), requires_grad=False)
    y_train = Tensor((X_train.data.sum(axis=1, keepdims=True) * 0.5 + 
                     np.random.randn(100, 1) * 0.1).astype(np.float32), requires_grad=False)
    
    print("Training model:")
    num_epochs = 20
    batch_size = 16
    
    for epoch in range(num_epochs):
        # Mini-batch training
        indices = np.random.permutation(len(X_train.data))
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train.data), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = Tensor(X_train.data[batch_indices], requires_grad=False)
            y_batch = Tensor(y_train.data[batch_indices], requires_grad=False)
            
            # Forward pass
            predictions = model(X_batch)
            loss = mse_loss(predictions, y_batch)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            epoch_loss += loss.data.item()
            n_batches += 1
        
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / n_batches
            print(f"Epoch {epoch+1:2d}/{num_epochs}: Loss = {avg_loss:.4f}")
    print()


def example_classification_training():
    """Example: Training classifier with cross-entropy loss."""
    print("=" * 60)
    print("Classification Training with Optimizer")
    print("=" * 60)
    
    # Classification model
    model = Sequential(
        Dense(10, 20, activation=relu),
        Dense(20, 10, activation=relu),
        Dense(10, 3, activation=None)  # 3 classes
    )
    
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Generate classification data
    np.random.seed(42)
    X = Tensor(np.random.randn(50, 10).astype(np.float32), requires_grad=False)
    y = Tensor(np.random.randint(0, 3, (50,)).astype(np.int32), requires_grad=False)
    
    print("Training classifier:")
    for epoch in range(15):
        predictions = model(X)
        loss = cross_entropy_loss(predictions, y)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:2d}: Loss = {loss.data.item():.4f}")
    print()


def example_optimizer_state():
    """Example: Understanding optimizer state."""
    print("=" * 60)
    print("Optimizer State (Adam)")
    print("=" * 60)
    
    model = Sequential(Dense(5, 1, activation=None))
    optimizer = Adam(model.parameters(), lr=0.001)
    
    X = Tensor(np.random.randn(10, 5).astype(np.float32), requires_grad=False)
    y = Tensor(np.random.randn(10, 1).astype(np.float32), requires_grad=False)
    
    print("Adam optimizer maintains:")
    print(f"  - First moment estimates (m): {len(optimizer.m)} tensors")
    print(f"  - Second moment estimates (v): {len(optimizer.v)} tensors")
    print(f"  - Time step (t): {optimizer.t}")
    
    # Train for a few steps
    for step in range(3):
        pred = model(X)
        loss = mse_loss(pred, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"\nAfter step {step+1}:")
        print(f"  Time step: {optimizer.t}")
        print(f"  First moment norm: {np.linalg.norm(optimizer.m[0]):.6f}")
        print(f"  Second moment norm: {np.linalg.norm(optimizer.v[0]):.6f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OPTIMIZERS EXAMPLES - Optimization Algorithms")
    print("=" * 60 + "\n")
    
    example_sgd_basic()
    example_sgd_momentum()
    example_sgd_weight_decay()
    example_adam_basic()
    example_adam_parameters()
    example_rmsprop()
    example_optimizer_comparison()
    example_training_loop()
    example_classification_training()
    example_optimizer_state()
    
    print("=" * 60)
    print("All optimizers examples completed!")
    print("=" * 60)
