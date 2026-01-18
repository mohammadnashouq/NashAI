# Neural Network Framework from Scratch

A complete deep learning framework implemented from scratch with automatic differentiation (autograd), neural network layers, activation functions, loss functions, and optimizers.

## Features

- ✅ **Automatic Differentiation**: Full autograd engine with computational graph
- ✅ **Neural Network Layers**: Dense (fully connected) layers
- ✅ **Activation Functions**: ReLU, GELU, Sigmoid, Tanh, Softmax, Leaky ReLU
- ✅ **Loss Functions**: MSE, Cross-Entropy, Binary Cross-Entropy, L1
- ✅ **Optimizers**: SGD, Adam, RMSprop
- ✅ **No External Dependencies**: Pure NumPy implementation (except NumPy itself)

## Architecture

### 1. Autograd Engine (`tensor.py`)

The `Tensor` class implements automatic differentiation:

```python
from nn import Tensor

# Create tensors with gradient tracking
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = Tensor([2.0, 3.0, 4.0], requires_grad=True)

# Perform operations
z = x * y + x
loss = z.sum()

# Backpropagate
loss.backward()

# Access gradients
print(x.grad)  # Gradient of loss w.r.t. x
print(y.grad)  # Gradient of loss w.r.t. y
```

**Key Features:**
- Computational graph tracking
- Automatic gradient computation via backpropagation
- Chain rule implementation
- Broadcasting support

### 2. Neural Network Layers (`layers.py`)

#### Dense Layer

```python
from nn import Dense, relu

# Create a fully connected layer
layer = Dense(in_features=784, out_features=128, use_bias=True, activation=relu)

# Forward pass
output = layer(input_tensor)
```

#### Sequential Container

```python
from nn import Sequential, Dense, relu, softmax

# Stack multiple layers
model = Sequential(
    Dense(784, 256, activation=relu),
    Dense(256, 128, activation=relu),
    Dense(128, 10, activation=None)  # No activation for logits
)
```

### 3. Activation Functions (`activations.py`)

Available activations:
- `relu(x)`: Rectified Linear Unit
- `gelu(x)`: Gaussian Error Linear Unit
- `sigmoid(x)`: Sigmoid function
- `tanh(x)`: Hyperbolic tangent
- `softmax(x, axis=-1)`: Softmax normalization
- `leaky_relu(x, alpha=0.01)`: Leaky ReLU

All activations support automatic differentiation.

### 4. Loss Functions (`losses.py`)

#### Mean Squared Error (MSE)

```python
from nn.losses import mse_loss

loss = mse_loss(predictions, targets)
```

#### Cross-Entropy Loss

```python
from nn.losses import cross_entropy_loss

# For multi-class classification
# predictions: (batch_size, num_classes) - logits
# targets: (batch_size,) - class indices OR (batch_size, num_classes) - one-hot
loss = cross_entropy_loss(predictions, targets)
```

#### Binary Cross-Entropy

```python
from nn.losses import binary_cross_entropy_loss

# For binary classification
# predictions: (batch_size,) - probabilities (after sigmoid)
# targets: (batch_size,) - binary labels
loss = binary_cross_entropy_loss(predictions, targets)
```

### 5. Optimizers (`optim.py`)

#### Stochastic Gradient Descent (SGD)

```python
from nn.optim import SGD

optimizer = SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,      # Optional momentum
    weight_decay=0.0001  # Optional L2 regularization
)
```

#### Adam Optimizer

```python
from nn.optim import Adam

optimizer = Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),  # Beta1 and Beta2
    eps=1e-8,
    weight_decay=0.0001
)
```

#### RMSprop

```python
from nn.optim import RMSprop

optimizer = RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
    eps=1e-8
)
```

## Training Example

```python
import numpy as np
from nn import Tensor, Dense, Sequential, relu
from nn.losses import mse_loss
from nn.optim import Adam

# 1. Create model
model = Sequential(
    Dense(10, 64, activation=relu),
    Dense(64, 32, activation=relu),
    Dense(32, 1, activation=None)
)

# 2. Define loss and optimizer
criterion = mse_loss
optimizer = Adam(model.parameters(), lr=0.001)

# 3. Training loop
for epoch in range(num_epochs):
    # Forward pass
    X_tensor = Tensor(X_train, requires_grad=False)
    y_tensor = Tensor(y_train, requires_grad=False)
    
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.data.item():.4f}")
```

## Complete Example

See `train_example.py` for complete examples including:
- Simple regression
- Binary classification
- Multi-class classification

Run examples:
```bash
python nn/train_example.py
```

## Implementation Details

### Computational Graph

The autograd engine builds a computational graph during the forward pass:
- Each `Tensor` tracks its operation (`_op`)
- Child tensors are stored (`_children`)
- Backward function is stored (`_backward_fn`)

### Backpropagation

When `backward()` is called:
1. Gradient flows from output to input
2. Chain rule is applied at each node
3. Gradients are accumulated for nodes used multiple times

### Gradient Computation

Each operation defines its backward function:
- **Addition**: Gradient flows to both operands
- **Multiplication**: `d/dx (x*y) = y`, `d/dy (x*y) = x`
- **Matrix Multiplication**: `d/dA (A@B) = grad @ B^T`, `d/dB (A@B) = A^T @ grad`
- **ReLU**: Gradient is 1 where input > 0, else 0
- **Softmax**: More complex, involves Jacobian matrix

## File Structure

```
nn/
├── __init__.py          # Package exports
├── tensor.py            # Autograd engine (Tensor class)
├── layers.py            # Neural network layers
├── activations.py       # Activation functions
├── losses.py            # Loss functions
├── optim.py             # Optimizers
├── train_example.py     # Training examples
└── README.md            # This file
```

## Dependencies

- NumPy (for array operations)

## Notes

- All operations use float32 for efficiency
- Numerical stability is handled (e.g., log(0) protection)
- Broadcasting is supported for tensor operations
- The framework is designed for educational purposes and may not be as optimized as PyTorch/TensorFlow

## Future Enhancements

Potential additions:
- Convolutional layers
- Recurrent layers (LSTM, GRU)
- Batch normalization
- Dropout
- More optimizers (AdaGrad, AdaDelta)
- GPU support
- More efficient memory management
