"""
MNIST Classification with MLP - NashAI vs PyTorch Comparison

This benchmark compares a Multi-Layer Perceptron trained on MNIST using:
1. NashAI (our from-scratch implementation)
2. PyTorch (industry standard)

Metrics compared:
- Training accuracy
- Test accuracy
- Training time
- Loss convergence
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# NashAI imports
from nnn.tensor import Tensor
from nnn.layers import Dense, Sequential
from nnn.activations import relu, sigmoid
from nnn.losses import cross_entropy_loss
from nnn.optim import SGD, Adam

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Sklearn for data loading
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_mnist(subset_size=10000):
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    
    # Fetch MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Use subset for faster training
    if subset_size and subset_size < len(X):
        indices = np.random.permutation(len(X))[:subset_size]
        X, y = X[indices], y[indices]
    
    # Normalize to [0, 1]
    X = X.astype(np.float32) / 255.0
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def one_hot_encode(y, num_classes=10):
    """Convert labels to one-hot encoding."""
    one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot


# ============================================
# NashAI Implementation
# ============================================

class NashAIMLP:
    """MLP using NashAI library."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        self.fc1 = Dense(input_size, hidden_size)
        self.fc2 = Dense(hidden_size, hidden_size)
        self.fc3 = Dense(hidden_size, num_classes)
    
    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters()
    
    def zero_grad(self):
        self.fc1.zero_grad()
        self.fc2.zero_grad()
        self.fc3.zero_grad()


def train_nashai(X_train, y_train, X_test, y_test, epochs=10, batch_size=64, lr=0.01):
    """Train MLP using NashAI."""
    print("\n" + "="*50)
    print("Training with NashAI")
    print("="*50)
    
    # Convert labels to one-hot
    y_train_oh = one_hot_encode(y_train)
    y_test_oh = one_hot_encode(y_test)
    
    # Initialize model and optimizer
    model = NashAIMLP()
    optimizer = Adam(model.parameters(), lr=lr)
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    n_samples = len(X_train)
    n_batches = n_samples // batch_size
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train_oh[indices]
        y_labels = y_train[indices]
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            
            # Get batch
            X_batch = Tensor(X_shuffled[start:end], requires_grad=True)
            y_batch = Tensor(y_shuffled[start:end])
            
            # Forward pass
            logits = model.forward(X_batch)
            loss = cross_entropy_loss(logits, y_batch)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.data.item()
            
            # Compute accuracy
            predictions = np.argmax(logits.data, axis=1)
            epoch_correct += np.sum(predictions == y_labels[start:end])
        
        # Compute metrics
        avg_loss = epoch_loss / n_batches
        train_acc = epoch_correct / (n_batches * batch_size) * 100
        
        # Test accuracy
        test_logits = model.forward(Tensor(X_test))
        test_preds = np.argmax(test_logits.data, axis=1)
        test_acc = np.mean(test_preds == y_test) * 100
        
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'training_time': training_time,
        'final_test_acc': test_accs[-1]
    }


# ============================================
# PyTorch Implementation
# ============================================

class PyTorchMLP(nn.Module):
    """MLP using PyTorch."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_pytorch(X_train, y_train, X_test, y_test, epochs=10, batch_size=64, lr=0.01):
    """Train MLP using PyTorch."""
    print("\n" + "="*50)
    print("Training with PyTorch")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = PyTorchMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            epoch_correct += predicted.eq(y_batch).sum().item()
        
        # Compute metrics
        avg_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / total * 100
        
        # Test accuracy
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t.to(device))
            _, test_preds = test_outputs.max(1)
            test_acc = (test_preds.cpu() == y_test_t).float().mean().item() * 100
        
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'training_time': training_time,
        'final_test_acc': test_accs[-1]
    }


# ============================================
# Comparison and Visualization
# ============================================

def plot_comparison(nashai_results, pytorch_results):
    """Plot comparison of NashAI vs PyTorch training."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(nashai_results['train_losses']) + 1)
    
    # Loss comparison
    axes[0].plot(epochs, nashai_results['train_losses'], 'b-', label='NashAI', linewidth=2)
    axes[0].plot(epochs, pytorch_results['train_losses'], 'r--', label='PyTorch', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Training accuracy comparison
    axes[1].plot(epochs, nashai_results['train_accs'], 'b-', label='NashAI', linewidth=2)
    axes[1].plot(epochs, pytorch_results['train_accs'], 'r--', label='PyTorch', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Test accuracy comparison
    axes[2].plot(epochs, nashai_results['test_accs'], 'b-', label='NashAI', linewidth=2)
    axes[2].plot(epochs, pytorch_results['test_accs'], 'r--', label='PyTorch', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Test Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mnist_mlp_comparison.png', dpi=150)
    plt.show()
    print("\nPlot saved as 'mnist_mlp_comparison.png'")


def print_summary(nashai_results, pytorch_results):
    """Print comparison summary."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY: MNIST MLP")
    print("="*60)
    
    print(f"\n{'Metric':<25} {'NashAI':>15} {'PyTorch':>15}")
    print("-"*55)
    print(f"{'Final Test Accuracy':<25} {nashai_results['final_test_acc']:>14.2f}% {pytorch_results['final_test_acc']:>14.2f}%")
    print(f"{'Training Time':<25} {nashai_results['training_time']:>13.2f}s {pytorch_results['training_time']:>13.2f}s")
    print(f"{'Time Ratio (NashAI/PT)':<25} {nashai_results['training_time']/pytorch_results['training_time']:>15.2f}x")
    
    print("\n" + "="*60)
    print("Analysis:")
    print("="*60)
    
    acc_diff = abs(nashai_results['final_test_acc'] - pytorch_results['final_test_acc'])
    if acc_diff < 2.0:
        print(f"Accuracy is comparable (difference: {acc_diff:.2f}%)")
    else:
        print(f"Accuracy difference is notable: {acc_diff:.2f}%")
    
    time_ratio = nashai_results['training_time'] / pytorch_results['training_time']
    print(f"NashAI is {time_ratio:.1f}x slower than PyTorch (expected due to pure Python)")


def main():
    """Run the MNIST MLP comparison."""
    print("="*60)
    print("MNIST MLP BENCHMARK: NashAI vs PyTorch")
    print("="*60)
    
    # Parameters
    SUBSET_SIZE = 10000  # Use subset for faster demo (set to None for full dataset)
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    
    print(f"\nParameters:")
    print(f"  Dataset size: {SUBSET_SIZE if SUBSET_SIZE else 'Full'}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_mnist(SUBSET_SIZE)
    
    # Train with NashAI
    nashai_results = train_nashai(
        X_train, y_train, X_test, y_test,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE
    )
    
    # Train with PyTorch
    pytorch_results = train_pytorch(
        X_train, y_train, X_test, y_test,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE
    )
    
    # Print summary
    print_summary(nashai_results, pytorch_results)
    
    # Plot comparison
    try:
        plot_comparison(nashai_results, pytorch_results)
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    return nashai_results, pytorch_results


if __name__ == '__main__':
    main()
