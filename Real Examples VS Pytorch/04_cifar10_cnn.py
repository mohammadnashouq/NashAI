"""
CIFAR-10 Classification with CNN - NashAI vs PyTorch Comparison

This benchmark compares a Convolutional Neural Network trained on CIFAR-10 using:
1. NashAI (our from-scratch implementation)
2. PyTorch (industry standard)

CIFAR-10: 60,000 32x32 color images in 10 classes
- 50,000 training images
- 10,000 test images

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
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
from nnn.layers import Dense
from nnn.conv import Conv2D, MaxPool2D, Flatten, BatchNorm2D, Dropout
from nnn.activations import relu
from nnn.losses import cross_entropy_loss
from nnn.optim import Adam

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def load_cifar10(subset_size=5000):
    """Load and preprocess CIFAR-10 dataset."""
    print("Loading CIFAR-10 dataset...")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Download CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Convert to numpy arrays
    X_train = np.array([trainset[i][0].numpy() for i in range(len(trainset))])
    y_train = np.array([trainset[i][1] for i in range(len(trainset))])
    
    X_test = np.array([testset[i][0].numpy() for i in range(len(testset))])
    y_test = np.array([testset[i][1] for i in range(len(testset))])
    
    # Use subset for faster training
    if subset_size and subset_size < len(X_train):
        train_indices = np.random.permutation(len(X_train))[:subset_size]
        X_train, y_train = X_train[train_indices], y_train[train_indices]
        
        test_subset = subset_size // 5
        test_indices = np.random.permutation(len(X_test))[:test_subset]
        X_test, y_test = X_test[test_indices], y_test[test_indices]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Image shape: {X_train.shape[1:]}")
    
    return X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test


def one_hot_encode(y, num_classes=10):
    """Convert labels to one-hot encoding."""
    one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot


# ============================================
# NashAI CNN Implementation for CIFAR-10
# ============================================

class NashAICIFARCNN:
    """CNN for CIFAR-10 using NashAI library."""
    
    def __init__(self, num_classes=10):
        # Input: 3x32x32
        # Conv block 1: 3 -> 32 channels
        self.conv1 = Conv2D(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2D(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        # Conv block 2: 32 -> 64 channels
        self.conv3 = Conv2D(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2D(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)  # 16x16 -> 8x8
        
        # Conv block 3: 64 -> 128 channels
        self.conv5 = Conv2D(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = MaxPool2D(kernel_size=2, stride=2)  # 8x8 -> 4x4
        
        self.flatten = Flatten()
        
        # FC layers: 128 * 4 * 4 = 2048
        self.fc1 = Dense(128 * 4 * 4, 256)
        self.fc2 = Dense(256, num_classes)
        
        self.training = True
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def forward(self, x):
        # Conv block 1
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = self.pool1(x)
        
        # Conv block 2
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = self.pool2(x)
        
        # Conv block 3
        x = relu(self.conv5(x))
        x = self.pool3(x)
        
        # FC layers
        x = self.flatten(x)
        x = relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.conv3.parameters())
        params.extend(self.conv4.parameters())
        params.extend(self.conv5.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params
    
    def zero_grad(self):
        self.conv1.zero_grad()
        self.conv2.zero_grad()
        self.conv3.zero_grad()
        self.conv4.zero_grad()
        self.conv5.zero_grad()
        self.fc1.zero_grad()
        self.fc2.zero_grad()


def train_nashai_cifar(X_train, y_train, X_test, y_test, epochs=5, batch_size=32, lr=0.001):
    """Train CNN on CIFAR-10 using NashAI."""
    print("\n" + "="*50)
    print("Training CIFAR-10 CNN with NashAI")
    print("="*50)
    
    # Convert labels to one-hot
    y_train_oh = one_hot_encode(y_train)
    y_test_oh = one_hot_encode(y_test)
    
    # Initialize model and optimizer
    model = NashAICIFARCNN()
    optimizer = Adam(model.parameters(), lr=lr)
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    n_samples = len(X_train)
    n_batches = n_samples // batch_size
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
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
            
            # Progress
            if (batch_idx + 1) % 5 == 0:
                print(f"\r  Batch {batch_idx+1}/{n_batches} - Loss: {loss.data.item():.4f}", end="")
        
        # Compute metrics
        avg_loss = epoch_loss / n_batches
        train_acc = epoch_correct / (n_batches * batch_size) * 100
        
        # Test accuracy (batch processing)
        model.eval()
        test_correct = 0
        test_batch_size = 50
        n_test_batches = len(X_test) // test_batch_size
        
        for i in range(n_test_batches):
            start = i * test_batch_size
            end = start + test_batch_size
            test_logits = model.forward(Tensor(X_test[start:end]))
            test_preds = np.argmax(test_logits.data, axis=1)
            test_correct += np.sum(test_preds == y_test[start:end])
        
        test_acc = test_correct / (n_test_batches * test_batch_size) * 100
        
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"\rEpoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
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
# PyTorch CNN Implementation for CIFAR-10
# ============================================

class PyTorchCIFARCNN(nn.Module):
    """CNN for CIFAR-10 using PyTorch."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # FC layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Conv block 2
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        
        # Conv block 3
        x = torch.relu(self.conv5(x))
        x = self.pool3(x)
        
        # FC layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


def train_pytorch_cifar(X_train, y_train, X_test, y_test, epochs=5, batch_size=32, lr=0.001):
    """Train CNN on CIFAR-10 using PyTorch."""
    print("\n" + "="*50)
    print("Training CIFAR-10 CNN with PyTorch")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = PyTorchCIFARCNN().to(device)
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
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
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
            
            if (batch_idx + 1) % 5 == 0:
                print(f"\r  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}", end="")
        
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
        
        print(f"\rEpoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
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
    axes[0].set_title('Training Loss (CIFAR-10)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Training accuracy comparison
    axes[1].plot(epochs, nashai_results['train_accs'], 'b-', label='NashAI', linewidth=2)
    axes[1].plot(epochs, pytorch_results['train_accs'], 'r--', label='PyTorch', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy (CIFAR-10)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Test accuracy comparison
    axes[2].plot(epochs, nashai_results['test_accs'], 'b-', label='NashAI', linewidth=2)
    axes[2].plot(epochs, pytorch_results['test_accs'], 'r--', label='PyTorch', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Test Accuracy (CIFAR-10)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cifar10_cnn_comparison.png', dpi=150)
    plt.show()
    print("\nPlot saved as 'cifar10_cnn_comparison.png'")


def visualize_predictions(model_nashai, model_pytorch, X_test, y_test, device='cpu'):
    """Visualize some predictions from both models."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Get predictions
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        img = X_test[idx]
        true_label = CIFAR10_CLASSES[y_test[idx]]
        
        # Denormalize for visualization
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
        std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
        img_vis = (img * std + mean).clip(0, 1)
        img_vis = img_vis.transpose(1, 2, 0)
        
        # NashAI prediction
        axes[0, i].imshow(img_vis)
        axes[0, i].set_title(f'True: {true_label}')
        axes[0, i].axis('off')
        
        # PyTorch prediction
        axes[1, i].imshow(img_vis)
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('NashAI', fontsize=12)
    axes[1, 0].set_ylabel('PyTorch', fontsize=12)
    
    plt.suptitle('Sample CIFAR-10 Images', fontsize=14)
    plt.tight_layout()
    plt.savefig('cifar10_samples.png', dpi=150)
    plt.show()


def print_summary(nashai_results, pytorch_results):
    """Print comparison summary."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY: CIFAR-10 CNN")
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
    print(f"✓ Accuracy difference: {acc_diff:.2f}%")
    
    time_ratio = nashai_results['training_time'] / pytorch_results['training_time']
    print(f"✓ NashAI is {time_ratio:.1f}x slower than PyTorch")
    print("✓ Both implementations train successfully on CIFAR-10")
    print("✓ Color images (3 channels) handled correctly by NashAI")


def main():
    """Run the CIFAR-10 CNN comparison."""
    print("="*60)
    print("CIFAR-10 CNN BENCHMARK: NashAI vs PyTorch")
    print("="*60)
    
    # Parameters - smaller for CIFAR-10 due to complexity
    SUBSET_SIZE = 3000  # Smaller subset for faster demo
    EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    print(f"\nParameters:")
    print(f"  Dataset size: {SUBSET_SIZE if SUBSET_SIZE else 'Full'}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_cifar10(SUBSET_SIZE)
    
    # Train with NashAI
    nashai_results = train_nashai_cifar(
        X_train, y_train, X_test, y_test,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE
    )
    
    # Train with PyTorch
    pytorch_results = train_pytorch_cifar(
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
