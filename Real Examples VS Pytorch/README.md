# NashAI vs PyTorch - Real-World Comparisons

This directory contains comprehensive benchmarks comparing NashAI (our from-scratch implementation) with PyTorch on standard datasets.

## Benchmarks Overview

| Example | Dataset | Model Type | Metrics |
|---------|---------|------------|---------|
| `01_mnist_mlp.py` | MNIST | MLP (Dense layers) | Accuracy, Training Time, Loss |
| `02_mnist_cnn.py` | MNIST | CNN (Conv + Pool) | Accuracy, Training Time, Loss |
| `03_classical_ml.py` | Iris, Boston, Synthetic | Linear/Logistic Regression, SVM | Accuracy/MSE, Training Time |
| `04_cifar10_cnn.py` | CIFAR-10 | Deep CNN | Accuracy, Training Time, Loss |
| `05_sequence_rnn.py` | Synthetic Sequences | RNN/LSTM/GRU | Accuracy, Training Time |

## Running the Benchmarks

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

### Run Individual Benchmarks

```bash
# MNIST with MLP
python 01_mnist_mlp.py

# MNIST with CNN
python 02_mnist_cnn.py

# Classical ML algorithms
python 03_classical_ml.py

# CIFAR-10 CNN
python 04_cifar10_cnn.py

# Sequence modeling
python 05_sequence_rnn.py
```

### Run All Benchmarks

```bash
python run_all_benchmarks.py
```

## Expected Results

### Performance Notes

- **NashAI** is implemented in pure NumPy/Python without GPU acceleration
- **PyTorch** uses optimized C++ backends and optional CUDA
- NashAI will be slower but should achieve comparable accuracy
- These benchmarks demonstrate correctness, not production speed

### Typical Results

| Benchmark | NashAI Accuracy | PyTorch Accuracy | NashAI Time | PyTorch Time |
|-----------|-----------------|------------------|-------------|--------------|
| MNIST MLP | ~97% | ~98% | ~2-5min | ~10-30s |
| MNIST CNN | ~98% | ~99% | ~10-20min | ~1-2min |
| Logistic Reg | ~95% | ~95% | ~1-2s | ~0.5s |

## Understanding the Comparisons

Each benchmark file follows the same structure:

1. **Data Loading**: Load and preprocess dataset
2. **NashAI Implementation**: Build and train model using NashAI
3. **PyTorch Implementation**: Build and train equivalent model using PyTorch
4. **Results Comparison**: Compare accuracy, loss curves, and timing
5. **Visualization**: Plot training curves and results

## Key Insights

### Why NashAI is Slower

1. **No SIMD optimization**: NumPy operations are not as optimized as PyTorch's ATen
2. **No GPU**: NashAI runs on CPU only
3. **Python overhead**: Pure Python function calls vs compiled code
4. **No operator fusion**: Each operation is executed separately

### Why Accuracy Should Match

1. **Same algorithms**: Both implement the same mathematical operations
2. **Same architectures**: Identical network structures
3. **Same optimization**: SGD/Adam with same hyperparameters
4. **Numerical precision**: Both use float32 by default

## Contributing

Feel free to add more benchmarks! Follow the existing pattern:

```python
def run_nashai_benchmark():
    # NashAI implementation
    pass

def run_pytorch_benchmark():
    # PyTorch implementation
    pass

def compare_results(nashai_results, pytorch_results):
    # Analysis and visualization
    pass
```
