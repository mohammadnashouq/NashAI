# NashAI

A comprehensive Python library for AI, built from scratch with a focus on understanding the mathematical and theoretical foundations behind machine learning and deep learning.

## Goal

The goal of this project is to create a complete AI library that implements everything from mathematical foundations to modern deep learning architectures, all built from scratch. This library serves as both a learning resource and a production-ready toolkit for AI research and development.

## High-Level Strategy

You will progress in layers, each one building on the previous:

1. **Mathematical & theoretical foundations**
2. **Classical Machine Learning**
3. **Deep Learning from scratch**
4. **Modern architectures (Transformers, LLMs, VLMs)**
5. **Systems & optimization**
6. **Packaging as a clean, extensible Python library**
7. **Research-level extensions**

Each concept you learn:

📘 Read theory → ✍️ implement from scratch → 🧪 test → 📦 refactor into your library

---

## 🧱 PHASE 0 — Prerequisites (Non-Negotiable)

### Mathematics (Implement Everything)

You should code the math, not just read it.

### Topics

- Linear algebra (vectors, matrices, eigenvalues)
- Probability & statistics
- Optimization
- Information theory

### Implementation Tasks

```python
# Example targets
- Vector / Matrix class
- Dot product, norms
- Eigen decomposition
- Gradient descent optimizer
- Numerical differentiation
```

### Resources

- *Linear Algebra Done Right* – Axler
- *Pattern Recognition and Machine Learning* – Bishop
- *Deep Learning* – Goodfellow (math chapters)

### 📦 Library module

```
yourlib/
 └── math/
     ├── linalg.py
     ├── probability.py
     ├── optimization.py
```

---

## 🤖 PHASE 1 — Classical Machine Learning (From Scratch)

Implement without sklearn first.

### Algorithms to Implement

**Supervised Learning**
- Linear Regression
- Logistic Regression
- k-NN
- Naive Bayes
- SVM
- Decision Trees
- Random Forest

**Unsupervised Learning**
- k-Means
- GMM
- PCA
- ICA

### Core Concepts to Master

- Bias-variance tradeoff
- Loss functions
- Regularization
- Cross-validation

### Example Implementation Rule

```python
class LogisticRegression:
    def fit(self, X, y): ...
    def predict(self, X): ...
    def loss(self, X, y): ...
```

### 📦 Library module

```
yourlib/
 └── ml/
     ├── linear_models.py
     ├── trees.py
     ├── clustering.py
     ├── decomposition.py
```

### 📚 Reference

- *Hands-On ML* (theory only, not code)
- *Elements of Statistical Learning*

---

## 🔥 PHASE 2 — Deep Learning (NO PYTORCH AT FIRST)

You must build your own autograd engine.

### Step 1: Autograd Engine

- Computational graph
- Forward pass
- Backpropagation
- Chain rule

**Inspired by:**
- Karpathy's micrograd

```python
class Tensor:
    def backward(self): ...
```

### Step 2: Neural Network Components

- Dense layers
- Activations (ReLU, GELU, Softmax)
- Losses (MSE, Cross-Entropy)
- Optimizers (SGD, Adam)

### Step 3: Training Loop

```python
for batch in data:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### 📦 Library module

```
yourlib/
 └── nn/
     ├── tensor.py
     ├── layers.py
     ├── activations.py
     ├── losses.py
     ├── optim.py
```

### 📚 Reference

- *Deep Learning* – Goodfellow
- CS231n
- Karpathy videos

---

## 🧠 PHASE 3 — CNNs, RNNs, and Attention

### Implement From Scratch

- Convolutions
- Pooling
- BatchNorm
- Dropout
- RNN / LSTM / GRU
- Attention (scaled dot-product)

### Key Focus

- Shape management
- Memory efficiency
- Gradient flow issues

### 📦 Library module

```
yourlib/
 └── nn/
     ├── conv.py
     ├── rnn.py
     ├── attention.py
```

---

## 🧬 PHASE 4 — TRANSFORMERS & LLMS

This is where you become elite.

### Core Concepts

- Tokenization (BPE, WordPiece)
- Positional encodings
- Self-attention
- LayerNorm
- Residual connections
- Causal masking

### Implement a GPT-like Model

```python
class TransformerBlock:
    def forward(self, x): ...
```

**Then:**
- Language modeling
- Pretraining loop
- Sampling (top-k, nucleus)

### 📦 Library module

```
yourlib/
 └── llm/
     ├── tokenizer.py
     ├── transformer.py
     ├── gpt.py
     ├── sampling.py
```

### 📚 Reference

- *Attention Is All You Need*
- nanoGPT
- GPT-2 paper

---

## 👁️ PHASE 5 — Vision & VLMs

### Vision Models

- CNNs
- Vision Transformers
- Image embeddings

### Multimodal

- CLIP-style contrastive learning
- Image encoder + text encoder
- Shared embedding space

### 📦 Library module

```
yourlib/
 └── vlm/
     ├── vision_encoder.py
     ├── text_encoder.py
     ├── clip.py
```

### 📚 Reference

- CLIP paper
- ViT paper

---

## ⚙️ PHASE 6 — Systems & Performance

### Topics

- GPU kernels (CUDA later)
- Mixed precision
- Checkpointing
- Memory optimization
- Distributed training (conceptually)

You can bridge to PyTorch here and compare behavior.

---

## 📦 PHASE 7 — Open-Source Library Design

### Repo Structure

```
yourlib/
 ├── yourlib/
 │   ├── math/
 │   ├── ml/
 │   ├── nn/
 │   ├── llm/
 │   └── vlm/
 ├── tests/
 ├── examples/
 ├── docs/
 └── README.md
```

### Best Practices

- Full docstrings
- Type hints
- Unit tests
- Reproducible experiments
- Clear API consistency

---

## 🧪 PHASE 8 — Research Extensions (Optional but Powerful)

- Sparse attention
- MoE
- Quantization
- RLHF
- Multimodal agents
- New loss functions
- Diffusion Models
- Conditional Difuusion Models
- Auto encoders
- Conditional Auto Encoders
- Gan networks.
- Simolated anyling.
- Deep Fake Network.
- Comparision wiht pytorch.

This is where papers → code happens.

---

## 🧠 How Long This Takes (Realistic)

| Phase | Time |
|-------|------|
| Foundations | 1–2 months |
| ML | 1–2 months |
| DL core | 2 months |
| Transformers | 2 months |
| VLMs | 1–2 months |
| Polish | ongoing |

**⏱ 6–10 months of serious work**

---

## 🚀 Final Advice (Very Important)

- Never copy code blindly
- Write before reading implementations
- Use PyTorch only to verify correctness
- Explain every module in README
- Teach through your code

---

## Getting Started

This library is currently in active development. The structure is being built phase by phase, starting with mathematical foundations.

### Installation

```bash
# Coming soon
pip install nashai
```

### Usage

```python
# Coming soon
import nashai
```

---

## Contributing

Contributions are welcome! This is a learning project, so feel free to open issues, submit pull requests, or start discussions about implementations.

---

## License

[To be determined]

---

## Roadmap

- [x] Phase 0: Project structure and README
- [ ] Phase 0: Mathematical foundations implementation
- [ ] Phase 1: Classical ML algorithms
- [ ] Phase 2: Deep Learning from scratch
- [ ] Phase 3: CNNs, RNNs, Attention
- [ ] Phase 4: Transformers & LLMs
- [ ] Phase 5: Vision & VLMs
- [ ] Phase 6: Systems & Performance
- [ ] Phase 7: Library polish
- [ ] Phase 8: Research extensions

