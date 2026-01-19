"""
Transformer architecture components.

Implements:
- Positional Encodings (sinusoidal and learned)
- Layer Normalization
- Multi-Head Self-Attention with causal masking
- Feed-Forward Network
- Residual connections
- TransformerBlock
"""

import numpy as np
from typing import Optional, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnn.tensor import Tensor


# =============================================================================
# Positional Encodings
# =============================================================================

class SinusoidalPositionalEncoding:
    """
    Sinusoidal positional encoding from "Attention Is All You Need".
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Properties:
    - Fixed (no learnable parameters)
    - Can extrapolate to longer sequences
    - Each dimension corresponds to a sinusoid
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.training = True
        
        # Precompute positional encodings
        self.pe = self._create_pe_matrix(max_seq_len, d_model)
    
    def _create_pe_matrix(self, max_seq_len: int, d_model: int) -> np.ndarray:
        """Create the positional encoding matrix."""
        pe = np.zeros((max_seq_len, d_model), dtype=np.float32)
        position = np.arange(max_seq_len).reshape(-1, 1)
        
        # Compute division term
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.shape[1]
        
        # Get positional encoding for sequence length
        pe = self.pe[:seq_len, :]  # (seq_len, d_model)
        pe = pe.reshape(1, seq_len, self.d_model)  # (1, seq_len, d_model)
        
        # Add positional encoding
        out_data = x.data + pe
        
        # Apply dropout during training
        if self.training and self.dropout > 0:
            mask = (np.random.rand(*out_data.shape) > self.dropout).astype(np.float32)
            scale = 1.0 / (1.0 - self.dropout)
            out_data = out_data * mask * scale
        
        out = Tensor(out_data, requires_grad=x.requires_grad)
        
        if x.requires_grad:
            def _backward(grad):
                if self.training and self.dropout > 0:
                    grad_x = grad * mask * scale
                else:
                    grad_x = grad
                return (grad_x,)
            
            out._backward_fn = _backward
            out._children = (x,)
        
        return out
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class LearnedPositionalEncoding:
    """
    Learned positional encoding.
    
    Each position has a learnable embedding vector.
    Used in GPT and many modern models.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.training = True
        
        # Learnable position embeddings
        self.pe = Tensor(
            np.random.randn(max_seq_len, d_model).astype(np.float32) * 0.02,
            requires_grad=True
        )
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Add learned positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.shape[1]
        
        # Get positional encoding for sequence length
        pe = self.pe.data[:seq_len, :]  # (seq_len, d_model)
        pe = pe.reshape(1, seq_len, self.d_model)  # (1, seq_len, d_model)
        
        # Add positional encoding
        out_data = x.data + pe
        
        # Apply dropout during training
        if self.training and self.dropout > 0:
            mask = (np.random.rand(*out_data.shape) > self.dropout).astype(np.float32)
            scale = 1.0 / (1.0 - self.dropout)
            out_data = out_data * mask * scale
        else:
            mask = None
            scale = 1.0
        
        out = Tensor(out_data, requires_grad=x.requires_grad or self.pe.requires_grad)
        
        if out.requires_grad:
            def _backward(grad):
                grads = []
                
                # Gradient for input x
                if x.requires_grad:
                    if self.training and self.dropout > 0:
                        grad_x = grad * mask * scale
                    else:
                        grad_x = grad
                    grads.append(grad_x)
                else:
                    grads.append(None)
                
                # Gradient for positional encoding
                if self.pe.requires_grad:
                    if self.training and self.dropout > 0:
                        grad_pe_full = grad * mask * scale
                    else:
                        grad_pe_full = grad
                    # Sum over batch dimension
                    grad_pe = np.sum(grad_pe_full, axis=0)  # (seq_len, d_model)
                    
                    # Accumulate gradient (pad to full size)
                    if self.pe.grad is None:
                        self.pe.grad = np.zeros_like(self.pe.data)
                    self.pe.grad[:seq_len, :] += grad_pe
                
                return tuple(grads)
            
            out._backward_fn = _backward
            out._children = (x,)
        
        return out
    
    def parameters(self) -> List[Tensor]:
        return [self.pe]
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


# =============================================================================
# Layer Normalization
# =============================================================================

class LayerNorm:
    """
    Layer Normalization.
    
    Normalizes across the last dimension (features).
    
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Unlike BatchNorm:
    - Normalizes across features, not batch
    - Same behavior at train and test time
    - Works with any batch size
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.gamma = Tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
        
        Returns:
            Normalized tensor
        """
        # Compute mean and variance along last dimension
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        out_data = self.gamma.data * x_norm + self.beta.data
        
        out = Tensor(out_data, requires_grad=x.requires_grad or self.gamma.requires_grad)
        
        if out.requires_grad:
            def _backward(grad):
                N = x.data.shape[-1]  # normalized_shape
                
                # Gradients for gamma and beta
                if self.gamma.requires_grad:
                    grad_gamma = np.sum(grad * x_norm, axis=tuple(range(len(grad.shape) - 1)))
                    if self.gamma.grad is None:
                        self.gamma.grad = np.zeros_like(self.gamma.data)
                    self.gamma.grad += grad_gamma
                
                if self.beta.requires_grad:
                    grad_beta = np.sum(grad, axis=tuple(range(len(grad.shape) - 1)))
                    if self.beta.grad is None:
                        self.beta.grad = np.zeros_like(self.beta.data)
                    self.beta.grad += grad_beta
                
                # Gradient for input
                if x.requires_grad:
                    # d/dx LayerNorm
                    std_inv = 1.0 / np.sqrt(var + self.eps)
                    dx_norm = grad * self.gamma.data
                    
                    dvar = np.sum(dx_norm * (x.data - mean) * -0.5 * (var + self.eps) ** (-1.5), axis=-1, keepdims=True)
                    dmean = np.sum(dx_norm * -std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (x.data - mean), axis=-1, keepdims=True)
                    
                    grad_x = dx_norm * std_inv + dvar * 2.0 * (x.data - mean) / N + dmean / N
                    return (grad_x,)
                return (None,)
            
            out._backward_fn = _backward
            out._children = (x,)
        
        return out
    
    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


# =============================================================================
# Feed-Forward Network
# =============================================================================

class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Typically d_ff = 4 * d_model
    """
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.dropout = dropout
        self.training = True
        
        # Linear layers
        limit1 = np.sqrt(6.0 / (d_model + self.d_ff))
        limit2 = np.sqrt(6.0 / (self.d_ff + d_model))
        
        self.W1 = Tensor(
            np.random.uniform(-limit1, limit1, (d_model, self.d_ff)).astype(np.float32),
            requires_grad=True
        )
        self.b1 = Tensor(np.zeros(self.d_ff, dtype=np.float32), requires_grad=True)
        
        self.W2 = Tensor(
            np.random.uniform(-limit2, limit2, (self.d_ff, d_model)).astype(np.float32),
            requires_grad=True
        )
        self.b2 = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear + GELU activation
        hidden = x.data @ self.W1.data + self.b1.data
        
        # GELU activation (used in GPT-2, BERT)
        hidden_gelu = 0.5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + 0.044715 * hidden ** 3)))
        
        # Dropout
        if self.training and self.dropout > 0:
            mask1 = (np.random.rand(*hidden_gelu.shape) > self.dropout).astype(np.float32)
            scale = 1.0 / (1.0 - self.dropout)
            hidden_gelu = hidden_gelu * mask1 * scale
        else:
            mask1 = None
            scale = 1.0
        
        # Second linear
        out_data = hidden_gelu @ self.W2.data + self.b2.data
        
        out = Tensor(out_data, requires_grad=x.requires_grad)
        
        if x.requires_grad:
            def _backward(grad):
                # Gradient through second linear
                grad_hidden_gelu = grad @ self.W2.data.T
                
                if self.W2.requires_grad:
                    # Reshape for matmul: (batch * seq, d_ff).T @ (batch * seq, d_model)
                    hidden_flat = hidden_gelu.reshape(-1, self.d_ff)
                    grad_flat = grad.reshape(-1, self.d_model)
                    grad_W2 = hidden_flat.T @ grad_flat
                    if self.W2.grad is None:
                        self.W2.grad = np.zeros_like(self.W2.data)
                    self.W2.grad += grad_W2
                
                if self.b2.requires_grad:
                    grad_b2 = np.sum(grad, axis=(0, 1))
                    if self.b2.grad is None:
                        self.b2.grad = np.zeros_like(self.b2.data)
                    self.b2.grad += grad_b2
                
                # Gradient through dropout
                if self.training and self.dropout > 0:
                    grad_hidden_gelu = grad_hidden_gelu * mask1 * scale
                
                # Gradient through GELU
                # d/dx GELU(x) ≈ 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * d/dx(...)
                sqrt_2_pi = np.sqrt(2 / np.pi)
                tanh_arg = sqrt_2_pi * (hidden + 0.044715 * hidden ** 3)
                tanh_val = np.tanh(tanh_arg)
                sech2 = 1 - tanh_val ** 2
                dtanh_arg = sqrt_2_pi * (1 + 3 * 0.044715 * hidden ** 2)
                gelu_grad = 0.5 * (1 + tanh_val) + 0.5 * hidden * sech2 * dtanh_arg
                grad_hidden = grad_hidden_gelu * gelu_grad
                
                # Gradient through first linear
                grad_x = grad_hidden @ self.W1.data.T
                
                if self.W1.requires_grad:
                    x_flat = x.data.reshape(-1, self.d_model)
                    grad_hidden_flat = grad_hidden.reshape(-1, self.d_ff)
                    grad_W1 = x_flat.T @ grad_hidden_flat
                    if self.W1.grad is None:
                        self.W1.grad = np.zeros_like(self.W1.data)
                    self.W1.grad += grad_W1
                
                if self.b1.requires_grad:
                    grad_b1 = np.sum(grad_hidden, axis=(0, 1))
                    if self.b1.grad is None:
                        self.b1.grad = np.zeros_like(self.b1.data)
                    self.b1.grad += grad_b1
                
                return (grad_x,)
            
            out._backward_fn = _backward
            out._children = (x,)
        
        return out
    
    def parameters(self) -> List[Tensor]:
        return [self.W1, self.b1, self.W2, self.b2]
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


# =============================================================================
# Multi-Head Self-Attention with Causal Masking
# =============================================================================

class CausalSelfAttention:
    """
    Multi-Head Self-Attention with causal (autoregressive) masking.
    
    Each position can only attend to previous positions.
    Used in decoder-only models like GPT.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 512):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.training = True
        
        # Query, Key, Value projections (combined for efficiency)
        limit = np.sqrt(6.0 / (d_model + d_model))
        self.W_qkv = Tensor(
            np.random.uniform(-limit, limit, (d_model, 3 * d_model)).astype(np.float32),
            requires_grad=True
        )
        self.b_qkv = Tensor(np.zeros(3 * d_model, dtype=np.float32), requires_grad=True)
        
        # Output projection
        self.W_o = Tensor(
            np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32),
            requires_grad=True
        )
        self.b_o = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
        
        # Precompute causal mask
        self.causal_mask = self._create_causal_mask(max_seq_len)
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal attention mask."""
        # Lower triangular matrix: position i can attend to positions 0..i
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
        mask = mask * -1e9  # Large negative for masked positions
        return mask
    
    def __call__(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        """
        Apply causal self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional additional mask
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = x.data @ self.W_qkv.data + self.b_qkv.data  # (batch, seq, 3 * d_model)
        q, k, v = np.split(qkv, 3, axis=-1)  # Each: (batch, seq, d_model)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        # Now: (batch, heads, seq, d_k)
        
        # Compute attention scores
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)  # (batch, heads, seq, seq)
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores + causal_mask.reshape(1, 1, seq_len, seq_len)
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
        
        # Apply dropout to attention weights
        if self.training and self.dropout > 0:
            attn_mask = (np.random.rand(*attention_weights.shape) > self.dropout).astype(np.float32)
            attn_scale = 1.0 / (1.0 - self.dropout)
            attention_weights_dropped = attention_weights * attn_mask * attn_scale
        else:
            attention_weights_dropped = attention_weights
            attn_mask = None
            attn_scale = 1.0
        
        # Apply attention to values
        context = attention_weights_dropped @ v  # (batch, heads, seq, d_k)
        
        # Reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        out_data = context @ self.W_o.data + self.b_o.data
        
        out = Tensor(out_data, requires_grad=x.requires_grad)
        
        if x.requires_grad:
            def _backward(grad):
                # Gradient through output projection
                grad_context = grad @ self.W_o.data.T
                
                if self.W_o.requires_grad:
                    context_flat = context.reshape(-1, self.d_model)
                    grad_flat = grad.reshape(-1, self.d_model)
                    grad_W_o = context_flat.T @ grad_flat
                    if self.W_o.grad is None:
                        self.W_o.grad = np.zeros_like(self.W_o.data)
                    self.W_o.grad += grad_W_o
                
                if self.b_o.requires_grad:
                    grad_b_o = np.sum(grad, axis=(0, 1))
                    if self.b_o.grad is None:
                        self.b_o.grad = np.zeros_like(self.b_o.data)
                    self.b_o.grad += grad_b_o
                
                # Reshape gradient for multi-head
                grad_context = grad_context.reshape(batch_size, seq_len, self.num_heads, self.d_k)
                grad_context = grad_context.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)
                
                # Gradient through attention @ v
                grad_attn_weights = grad_context @ v.transpose(0, 1, 3, 2)
                grad_v = attention_weights_dropped.transpose(0, 1, 3, 2) @ grad_context
                
                # Gradient through dropout
                if self.training and self.dropout > 0:
                    grad_attn_weights = grad_attn_weights * attn_mask * attn_scale
                
                # Gradient through softmax
                sum_grad_attn = np.sum(grad_attn_weights * attention_weights, axis=-1, keepdims=True)
                grad_scores = attention_weights * (grad_attn_weights - sum_grad_attn)
                
                # Gradient through scaling
                grad_scores = grad_scores / np.sqrt(self.d_k)
                
                # Gradient through Q @ K^T
                grad_q = grad_scores @ k
                grad_k = grad_scores.transpose(0, 1, 3, 2) @ q
                
                # Reshape back
                grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
                grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
                grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
                
                # Combine Q, K, V gradients
                grad_qkv = np.concatenate([grad_q, grad_k, grad_v], axis=-1)
                
                # Gradient through QKV projection
                grad_x = grad_qkv @ self.W_qkv.data.T
                
                if self.W_qkv.requires_grad:
                    x_flat = x.data.reshape(-1, self.d_model)
                    grad_qkv_flat = grad_qkv.reshape(-1, 3 * self.d_model)
                    grad_W_qkv = x_flat.T @ grad_qkv_flat
                    if self.W_qkv.grad is None:
                        self.W_qkv.grad = np.zeros_like(self.W_qkv.data)
                    self.W_qkv.grad += grad_W_qkv
                
                if self.b_qkv.requires_grad:
                    grad_b_qkv = np.sum(grad_qkv, axis=(0, 1))
                    if self.b_qkv.grad is None:
                        self.b_qkv.grad = np.zeros_like(self.b_qkv.data)
                    self.b_qkv.grad += grad_b_qkv
                
                return (grad_x,)
            
            out._backward_fn = _backward
            out._children = (x,)
        
        return out
    
    def parameters(self) -> List[Tensor]:
        return [self.W_qkv, self.b_qkv, self.W_o, self.b_o]
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock:
    """
    A single Transformer decoder block.
    
    Architecture:
    x -> LayerNorm -> CausalSelfAttention -> + (residual) -> LayerNorm -> FFN -> + (residual) -> out
    
    Uses Pre-LayerNorm architecture (more stable for training).
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, dropout: float = 0.1, max_seq_len: int = 512):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff or 4 * d_model
        self.dropout = dropout
        self.training = True
        
        # Pre-norm architecture
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout, max_seq_len)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, self.d_ff, dropout)
    
    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        # Pre-norm: LayerNorm before attention
        attn_out = self.attn(self.ln1(x), mask)
        
        # Residual connection
        x_attn = Tensor(x.data + attn_out.data, requires_grad=x.requires_grad or attn_out.requires_grad)
        
        if x_attn.requires_grad:
            def _backward_attn(grad):
                grads = []
                if x.requires_grad:
                    grads.append(grad)
                else:
                    grads.append(None)
                if attn_out.requires_grad:
                    attn_out.backward(grad)
                return tuple(grads)
            
            x_attn._backward_fn = _backward_attn
            x_attn._children = (x,)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(self.ln2(x_attn))
        
        # Residual connection
        out_data = x_attn.data + ffn_out.data
        out = Tensor(out_data, requires_grad=x_attn.requires_grad or ffn_out.requires_grad)
        
        if out.requires_grad:
            def _backward_ffn(grad):
                grads = []
                if x_attn.requires_grad:
                    grads.append(grad)
                    x_attn.backward(grad)
                else:
                    grads.append(None)
                if ffn_out.requires_grad:
                    ffn_out.backward(grad)
                return tuple(grads)
            
            out._backward_fn = _backward_ffn
            out._children = (x_attn,)
        
        return out
    
    def __call__(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        return self.forward(x, mask)
    
    def parameters(self) -> List[Tensor]:
        params = []
        params.extend(self.ln1.parameters())
        params.extend(self.attn.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.ffn.parameters())
        return params
    
    def train(self):
        self.training = True
        self.attn.train()
        self.ffn.train()
    
    def eval(self):
        self.training = False
        self.attn.eval()
        self.ffn.eval()


# =============================================================================
# Utility Functions
# =============================================================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """Create causal attention mask."""
    mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
    mask = mask * -1e9
    return mask


def create_padding_mask(lengths: List[int], max_len: int) -> np.ndarray:
    """
    Create padding mask for variable length sequences.
    
    Args:
        lengths: List of sequence lengths
        max_len: Maximum sequence length
    
    Returns:
        Mask of shape (batch_size, 1, 1, max_len)
    """
    batch_size = len(lengths)
    mask = np.zeros((batch_size, 1, 1, max_len), dtype=np.float32)
    
    for i, length in enumerate(lengths):
        if length < max_len:
            mask[i, 0, 0, length:] = -1e9
    
    return mask
