"""
Attention mechanisms for neural networks.

This module implements:
- Scaled Dot-Product Attention
- Multi-Head Attention

Key concepts:
- Scaling by sqrt(d_k) to prevent large dot products
- Masking for causal/padding attention
- Multiple attention heads for richer representations
"""

import numpy as np
from typing import Optional, List, Tuple
from .tensor import Tensor
from .activations import softmax


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Scaled Dot-Product Attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        query: Query tensor of shape (batch, seq_len_q, d_k)
        key: Key tensor of shape (batch, seq_len_k, d_k)
        value: Value tensor of shape (batch, seq_len_k, d_v)
        mask: Optional mask tensor. Use -inf for positions to mask.
              Shape: (batch, seq_len_q, seq_len_k) or broadcastable
        dropout_p: Dropout probability on attention weights
        training: Whether in training mode (for dropout)
        
    Returns:
        output: Attention output of shape (batch, seq_len_q, d_v)
        attention_weights: Attention weights of shape (batch, seq_len_q, seq_len_k)
    """
    d_k = query.shape[-1]
    
    # Compute attention scores: Q @ K^T
    # query: (batch, seq_q, d_k), key: (batch, seq_k, d_k)
    # key transposed: (batch, d_k, seq_k)
    # scores: (batch, seq_q, seq_k)
    
    # Manual transpose of key for batch matrix multiplication
    key_T_data = key.data.transpose(0, 2, 1)
    
    # Compute Q @ K^T
    scores_data = query.data @ key_T_data
    
    # Scale by sqrt(d_k)
    scale = np.sqrt(d_k)
    scores_data = scores_data / scale
    
    # Apply mask (add -inf to masked positions)
    if mask is not None:
        scores_data = scores_data + mask.data
    
    # Apply softmax to get attention weights
    # Softmax along last dimension (over keys)
    scores_max = np.max(scores_data, axis=-1, keepdims=True)
    scores_exp = np.exp(scores_data - scores_max)
    attention_weights_data = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
    
    # Apply dropout to attention weights if training
    if training and dropout_p > 0:
        dropout_mask = (np.random.rand(*attention_weights_data.shape) > dropout_p).astype(np.float32)
        attention_weights_data = attention_weights_data * dropout_mask / (1 - dropout_p)
    else:
        dropout_mask = None
    
    # Apply attention to values: weights @ V
    # attention_weights: (batch, seq_q, seq_k), value: (batch, seq_k, d_v)
    # output: (batch, seq_q, d_v)
    output_data = attention_weights_data @ value.data
    
    # Create output tensors
    requires_grad = query.requires_grad or key.requires_grad or value.requires_grad
    
    output = Tensor(
        output_data,
        requires_grad=requires_grad,
        _op='attention',
        _children=(query, key, value)
    )
    
    attention_weights = Tensor(
        attention_weights_data,
        requires_grad=requires_grad,
        _op='attention_weights'
    )
    
    # Cache for backward
    cache = {
        'query': query,
        'key': key,
        'value': value,
        'attention_weights': attention_weights_data,
        'scale': scale,
        'dropout_mask': dropout_mask,
        'dropout_p': dropout_p
    }
    
    def _backward(grad):
        q, k, v = cache['query'], cache['key'], cache['value']
        attn_weights = cache['attention_weights']
        scale = cache['scale']
        
        # Gradient w.r.t. value
        # output = attn_weights @ v
        # grad_v = attn_weights^T @ grad_output
        grad_v = attn_weights.transpose(0, 2, 1) @ grad if v.requires_grad else None
        
        # Gradient w.r.t. attention weights
        # grad_attn = grad_output @ v^T
        grad_attn = grad @ v.data.transpose(0, 2, 1)
        
        # Apply dropout gradient
        if cache['dropout_mask'] is not None:
            grad_attn = grad_attn * cache['dropout_mask'] / (1 - cache['dropout_p'])
        
        # Gradient through softmax
        # d(softmax)/dx_i = softmax_i * (delta_ij - softmax_j)
        # For each row: grad_scores = attn * (grad_attn - sum(grad_attn * attn))
        grad_attn_sum = np.sum(grad_attn * attn_weights, axis=-1, keepdims=True)
        grad_scores = attn_weights * (grad_attn - grad_attn_sum)
        
        # Scale gradient
        grad_scores = grad_scores / scale
        
        # Gradient w.r.t. query
        # scores = q @ k^T, so grad_q = grad_scores @ k
        grad_q = grad_scores @ k.data if q.requires_grad else None
        
        # Gradient w.r.t. key
        # scores = q @ k^T, so grad_k = grad_scores^T @ q
        grad_k = grad_scores.transpose(0, 2, 1) @ q.data if k.requires_grad else None
        
        return (grad_q, grad_k, grad_v)
    
    output._backward_fn = _backward
    
    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-Head Attention.
    
    Splits the input into multiple heads, applies attention in parallel,
    then concatenates and projects the results.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
    where head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
    
    Input shape: (batch, seq_len, d_model)
    Output shape: (batch, seq_len, d_model)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_bias: bool = True
    ):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability on attention weights
            use_bias: Whether to use bias in projections
        """
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        self.use_bias = use_bias
        
        # Training mode
        self.training = True
        
        # Projection weights
        limit = np.sqrt(6.0 / (d_model + d_model))
        
        # Query projection
        self.W_q = Tensor(
            np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32),
            requires_grad=True
        )
        
        # Key projection
        self.W_k = Tensor(
            np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32),
            requires_grad=True
        )
        
        # Value projection
        self.W_v = Tensor(
            np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32),
            requires_grad=True
        )
        
        # Output projection
        self.W_o = Tensor(
            np.random.uniform(-limit, limit, (d_model, d_model)).astype(np.float32),
            requires_grad=True
        )
        
        if use_bias:
            self.b_q = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
            self.b_k = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
            self.b_v = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
            self.b_o = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
        else:
            self.b_q = self.b_k = self.b_v = self.b_o = None
        
        self._cache = {}
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
    
    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query tensor of shape (batch, seq_len_q, d_model)
            key: Key tensor of shape (batch, seq_len_k, d_model)
            value: Value tensor of shape (batch, seq_len_k, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Attention output of shape (batch, seq_len_q, d_model)
            attention_weights: Attention weights of shape (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Linear projections
        Q = query.data @ self.W_q.data
        K = key.data @ self.W_k.data
        V = value.data @ self.W_v.data
        
        if self.use_bias:
            Q = Q + self.b_q.data
            K = K + self.b_k.data
            V = V + self.b_v.data
        
        # Reshape for multi-head: (batch, seq, d_model) -> (batch, num_heads, seq, d_k)
        Q = Q.reshape(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Compute attention scores
        # Q: (batch, heads, seq_q, d_k), K: (batch, heads, seq_k, d_k)
        # scores: (batch, heads, seq_q, seq_k)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        
        # Apply mask
        if mask is not None:
            # Expand mask for heads if needed
            mask_data = mask.data
            if mask_data.ndim == 3:
                mask_data = mask_data[:, np.newaxis, :, :]
            scores = scores + mask_data
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attention_weights_data = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        
        # Apply dropout
        if self.training and self.dropout > 0:
            dropout_mask = (np.random.rand(*attention_weights_data.shape) > self.dropout).astype(np.float32)
            attention_weights_data = attention_weights_data * dropout_mask / (1 - self.dropout)
        else:
            dropout_mask = None
        
        # Apply attention to values
        # attention_weights: (batch, heads, seq_q, seq_k), V: (batch, heads, seq_k, d_k)
        # context: (batch, heads, seq_q, d_k)
        context = attention_weights_data @ V
        
        # Reshape back: (batch, heads, seq_q, d_k) -> (batch, seq_q, d_model)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)
        
        # Output projection
        output_data = context @ self.W_o.data
        if self.use_bias:
            output_data = output_data + self.b_o.data
        
        # Cache for backward
        self._cache['query'] = query
        self._cache['key'] = key
        self._cache['value'] = value
        self._cache['Q'] = Q
        self._cache['K'] = K
        self._cache['V'] = V
        self._cache['attention_weights'] = attention_weights_data
        self._cache['context'] = context
        self._cache['dropout_mask'] = dropout_mask
        
        # Create output tensors
        requires_grad = query.requires_grad or key.requires_grad or value.requires_grad
        
        output = Tensor(
            output_data,
            requires_grad=requires_grad,
            _op='multi_head_attention',
            _children=(query, key, value, self.W_q, self.W_k, self.W_v, self.W_o)
        )
        
        attention_weights = Tensor(
            attention_weights_data,
            requires_grad=False,  # Usually don't need gradients for attention weights
            _op='mha_weights'
        )
        
        def _backward(grad):
            batch_size, seq_len_q, _ = grad.shape
            seq_len_k = self._cache['K'].shape[2]
            
            Q = self._cache['Q']
            K = self._cache['K']
            V = self._cache['V']
            attn = self._cache['attention_weights']
            context = self._cache['context']
            
            # Gradient through output projection
            grad_context = grad @ self.W_o.data.T
            
            if self.W_o.requires_grad:
                grad_W_o = context.reshape(-1, self.d_model).T @ grad.reshape(-1, self.d_model)
                if self.W_o.grad is None:
                    self.W_o.grad = Tensor(np.zeros_like(self.W_o.data))
                self.W_o.grad.data += grad_W_o
            
            if self.use_bias and self.b_o.requires_grad:
                if self.b_o.grad is None:
                    self.b_o.grad = Tensor(np.zeros_like(self.b_o.data))
                self.b_o.grad.data += np.sum(grad, axis=(0, 1))
            
            # Reshape gradient for multi-head: (batch, seq_q, d_model) -> (batch, heads, seq_q, d_k)
            grad_context = grad_context.reshape(batch_size, seq_len_q, self.num_heads, self.d_k)
            grad_context = grad_context.transpose(0, 2, 1, 3)
            
            # Gradient through attention @ V
            grad_attn = grad_context @ V.transpose(0, 1, 3, 2)
            grad_V = attn.transpose(0, 1, 3, 2) @ grad_context
            
            # Apply dropout gradient
            if self._cache['dropout_mask'] is not None:
                grad_attn = grad_attn * self._cache['dropout_mask'] / (1 - self.dropout)
            
            # Gradient through softmax
            grad_attn_sum = np.sum(grad_attn * attn, axis=-1, keepdims=True)
            grad_scores = attn * (grad_attn - grad_attn_sum)
            
            # Scale gradient
            grad_scores = grad_scores / np.sqrt(self.d_k)
            
            # Gradient through Q @ K^T
            grad_Q = grad_scores @ K
            grad_K = grad_scores.transpose(0, 1, 3, 2) @ Q
            
            # Reshape back: (batch, heads, seq, d_k) -> (batch, seq, d_model)
            grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)
            grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_k, self.d_model)
            grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_k, self.d_model)
            
            # Gradient through input projections
            if self.W_q.requires_grad:
                grad_W_q = query.data.reshape(-1, self.d_model).T @ grad_Q.reshape(-1, self.d_model)
                if self.W_q.grad is None:
                    self.W_q.grad = Tensor(np.zeros_like(self.W_q.data))
                self.W_q.grad.data += grad_W_q
            
            if self.W_k.requires_grad:
                grad_W_k = key.data.reshape(-1, self.d_model).T @ grad_K.reshape(-1, self.d_model)
                if self.W_k.grad is None:
                    self.W_k.grad = Tensor(np.zeros_like(self.W_k.data))
                self.W_k.grad.data += grad_W_k
            
            if self.W_v.requires_grad:
                grad_W_v = value.data.reshape(-1, self.d_model).T @ grad_V.reshape(-1, self.d_model)
                if self.W_v.grad is None:
                    self.W_v.grad = Tensor(np.zeros_like(self.W_v.data))
                self.W_v.grad.data += grad_W_v
            
            if self.use_bias:
                if self.b_q.requires_grad:
                    if self.b_q.grad is None:
                        self.b_q.grad = Tensor(np.zeros_like(self.b_q.data))
                    self.b_q.grad.data += np.sum(grad_Q, axis=(0, 1))
                
                if self.b_k.requires_grad:
                    if self.b_k.grad is None:
                        self.b_k.grad = Tensor(np.zeros_like(self.b_k.data))
                    self.b_k.grad.data += np.sum(grad_K, axis=(0, 1))
                
                if self.b_v.requires_grad:
                    if self.b_v.grad is None:
                        self.b_v.grad = Tensor(np.zeros_like(self.b_v.data))
                    self.b_v.grad.data += np.sum(grad_V, axis=(0, 1))
            
            # Gradient w.r.t. inputs
            grad_query = grad_Q @ self.W_q.data.T if query.requires_grad else None
            grad_key = grad_K @ self.W_k.data.T if key.requires_grad else None
            grad_value = grad_V @ self.W_v.data.T if value.requires_grad else None
            
            return (grad_query, grad_key, grad_value, None, None, None, None)
        
        output._backward_fn = _backward
        
        return output, attention_weights
    
    def parameters(self) -> List[Tensor]:
        """Get trainable parameters."""
        params = [self.W_q, self.W_k, self.W_v, self.W_o]
        if self.use_bias:
            params.extend([self.b_q, self.b_k, self.b_v, self.b_o])
        return params
    
    def zero_grad(self):
        """Reset gradients."""
        self.W_q.grad = None
        self.W_k.grad = None
        self.W_v.grad = None
        self.W_o.grad = None
        if self.use_bias:
            self.b_q.grad = None
            self.b_k.grad = None
            self.b_v.grad = None
            self.b_o.grad = None
    
    def __repr__(self) -> str:
        return f"MultiHeadAttention(d_model={self.d_model}, num_heads={self.num_heads}, dropout={self.dropout})"


def create_causal_mask(seq_len: int) -> Tensor:
    """
    Create a causal (autoregressive) attention mask.
    
    The mask prevents attending to future positions.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Mask tensor of shape (1, seq_len, seq_len) with -inf for future positions
    """
    # Create mask: 0 for positions to attend, -inf for positions to mask
    mask = np.zeros((seq_len, seq_len), dtype=np.float32)
    # Set upper triangle (future positions) to -inf
    mask[np.triu_indices(seq_len, k=1)] = -1e9  # Use large negative instead of -inf
    return Tensor(mask.reshape(1, seq_len, seq_len))


def create_padding_mask(lengths: np.ndarray, max_len: int) -> Tensor:
    """
    Create a padding mask for variable-length sequences.
    
    Args:
        lengths: Array of sequence lengths, shape (batch,)
        max_len: Maximum sequence length
        
    Returns:
        Mask tensor of shape (batch, 1, max_len) with -inf for padding positions
    """
    batch_size = len(lengths)
    mask = np.zeros((batch_size, 1, max_len), dtype=np.float32)
    
    for i, length in enumerate(lengths):
        if length < max_len:
            mask[i, 0, length:] = -np.inf
    
    return Tensor(mask)
