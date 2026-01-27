"""
Text Encoder for VLMs.

Implements:
- Transformer-based text encoder
- Text embeddings for multimodal learning
- Compatible with CLIP-style training
"""

import numpy as np
from typing import Optional, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnn.tensor import Tensor


# =============================================================================
# Text Encoder Components
# =============================================================================

class TextTransformerBlock:
    """
    Transformer block for text encoding.
    
    Uses causal (autoregressive) attention for text.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 77,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mlp_dim = int(embed_dim * mlp_ratio)
        self.dropout = dropout
        self.training = True
        
        # Layer norms
        self.ln1 = self._create_layer_norm(embed_dim)
        self.ln2 = self._create_layer_norm(embed_dim)
        
        # Multi-head causal self-attention
        limit = np.sqrt(6.0 / (embed_dim + embed_dim))
        self.W_qkv = Tensor(
            np.random.uniform(-limit, limit, (embed_dim, 3 * embed_dim)).astype(np.float32),
            requires_grad=True
        )
        self.W_o = Tensor(
            np.random.uniform(-limit, limit, (embed_dim, embed_dim)).astype(np.float32),
            requires_grad=True
        )
        
        # Causal mask
        self.causal_mask = self._create_causal_mask(max_seq_len)
        
        # MLP
        limit1 = np.sqrt(6.0 / (embed_dim + self.mlp_dim))
        limit2 = np.sqrt(6.0 / (self.mlp_dim + embed_dim))
        self.mlp_fc1 = Tensor(
            np.random.uniform(-limit1, limit1, (embed_dim, self.mlp_dim)).astype(np.float32),
            requires_grad=True
        )
        self.mlp_fc2 = Tensor(
            np.random.uniform(-limit2, limit2, (self.mlp_dim, embed_dim)).astype(np.float32),
            requires_grad=True
        )
    
    def _create_layer_norm(self, dim: int):
        """Create layer normalization parameters."""
        return {
            'gamma': Tensor(np.ones(dim, dtype=np.float32), requires_grad=True),
            'beta': Tensor(np.zeros(dim, dtype=np.float32), requires_grad=True),
        }
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal attention mask."""
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
        mask = mask * -1e9
        return mask
    
    def _layer_norm(self, x: np.ndarray, ln: dict, eps: float = 1e-5) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return ln['gamma'].data * x_norm + ln['beta'].data
    
    def __call__(self, x: Tensor, attention_mask: Optional[np.ndarray] = None) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Optional padding mask
        
        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Self-attention with causal masking
        x_norm = self._layer_norm(x.data, self.ln1)
        attn_out = self._causal_attention(x_norm, seq_len, attention_mask)
        x_data = x.data + attn_out  # Residual
        
        # MLP
        x_norm = self._layer_norm(x_data, self.ln2)
        mlp_out = self._mlp(x_norm)
        out_data = x_data + mlp_out  # Residual
        
        return Tensor(out_data, requires_grad=x.requires_grad)
    
    def _causal_attention(
        self, x: np.ndarray, seq_len: int, attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Multi-head causal self-attention."""
        batch_size = x.shape[0]
        
        # Compute Q, K, V
        qkv = x @ self.W_qkv.data  # (B, S, 3*D)
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores + causal_mask.reshape(1, 1, seq_len, seq_len)
        
        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: (B, S) -> (B, 1, 1, S)
            padding_mask = (1 - attention_mask.reshape(batch_size, 1, 1, seq_len)) * -1e9
            scores = scores + padding_mask
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
        
        # Apply to values
        context = attn_weights @ v  # (B, H, S, D/H)
        
        # Reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        out = context @ self.W_o.data
        
        return out
    
    def _mlp(self, x: np.ndarray) -> np.ndarray:
        """MLP with GELU activation."""
        hidden = x @ self.mlp_fc1.data
        
        # GELU activation
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        hidden = 0.5 * hidden * (1 + np.tanh(sqrt_2_pi * (hidden + 0.044715 * hidden ** 3)))
        
        out = hidden @ self.mlp_fc2.data
        return out
    
    def parameters(self) -> List[Tensor]:
        params = [
            self.ln1['gamma'], self.ln1['beta'],
            self.W_qkv, self.W_o,
            self.ln2['gamma'], self.ln2['beta'],
            self.mlp_fc1, self.mlp_fc2,
        ]
        return params
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class TextEncoder:
    """
    Transformer-based text encoder for CLIP-style training.
    
    Encodes text sequences into fixed-dimensional embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int = 49408,
        max_seq_len: int = 77,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            max_seq_len: Maximum sequence length
            embed_dim: Embedding dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.training = True
        
        # Token embedding
        self.token_embedding = Tensor(
            np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02,
            requires_grad=True
        )
        
        # Position embedding
        self.position_embedding = Tensor(
            np.random.randn(max_seq_len, embed_dim).astype(np.float32) * 0.02,
            requires_grad=True
        )
        
        # Transformer blocks
        self.blocks = [
            TextTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, max_seq_len)
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.ln_final = {
            'gamma': Tensor(np.ones(embed_dim, dtype=np.float32), requires_grad=True),
            'beta': Tensor(np.zeros(embed_dim, dtype=np.float32), requires_grad=True),
        }
        
        # Text projection (optional, for matching vision embedding dim)
        self.text_projection = None
    
    def set_projection(self, projection_dim: int):
        """Set up projection layer to match vision embedding dimension."""
        limit = np.sqrt(6.0 / (self.embed_dim + projection_dim))
        self.text_projection = Tensor(
            np.random.uniform(-limit, limit, (self.embed_dim, projection_dim)).astype(np.float32),
            requires_grad=True
        )
    
    def __call__(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        """
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len), 1 for valid, 0 for pad
        
        Returns:
            Text embeddings of shape (batch, embed_dim) or (batch, projection_dim)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        tok_emb = self.token_embedding.data[input_ids]  # (B, S, D)
        
        # Position embeddings
        positions = np.arange(seq_len)
        pos_emb = self.position_embedding.data[positions]  # (S, D)
        
        # Combine
        x_data = tok_emb + pos_emb
        x = Tensor(x_data, requires_grad=True)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        x_norm = (x.data - mean) / np.sqrt(var + 1e-5)
        x_data = self.ln_final['gamma'].data * x_norm + self.ln_final['beta'].data
        
        # Pool at EOS token position (last non-padded token)
        if attention_mask is not None:
            # Find the last valid position for each sequence
            eos_positions = np.sum(attention_mask, axis=1).astype(int) - 1
            eos_positions = np.clip(eos_positions, 0, seq_len - 1)
            embeddings = x_data[np.arange(batch_size), eos_positions, :]
        else:
            # Use last position
            embeddings = x_data[:, -1, :]
        
        # Apply projection if set
        if self.text_projection is not None:
            embeddings = embeddings @ self.text_projection.data
        
        return Tensor(embeddings, requires_grad=True)
    
    def encode_text(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        normalize: bool = True,
    ) -> Tensor:
        """
        Encode text with optional L2 normalization.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            normalize: Whether to L2 normalize embeddings
        
        Returns:
            Text embeddings
        """
        embeddings = self(input_ids, attention_mask)
        
        if normalize:
            # L2 normalize
            norm = np.sqrt(np.sum(embeddings.data ** 2, axis=-1, keepdims=True) + 1e-8)
            embeddings = Tensor(embeddings.data / norm, requires_grad=embeddings.requires_grad)
        
        return embeddings
    
    def parameters(self) -> List[Tensor]:
        params = [self.token_embedding, self.position_embedding]
        
        for block in self.blocks:
            params.extend(block.parameters())
        
        params.extend([self.ln_final['gamma'], self.ln_final['beta']])
        
        if self.text_projection is not None:
            params.append(self.text_projection)
        
        return params
    
    def train(self):
        self.training = True
        for block in self.blocks:
            block.train()
    
    def eval(self):
        self.training = False
        for block in self.blocks:
            block.eval()


# =============================================================================
# Configuration
# =============================================================================

class TextEncoderConfig:
    """Configuration for Text Encoder."""
    
    def __init__(
        self,
        vocab_size: int = 49408,
        max_seq_len: int = 77,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
    
    @classmethod
    def clip_base(cls) -> 'TextEncoderConfig':
        """CLIP ViT-B/32 text encoder configuration."""
        return cls(
            vocab_size=49408,
            max_seq_len=77,
            embed_dim=512,
            num_layers=12,
            num_heads=8,
        )
    
    @classmethod
    def clip_large(cls) -> 'TextEncoderConfig':
        """CLIP ViT-L/14 text encoder configuration."""
        return cls(
            vocab_size=49408,
            max_seq_len=77,
            embed_dim=768,
            num_layers=12,
            num_heads=12,
        )
    
    @classmethod
    def tiny(cls, vocab_size: int = 1000) -> 'TextEncoderConfig':
        """Tiny configuration for testing."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=32,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
        )
    
    @classmethod
    def small(cls, vocab_size: int = 10000) -> 'TextEncoderConfig':
        """Small configuration for experiments."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=64,
            embed_dim=256,
            num_layers=6,
            num_heads=8,
        )


def create_text_encoder(config: TextEncoderConfig) -> TextEncoder:
    """Create text encoder from configuration."""
    return TextEncoder(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
    )
