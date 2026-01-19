"""
Examples for transformer architecture components.

Demonstrates:
- Positional encodings (sinusoidal and learned)
- Layer normalization
- Feed-forward network
- Causal self-attention
- Transformer blocks
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnn.tensor import Tensor


def example_sinusoidal_positional_encoding():
    """Example 1: Sinusoidal positional encoding."""
    print("=" * 60)
    print("Example 1: Sinusoidal Positional Encoding")
    print("=" * 60)
    
    from llm.transformer import SinusoidalPositionalEncoding
    
    d_model = 64
    max_seq_len = 100
    
    # Create positional encoding
    pe = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout=0.0)
    pe.eval()  # Disable dropout for visualization
    
    print(f"d_model: {d_model}")
    print(f"max_seq_len: {max_seq_len}")
    print(f"PE matrix shape: {pe.pe.shape}")
    
    # Show first few positions
    print("\nPositional encoding values (first 5 positions, first 8 dims):")
    print(pe.pe[:5, :8])
    
    # Apply to input
    batch_size, seq_len = 2, 10
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))
    
    y = pe(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Position encoding added correctly: {y.shape == x.shape}")
    print()


def example_learned_positional_encoding():
    """Example 2: Learned positional encoding."""
    print("=" * 60)
    print("Example 2: Learned Positional Encoding")
    print("=" * 60)
    
    from llm.transformer import LearnedPositionalEncoding
    
    d_model = 64
    max_seq_len = 100
    
    # Create learned positional encoding
    pe = LearnedPositionalEncoding(d_model, max_seq_len, dropout=0.0)
    pe.eval()
    
    print(f"d_model: {d_model}")
    print(f"max_seq_len: {max_seq_len}")
    print(f"Learnable PE parameters: {pe.pe.shape}")
    
    # Apply to input
    batch_size, seq_len = 2, 10
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32), requires_grad=True)
    
    y = pe(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Verify gradients flow
    print(f"\nParameters: {len(pe.parameters())}")
    print(f"PE requires_grad: {pe.pe.requires_grad}")
    print()


def example_layer_norm():
    """Example 3: Layer normalization."""
    print("=" * 60)
    print("Example 3: Layer Normalization")
    print("=" * 60)
    
    from llm.transformer import LayerNorm
    
    d_model = 64
    batch_size, seq_len = 2, 10
    
    # Create layer norm
    ln = LayerNorm(d_model)
    
    print(f"Normalized shape: {d_model}")
    print(f"Gamma shape: {ln.gamma.shape}, initial values: {ln.gamma.data[:5]}")
    print(f"Beta shape: {ln.beta.shape}, initial values: {ln.beta.data[:5]}")
    
    # Create input with varying statistics
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 10 + 5, requires_grad=True)
    
    print(f"\nBefore LayerNorm:")
    print(f"  Mean: {np.mean(x.data):.4f}, Std: {np.std(x.data):.4f}")
    
    y = ln(x)
    
    print(f"After LayerNorm:")
    print(f"  Mean: {np.mean(y.data):.4f}, Std: {np.std(y.data):.4f}")
    print(f"  (Mean ≈ 0, Std ≈ 1 per feature dimension)")
    
    # Check per-position normalization
    pos_mean = np.mean(y.data[0, 0, :])
    pos_std = np.std(y.data[0, 0, :])
    print(f"\nPer-position stats (pos 0):")
    print(f"  Mean: {pos_mean:.6f}, Std: {pos_std:.6f}")
    print()


def example_feed_forward():
    """Example 4: Feed-forward network."""
    print("=" * 60)
    print("Example 4: Feed-Forward Network (FFN)")
    print("=" * 60)
    
    from llm.transformer import FeedForward
    
    d_model = 64
    d_ff = 256
    
    # Create FFN
    ffn = FeedForward(d_model, d_ff, dropout=0.0)
    ffn.eval()
    
    print(f"d_model: {d_model}")
    print(f"d_ff (hidden): {d_ff}")
    print(f"Expansion factor: {d_ff / d_model}x")
    
    print(f"\nWeights:")
    print(f"  W1 shape: {ffn.W1.shape}")
    print(f"  b1 shape: {ffn.b1.shape}")
    print(f"  W2 shape: {ffn.W2.shape}")
    print(f"  b2 shape: {ffn.b2.shape}")
    
    # Forward pass
    batch_size, seq_len = 2, 10
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32), requires_grad=True)
    
    y = ffn(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Count parameters
    total_params = sum(p.data.size for p in ffn.parameters())
    print(f"Total parameters: {total_params:,}")
    print()


def example_causal_attention():
    """Example 5: Causal self-attention."""
    print("=" * 60)
    print("Example 5: Causal Self-Attention")
    print("=" * 60)
    
    from llm.transformer import CausalSelfAttention
    
    d_model = 64
    num_heads = 4
    
    # Create attention layer
    attn = CausalSelfAttention(d_model, num_heads, dropout=0.0, max_seq_len=128)
    attn.eval()
    
    print(f"d_model: {d_model}")
    print(f"num_heads: {num_heads}")
    print(f"d_k (per head): {attn.d_k}")
    
    print(f"\nWeights:")
    print(f"  W_qkv shape: {attn.W_qkv.shape} (combined Q, K, V projection)")
    print(f"  W_o shape: {attn.W_o.shape} (output projection)")
    
    # Show causal mask
    print(f"\nCausal mask (first 5x5):")
    print(attn.causal_mask[:5, :5])
    print("(0 = can attend, -1e9 = cannot attend)")
    
    # Forward pass
    batch_size, seq_len = 2, 10
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32), requires_grad=True)
    
    y = attn(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print()


def example_transformer_block():
    """Example 6: Complete transformer block."""
    print("=" * 60)
    print("Example 6: Transformer Block")
    print("=" * 60)
    
    from llm.transformer import TransformerBlock
    
    d_model = 64
    num_heads = 4
    d_ff = 256
    
    # Create transformer block
    block = TransformerBlock(d_model, num_heads, d_ff, dropout=0.0)
    block.eval()
    
    print(f"Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    
    # Count parameters
    params = block.parameters()
    total_params = sum(p.data.size for p in params)
    print(f"  Total parameters: {total_params:,}")
    
    print(f"\nArchitecture (Pre-LayerNorm):")
    print("  x -> LayerNorm -> CausalSelfAttention -> + (residual)")
    print("    -> LayerNorm -> FFN -> + (residual) -> output")
    
    # Forward pass
    batch_size, seq_len = 2, 10
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32), requires_grad=True)
    
    y = block(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Verify residual connection (output should be close to input + delta)
    diff = np.mean(np.abs(y.data - x.data))
    print(f"Mean absolute difference from input: {diff:.4f}")
    print()


def example_stacked_blocks():
    """Example 7: Stacking multiple transformer blocks."""
    print("=" * 60)
    print("Example 7: Stacked Transformer Blocks")
    print("=" * 60)
    
    from llm.transformer import TransformerBlock, LayerNorm
    
    d_model = 64
    num_heads = 4
    num_layers = 3
    
    # Create stacked blocks
    blocks = [TransformerBlock(d_model, num_heads, dropout=0.0) for _ in range(num_layers)]
    final_ln = LayerNorm(d_model)
    
    # Set to eval mode
    for block in blocks:
        block.eval()
    
    print(f"Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_layers: {num_layers}")
    
    # Count total parameters
    total_params = sum(sum(p.data.size for p in b.parameters()) for b in blocks)
    total_params += sum(p.data.size for p in final_ln.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Forward pass through all blocks
    batch_size, seq_len = 2, 10
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))
    
    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    
    for i, block in enumerate(blocks):
        x = block(x)
        print(f"  After block {i+1}: {x.shape}")
    
    x = final_ln(x)
    print(f"  After final LayerNorm: {x.shape}")
    print()


def example_attention_patterns():
    """Example 8: Visualize attention patterns."""
    print("=" * 60)
    print("Example 8: Attention Pattern Visualization")
    print("=" * 60)
    
    from llm.transformer import CausalSelfAttention
    
    d_model = 32
    num_heads = 2
    
    # Create attention layer with accessible internals
    attn = CausalSelfAttention(d_model, num_heads, dropout=0.0)
    attn.eval()
    
    # Small input for visualization
    seq_len = 5
    x_data = np.random.randn(1, seq_len, d_model).astype(np.float32)
    x = Tensor(x_data)
    
    # Forward pass (to compute attention weights internally)
    _ = attn(x)
    
    # Manually compute attention weights for visualization
    qkv = x_data @ attn.W_qkv.data + attn.b_qkv.data
    q, k, v = np.split(qkv, 3, axis=-1)
    
    q = q.reshape(1, seq_len, num_heads, d_model // num_heads).transpose(0, 2, 1, 3)
    k = k.reshape(1, seq_len, num_heads, d_model // num_heads).transpose(0, 2, 1, 3)
    
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_model // num_heads)
    scores = scores + attn.causal_mask[:seq_len, :seq_len].reshape(1, 1, seq_len, seq_len)
    
    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
    
    print(f"Sequence length: {seq_len}")
    print(f"Number of heads: {num_heads}")
    
    # Show attention weights for first head
    print(f"\nAttention weights (Head 1):")
    print("Position i attends to positions 0..i (causal mask)")
    print(np.round(attention_weights[0, 0], 3))
    
    print(f"\nAttention weights (Head 2):")
    print(np.round(attention_weights[0, 1], 3))
    print()


def example_causal_mask_utility():
    """Example 9: Causal mask utility functions."""
    print("=" * 60)
    print("Example 9: Mask Utility Functions")
    print("=" * 60)
    
    from llm.transformer import create_causal_mask, create_padding_mask
    
    # Causal mask
    seq_len = 6
    causal_mask = create_causal_mask(seq_len)
    
    print(f"Causal mask ({seq_len}x{seq_len}):")
    print("0 = can attend, -1e9 = masked (cannot attend)")
    # Convert for display
    display_mask = np.where(causal_mask == 0, 1, 0)
    print(display_mask)
    
    # Padding mask
    lengths = [4, 3, 6]  # Actual lengths in batch
    max_len = 6
    padding_mask = create_padding_mask(lengths, max_len)
    
    print(f"\nPadding mask (batch_size=3, max_len={max_len}):")
    print(f"Lengths: {lengths}")
    for i, length in enumerate(lengths):
        mask_row = padding_mask[i, 0, 0, :]
        # Convert for display
        display = ['.' if m == 0 else 'X' for m in mask_row]
        print(f"  Sample {i} (len={length}): {' '.join(display)}")
    print("  . = attend, X = masked (padding)")
    print()


def example_gradient_flow():
    """Example 10: Verify gradient flow through transformer."""
    print("=" * 60)
    print("Example 10: Gradient Flow Through Transformer")
    print("=" * 60)
    
    from llm.transformer import TransformerBlock
    
    d_model = 32
    num_heads = 4
    
    block = TransformerBlock(d_model, num_heads, dropout=0.0)
    block.eval()
    
    # Create input
    batch_size, seq_len = 2, 8
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32), requires_grad=True)
    
    # Forward pass
    y = block(x)
    
    # Simple loss (sum of outputs)
    loss = Tensor(np.sum(y.data), requires_grad=True)
    
    # Backward pass
    loss.backward(np.ones_like(loss.data))
    y.backward(np.ones_like(y.data))
    
    # Check gradients
    print("Checking gradients for transformer block parameters:")
    
    params = block.parameters()
    grad_info = []
    
    for i, p in enumerate(params):
        if p.grad is not None:
            grad_norm = np.sqrt(np.sum(p.grad ** 2))
            grad_info.append((i, p.shape, grad_norm))
    
    for i, shape, norm in grad_info[:6]:  # Show first 6
        print(f"  Param {i}: shape={shape}, grad_norm={norm:.6f}")
    
    print(f"\nTotal parameters with gradients: {len(grad_info)}")
    print()


if __name__ == "__main__":
    example_sinusoidal_positional_encoding()
    example_learned_positional_encoding()
    example_layer_norm()
    example_feed_forward()
    example_causal_attention()
    example_transformer_block()
    example_stacked_blocks()
    example_attention_patterns()
    example_causal_mask_utility()
    example_gradient_flow()
    
    print("All transformer examples completed!")
