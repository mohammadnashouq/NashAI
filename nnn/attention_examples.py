"""
Examples demonstrating attention mechanisms.

This file shows how to use:
- scaled_dot_product_attention for basic attention
- MultiHeadAttention for transformer-style attention
- Masking for causal and padding scenarios
"""

import numpy as np
from nnn import Tensor, Dense
from nnn.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    create_causal_mask,
    create_padding_mask
)
from nnn.activations import softmax
from nnn.optim import Adam
from nnn.losses import mse_loss


def example_scaled_dot_product_attention():
    """Example: Basic scaled dot-product attention."""
    print("=" * 60)
    print("Scaled Dot-Product Attention")
    print("=" * 60)
    
    # Create Q, K, V tensors
    # Shape: (batch, seq_len, d_k)
    batch_size = 2
    seq_len = 4
    d_k = 8
    
    np.random.seed(42)
    Q = Tensor(np.random.randn(batch_size, seq_len, d_k).astype(np.float32))
    K = Tensor(np.random.randn(batch_size, seq_len, d_k).astype(np.float32))
    V = Tensor(np.random.randn(batch_size, seq_len, d_k).astype(np.float32))
    
    print(f"Query shape: {Q.shape}")
    print(f"Key shape: {K.shape}")
    print(f"Value shape: {V.shape}")
    
    # Apply attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    print("\nAttention weights for first batch (rows sum to 1):")
    print(attention_weights.data[0])
    print(f"Row sums: {attention_weights.data[0].sum(axis=1)}")
    
    print("\nFormula: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V")
    print(f"Scaling factor: 1/sqrt({d_k}) = {1/np.sqrt(d_k):.4f}")
    print()


def example_attention_interpretation():
    """Example: Interpreting attention weights."""
    print("=" * 60)
    print("Interpreting Attention Weights")
    print("=" * 60)
    
    # Simple example: 4 tokens attending to each other
    # Token embeddings (simple one-hot style)
    tokens = ["The", "cat", "sat", "down"]
    seq_len = 4
    d_k = 4
    
    # Create Q, K, V (identity for simplicity)
    # Make "cat" and "sat" similar
    embeddings = np.array([
        [1, 0, 0, 0],  # The
        [0, 1, 0.8, 0],  # cat (similar to sat)
        [0, 0.8, 1, 0],  # sat (similar to cat)
        [0, 0, 0, 1],  # down
    ], dtype=np.float32).reshape(1, seq_len, d_k)
    
    Q = Tensor(embeddings)
    K = Tensor(embeddings)
    V = Tensor(embeddings)
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print("Tokens:", tokens)
    print("\nAttention matrix (who attends to whom):")
    print("         ", "  ".join(f"{t:>5}" for t in tokens))
    for i, token in enumerate(tokens):
        weights = attention_weights.data[0, i]
        print(f"{token:>8}:", "  ".join(f"{w:5.2f}" for w in weights))
    
    print("\nObservation: 'cat' and 'sat' attend strongly to each other")
    print("due to their similar embeddings.")
    print()


def example_causal_mask():
    """Example: Causal (autoregressive) attention mask."""
    print("=" * 60)
    print("Causal Attention Mask")
    print("=" * 60)
    
    seq_len = 5
    
    # Create causal mask
    mask = create_causal_mask(seq_len)
    
    print(f"Causal mask shape: {mask.shape}")
    print("\nMask (0 = attend, -inf = don't attend):")
    print(mask.data[0])
    
    # Apply to attention
    Q = Tensor(np.random.randn(1, seq_len, 8).astype(np.float32))
    K = Tensor(np.random.randn(1, seq_len, 8).astype(np.float32))
    V = Tensor(np.random.randn(1, seq_len, 8).astype(np.float32))
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    print("\nAttention weights with causal mask:")
    print("(Each position can only attend to previous positions)")
    np.set_printoptions(precision=2, suppress=True)
    print(attention_weights.data[0])
    np.set_printoptions()
    
    print("\nNote: Upper triangle is 0 (can't attend to future)")
    print()


def example_padding_mask():
    """Example: Padding mask for variable-length sequences."""
    print("=" * 60)
    print("Padding Mask for Variable-Length Sequences")
    print("=" * 60)
    
    # Batch of 3 sequences with different lengths
    # Padded to max_len=5
    lengths = np.array([3, 5, 2])  # Actual lengths
    max_len = 5
    
    # Create padding mask
    mask = create_padding_mask(lengths, max_len)
    
    print(f"Sequence lengths: {lengths}")
    print(f"Max length: {max_len}")
    print(f"\nPadding mask shape: {mask.shape}")
    print("Mask (0 = valid, -inf = padding):")
    for i, length in enumerate(lengths):
        print(f"  Seq {i} (len={length}): {mask.data[i, 0]}")
    
    # Apply to attention
    batch_size = 3
    d_k = 8
    
    Q = Tensor(np.random.randn(batch_size, max_len, d_k).astype(np.float32))
    K = Tensor(np.random.randn(batch_size, max_len, d_k).astype(np.float32))
    V = Tensor(np.random.randn(batch_size, max_len, d_k).astype(np.float32))
    
    # Expand mask for attention: (batch, 1, max_len) -> (batch, max_len, max_len)
    mask_expanded = Tensor(np.tile(mask.data, (1, max_len, 1)))
    
    output, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask_expanded)
    
    print("\nAttention weights for sequence 0 (length 3):")
    print("(positions 3,4 are padding - zero attention)")
    np.set_printoptions(precision=2, suppress=True)
    print(attention_weights.data[0])
    np.set_printoptions()
    print()


def example_multihead_attention():
    """Example: Multi-Head Attention."""
    print("=" * 60)
    print("Multi-Head Attention")
    print("=" * 60)
    
    # Parameters
    d_model = 64  # Model dimension
    num_heads = 4  # Number of attention heads
    d_k = d_model // num_heads  # Dimension per head
    
    # Create Multi-Head Attention layer
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    print(f"Layer: {mha}")
    print(f"d_model: {d_model}")
    print(f"num_heads: {num_heads}")
    print(f"d_k per head: {d_k}")
    
    # Input tensors
    batch_size = 2
    seq_len = 6
    
    Q = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32), 
               requires_grad=True)
    K = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32),
               requires_grad=True)
    V = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32),
               requires_grad=True)
    
    print(f"\nInput shapes:")
    print(f"  Query: {Q.shape}")
    print(f"  Key: {K.shape}")
    print(f"  Value: {V.shape}")
    
    # Forward pass
    output, attention_weights = mha(Q, K, V)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"  (batch, num_heads, seq_q, seq_k)")
    print()


def example_self_attention():
    """Example: Self-Attention (Q=K=V)."""
    print("=" * 60)
    print("Self-Attention")
    print("=" * 60)
    
    # In self-attention, Q, K, V all come from the same input
    d_model = 32
    num_heads = 4
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # Single input for Q, K, V
    batch_size = 2
    seq_len = 5
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32),
               requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    print("Self-attention: Q = K = V = input")
    
    # Self-attention
    output, attention_weights = mha(x, x, x)
    
    print(f"Output shape: {output.shape}")
    print(f"\nAttention pattern for batch 0, head 0:")
    np.set_printoptions(precision=2, suppress=True)
    print(attention_weights.data[0, 0])
    np.set_printoptions()
    print()


def example_cross_attention():
    """Example: Cross-Attention (encoder-decoder attention)."""
    print("=" * 60)
    print("Cross-Attention (Encoder-Decoder)")
    print("=" * 60)
    
    # In cross-attention, Q comes from decoder, K and V from encoder
    d_model = 32
    num_heads = 4
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    batch_size = 2
    encoder_seq_len = 8  # Source sequence length
    decoder_seq_len = 5  # Target sequence length
    
    # Encoder output (used as K and V)
    encoder_output = Tensor(
        np.random.randn(batch_size, encoder_seq_len, d_model).astype(np.float32),
        requires_grad=True
    )
    
    # Decoder state (used as Q)
    decoder_state = Tensor(
        np.random.randn(batch_size, decoder_seq_len, d_model).astype(np.float32),
        requires_grad=True
    )
    
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Decoder state shape: {decoder_state.shape}")
    print("\nCross-attention: Q from decoder, K and V from encoder")
    
    # Cross-attention
    output, attention_weights = mha(decoder_state, encoder_output, encoder_output)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"  Each decoder position attends to all encoder positions")
    print()


def example_causal_multihead_attention():
    """Example: Causal Multi-Head Attention for language models."""
    print("=" * 60)
    print("Causal Multi-Head Attention")
    print("=" * 60)
    
    d_model = 32
    num_heads = 4
    seq_len = 6
    batch_size = 2
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # Create causal mask
    mask = create_causal_mask(seq_len)
    
    # Input
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))
    
    print(f"Input shape: {x.shape}")
    print("Using causal mask for autoregressive attention")
    
    # Expand mask for MHA: (1, seq, seq) -> will be broadcast
    output, attention_weights = mha(x, x, x, mask=mask)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"\nCausal attention pattern (batch 0, head 0):")
    np.set_printoptions(precision=2, suppress=True)
    print(attention_weights.data[0, 0])
    np.set_printoptions()
    print("(Upper triangle is ~0, can't attend to future)")
    print()


def example_attention_with_dropout():
    """Example: Attention with dropout during training."""
    print("=" * 60)
    print("Attention with Dropout")
    print("=" * 60)
    
    d_model = 32
    num_heads = 4
    dropout_rate = 0.3
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout_rate)
    
    print(f"Layer: {mha}")
    
    x = Tensor(np.random.randn(2, 5, d_model).astype(np.float32))
    
    # Training mode
    mha.train()
    output_train, _ = mha(x, x, x)
    
    # Eval mode
    mha.eval()
    output_eval, _ = mha(x, x, x)
    
    print("\nTraining mode: Dropout applied to attention weights")
    print("Eval mode: No dropout")
    
    # Compare outputs (they will differ due to dropout)
    diff = np.abs(output_train.data - output_eval.data).mean()
    print(f"\nMean difference between train/eval outputs: {diff:.4f}")
    print("(Difference is expected due to dropout)")
    print()


def example_attention_backward():
    """Example: Attention backward pass."""
    print("=" * 60)
    print("Attention Backward Pass")
    print("=" * 60)
    
    d_model = 16
    num_heads = 2
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # Input with gradients
    x = Tensor(np.random.randn(2, 4, d_model).astype(np.float32), requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    
    # Forward
    output, _ = mha(x, x, x)
    
    # Compute loss (sum of outputs)
    loss = output.sum()
    
    print(f"Output shape: {output.shape}")
    print(f"Loss (sum): {loss.data.item():.4f}")
    
    # Backward
    loss.backward()
    
    print("\nGradients computed:")
    print(f"  W_q gradient shape: {mha.W_q.grad.data.shape}")
    print(f"  W_k gradient shape: {mha.W_k.grad.data.shape}")
    print(f"  W_v gradient shape: {mha.W_v.grad.data.shape}")
    print(f"  W_o gradient shape: {mha.W_o.grad.data.shape}")
    print()


def example_transformer_layer():
    """Example: Simple transformer layer (attention + feedforward)."""
    print("=" * 60)
    print("Simple Transformer Layer")
    print("=" * 60)
    
    # Transformer layer: Self-Attention + FFN
    d_model = 64
    num_heads = 4
    d_ff = 256  # Feedforward dimension
    
    # Components
    self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    ffn1 = Dense(d_model, d_ff)
    ffn2 = Dense(d_ff, d_model)
    
    print("Transformer Layer Architecture:")
    print(f"  1. Self-Attention: d_model={d_model}, heads={num_heads}")
    print(f"  2. FFN: {d_model} -> {d_ff} -> {d_model}")
    print()
    
    # Input
    batch_size = 2
    seq_len = 8
    x = Tensor(np.random.randn(batch_size, seq_len, d_model).astype(np.float32),
               requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    
    # Self-attention (with residual)
    attn_out, _ = self_attention(x, x, x)
    x_after_attn = Tensor(x.data + attn_out.data)  # Residual connection
    print(f"After self-attention: {x_after_attn.shape}")
    
    # FFN (with residual)
    # Reshape for dense: (batch, seq, d_model) -> (batch * seq, d_model)
    x_flat = Tensor(x_after_attn.data.reshape(-1, d_model))
    ffn_out = ffn2(Tensor(np.maximum(0, ffn1(x_flat).data)))  # ReLU activation
    ffn_out = Tensor(ffn_out.data.reshape(batch_size, seq_len, d_model))
    output = Tensor(x_after_attn.data + ffn_out.data)  # Residual connection
    
    print(f"After FFN: {output.shape}")
    print()


def example_attention_training():
    """Example: Training with attention."""
    print("=" * 60)
    print("Training with Attention")
    print("=" * 60)
    
    # Simple task: Learn to copy input
    np.random.seed(42)
    
    d_model = 16
    num_heads = 2
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output_proj = Dense(d_model, d_model)
    
    # Optimizer
    params = mha.parameters() + output_proj.parameters()
    optimizer = Adam(params, lr=0.01)
    
    print("Task: Learn identity mapping through attention")
    print(f"Model: MultiHeadAttention({d_model}, {num_heads}) + Dense")
    print()
    
    # Training loop
    for epoch in range(5):
        total_loss = 0
        
        for _ in range(10):
            # Generate batch
            batch_size = 4
            seq_len = 5
            x_data = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            
            x = Tensor(x_data, requires_grad=True)
            target = Tensor(x_data)  # Target = input (identity)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward
            attn_out, _ = mha(x, x, x)
            # Flatten for dense
            attn_flat = Tensor(attn_out.data.reshape(-1, d_model))
            pred_flat = output_proj(attn_flat)
            pred = Tensor(pred_flat.data.reshape(batch_size, seq_len, d_model))
            
            # Loss
            loss = mse_loss(pred, target)
            total_loss += loss.data.item()
            
            # Backward
            loss.backward()
            
            # Update
            optimizer.step()
        
        avg_loss = total_loss / 10
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
    
    print()


if __name__ == "__main__":
    # Run all examples
    example_scaled_dot_product_attention()
    example_attention_interpretation()
    example_causal_mask()
    example_padding_mask()
    example_multihead_attention()
    example_self_attention()
    example_cross_attention()
    example_causal_multihead_attention()
    example_attention_with_dropout()
    example_attention_backward()
    example_transformer_layer()
    example_attention_training()
    
    print("=" * 60)
    print("All Attention examples completed!")
    print("=" * 60)
