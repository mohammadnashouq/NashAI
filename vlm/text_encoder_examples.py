"""
Text Encoder Examples.

Demonstrates usage of:
- Transformer-based text encoder
- Text embeddings
- Attention mechanisms in text
- Configuration options
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnn.tensor import Tensor
from vlm.text_encoder import (
    TextTransformerBlock,
    TextEncoder,
    TextEncoderConfig,
    create_text_encoder,
)


def example_text_transformer_block():
    """Example 1: Single Transformer Block."""
    print("=" * 60)
    print("Example 1: Text Transformer Block")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 16
    embed_dim = 64
    
    # Create input
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32))
    print(f"Input shape: {x.shape}")
    
    # Create transformer block
    block = TextTransformerBlock(
        embed_dim=embed_dim,
        num_heads=4,
        mlp_ratio=4.0,
        max_seq_len=32,
    )
    
    # Forward pass
    out = block(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.data.size for p in block.parameters()):,}")
    print()


def example_text_encoder_basic():
    """Example 2: Basic Text Encoder Usage."""
    print("=" * 60)
    print("Example 2: Basic Text Encoder")
    print("=" * 60)
    
    vocab_size = 1000
    max_seq_len = 32
    embed_dim = 64
    
    # Create encoder
    encoder = TextEncoder(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=4,
    )
    
    # Create input token IDs
    batch_size = 2
    input_ids = np.random.randint(0, vocab_size, (batch_size, 16))
    print(f"Input token IDs shape: {input_ids.shape}")
    
    # Forward pass
    out = encoder(input_ids)
    print(f"Output embedding shape: {out.shape}")
    print()


def example_attention_mask():
    """Example 3: Using Attention Mask."""
    print("=" * 60)
    print("Example 3: Attention Mask for Padding")
    print("=" * 60)
    
    vocab_size = 100
    
    # Create encoder
    encoder = TextEncoder(
        vocab_size=vocab_size,
        max_seq_len=32,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
    )
    
    # Create input with different lengths (padded with 0)
    # Sequence 1: 10 real tokens + 6 padding
    # Sequence 2: 8 real tokens + 8 padding
    input_ids = np.array([
        [5, 12, 8, 3, 7, 15, 2, 9, 11, 4, 0, 0, 0, 0, 0, 0],  # 10 tokens
        [6, 14, 1, 8, 3, 12, 7, 5, 0, 0, 0, 0, 0, 0, 0, 0],   # 8 tokens
    ], dtype=np.int64)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Forward pass with mask
    out = encoder(input_ids, attention_mask=attention_mask)
    print(f"Output shape: {out.shape}")
    
    # The encoder pools at the last non-padded position
    print("Pooling at EOS (last valid token) position")
    print()


def example_normalized_embeddings():
    """Example 4: L2 Normalized Embeddings."""
    print("=" * 60)
    print("Example 4: L2 Normalized Embeddings")
    print("=" * 60)
    
    encoder = TextEncoder(
        vocab_size=100,
        max_seq_len=32,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
    )
    
    input_ids = np.random.randint(0, 100, (3, 12))
    
    # Get normalized embeddings
    emb = encoder.encode_text(input_ids, normalize=True)
    print(f"Embedding shape: {emb.shape}")
    
    # Check norms
    norms = np.sqrt(np.sum(emb.data ** 2, axis=-1))
    print(f"Embedding norms (should be ~1): {norms}")
    
    # Without normalization
    emb_raw = encoder.encode_text(input_ids, normalize=False)
    norms_raw = np.sqrt(np.sum(emb_raw.data ** 2, axis=-1))
    print(f"Raw embedding norms: {norms_raw}")
    print()


def example_text_projection():
    """Example 5: Text Projection to Different Dimension."""
    print("=" * 60)
    print("Example 5: Text Projection Layer")
    print("=" * 60)
    
    embed_dim = 64
    projection_dim = 128
    
    encoder = TextEncoder(
        vocab_size=100,
        max_seq_len=32,
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=4,
    )
    
    print(f"Original embed dim: {embed_dim}")
    
    # Set projection
    encoder.set_projection(projection_dim)
    print(f"Projection dim: {projection_dim}")
    
    # Forward pass
    input_ids = np.random.randint(0, 100, (2, 16))
    out = encoder(input_ids)
    print(f"Output shape after projection: {out.shape}")
    print()


def example_config_presets():
    """Example 6: Configuration Presets."""
    print("=" * 60)
    print("Example 6: Configuration Presets")
    print("=" * 60)
    
    # Tiny config for testing
    tiny = TextEncoderConfig.tiny(vocab_size=500)
    print(f"Tiny: vocab={tiny.vocab_size}, dim={tiny.embed_dim}, "
          f"layers={tiny.num_layers}, heads={tiny.num_heads}")
    
    # Small config
    small = TextEncoderConfig.small(vocab_size=5000)
    print(f"Small: vocab={small.vocab_size}, dim={small.embed_dim}, "
          f"layers={small.num_layers}, heads={small.num_heads}")
    
    # CLIP base
    clip_base = TextEncoderConfig.clip_base()
    print(f"CLIP-Base: vocab={clip_base.vocab_size}, dim={clip_base.embed_dim}, "
          f"layers={clip_base.num_layers}, heads={clip_base.num_heads}")
    
    # Create model from config
    encoder = create_text_encoder(tiny)
    input_ids = np.random.randint(0, tiny.vocab_size, (1, 16))
    out = encoder(input_ids)
    print(f"\nTiny encoder output: {out.shape}")
    print()


def example_train_eval_modes():
    """Example 7: Training vs Evaluation Modes."""
    print("=" * 60)
    print("Example 7: Training vs Evaluation Modes")
    print("=" * 60)
    
    encoder = TextEncoder(
        vocab_size=100,
        max_seq_len=32,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
    )
    
    input_ids = np.random.randint(0, 100, (2, 16))
    
    # Training mode
    encoder.train()
    print(f"Training mode: {encoder.training}")
    out_train = encoder(input_ids)
    
    # Evaluation mode
    encoder.eval()
    print(f"Eval mode: {not encoder.training}")
    out_eval = encoder(input_ids)
    
    print(f"Output shape: {out_train.shape}")
    print()


def example_parameter_count():
    """Example 8: Parameter Count Analysis."""
    print("=" * 60)
    print("Example 8: Parameter Count Analysis")
    print("=" * 60)
    
    configs = [
        ("Tiny", TextEncoderConfig.tiny()),
        ("Small", TextEncoderConfig.small()),
    ]
    
    for name, config in configs:
        encoder = create_text_encoder(config)
        
        total_params = sum(p.data.size for p in encoder.parameters())
        
        # Breakdown
        token_emb = encoder.token_embedding.data.size
        pos_emb = encoder.position_embedding.data.size
        
        print(f"\n{name} Encoder:")
        print(f"  Token embedding: {token_emb:,}")
        print(f"  Position embedding: {pos_emb:,}")
        print(f"  Total parameters: {total_params:,}")
    print()


def example_batch_processing():
    """Example 9: Batch Processing."""
    print("=" * 60)
    print("Example 9: Batch Processing")
    print("=" * 60)
    
    config = TextEncoderConfig.tiny()
    encoder = create_text_encoder(config)
    
    for batch_size in [1, 4, 8]:
        input_ids = np.random.randint(0, config.vocab_size, (batch_size, 20))
        out = encoder.encode_text(input_ids, normalize=True)
        print(f"Batch {batch_size}: input {input_ids.shape} -> output {out.shape}")
    print()


def example_similarity_computation():
    """Example 10: Text Similarity."""
    print("=" * 60)
    print("Example 10: Text Similarity Computation")
    print("=" * 60)
    
    encoder = TextEncoder(
        vocab_size=100,
        max_seq_len=32,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
    )
    
    # Create two batches of "text"
    text_a = np.random.randint(0, 100, (3, 16))
    text_b = np.random.randint(0, 100, (3, 16))
    
    # Get normalized embeddings
    emb_a = encoder.encode_text(text_a, normalize=True)
    emb_b = encoder.encode_text(text_b, normalize=True)
    
    # Compute cosine similarity (dot product of normalized vectors)
    similarity = emb_a.data @ emb_b.data.T
    
    print("Similarity matrix (3x3):")
    print(similarity)
    print("\nDiagonal (self-similarity with different random init):")
    print(np.diag(similarity))
    print()


def example_causal_attention():
    """Example 11: Understanding Causal Attention."""
    print("=" * 60)
    print("Example 11: Causal Attention Masking")
    print("=" * 60)
    
    # The text encoder uses causal (autoregressive) attention
    # This means each position can only attend to itself and previous positions
    
    block = TextTransformerBlock(
        embed_dim=64,
        num_heads=4,
        max_seq_len=8,
    )
    
    # Visualize the causal mask
    print("Causal mask (8x8):")
    print("0 = can attend, -inf = cannot attend")
    mask = block.causal_mask[:8, :8]
    mask_display = np.where(mask < -1e8, "X", ".")
    for i, row in enumerate(mask_display):
        print(f"  Position {i}: {' '.join(row)}")
    
    print("\nThis ensures autoregressive property:")
    print("- Position 0 sees only position 0")
    print("- Position 1 sees positions 0, 1")
    print("- etc.")
    print()


def example_embedding_visualization():
    """Example 12: Embedding Space Visualization."""
    print("=" * 60)
    print("Example 12: Embedding Space Analysis")
    print("=" * 60)
    
    encoder = TextEncoder(
        vocab_size=100,
        max_seq_len=32,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
    )
    
    # Generate multiple "sentences"
    num_samples = 10
    input_ids = np.random.randint(0, 100, (num_samples, 16))
    
    # Get embeddings
    embeddings = encoder.encode_text(input_ids, normalize=True)
    
    # Compute pairwise similarities
    similarity_matrix = embeddings.data @ embeddings.data.T
    
    print(f"Generated {num_samples} text embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"\nSimilarity statistics:")
    print(f"  Min: {np.min(similarity_matrix):.4f}")
    print(f"  Max: {np.max(similarity_matrix):.4f}")
    print(f"  Mean (off-diagonal): {np.mean(similarity_matrix - np.eye(num_samples)):.4f}")
    
    # Average cosine similarity (excluding diagonal)
    mask = 1 - np.eye(num_samples)
    avg_sim = np.sum(similarity_matrix * mask) / np.sum(mask)
    print(f"  Average pairwise similarity: {avg_sim:.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Text Encoder Examples")
    print("=" * 60 + "\n")
    
    example_text_transformer_block()
    example_text_encoder_basic()
    example_attention_mask()
    example_normalized_embeddings()
    example_text_projection()
    example_config_presets()
    example_train_eval_modes()
    example_parameter_count()
    example_batch_processing()
    example_similarity_computation()
    example_causal_attention()
    example_embedding_visualization()
    
    print("=" * 60)
    print("All text encoder examples completed!")
    print("=" * 60)
