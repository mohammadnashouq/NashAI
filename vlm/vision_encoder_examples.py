"""
Vision Encoder Examples.

Demonstrates usage of:
- CNN-based vision encoder (ResNet-style)
- Vision Transformer (ViT)
- Patch embedding
- Image feature extraction
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnn.tensor import Tensor
from vlm.vision_encoder import (
    ConvBlock,
    ResidualBlock,
    CNNEncoder,
    PatchEmbedding,
    ViTBlock,
    VisionTransformer,
    ViTConfig,
)


def example_conv_block():
    """Example 1: Basic ConvBlock."""
    print("=" * 60)
    print("Example 1: ConvBlock (Conv -> BatchNorm -> ReLU)")
    print("=" * 60)
    
    # Create input image (batch=2, channels=3, height=32, width=32)
    x = Tensor(np.random.randn(2, 3, 32, 32).astype(np.float32), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Create ConvBlock
    block = ConvBlock(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    
    # Forward pass
    out = block(x)
    print(f"Output shape: {out.shape}")
    print(f"Number of parameters: {sum(p.data.size for p in block.parameters())}")
    print()


def example_residual_block():
    """Example 2: ResidualBlock with skip connection."""
    print("=" * 60)
    print("Example 2: ResidualBlock (with skip connection)")
    print("=" * 60)
    
    # Create input
    x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Same dimensions (identity shortcut)
    block1 = ResidualBlock(in_channels=64, out_channels=64, stride=1)
    out1 = block1(x)
    print(f"Same dim output: {out1.shape}")
    
    # Downsampling (projection shortcut)
    block2 = ResidualBlock(in_channels=64, out_channels=128, stride=2)
    out2 = block2(x)
    print(f"Downsample output: {out2.shape}")
    print()


def example_cnn_encoder():
    """Example 3: Full CNN Encoder."""
    print("=" * 60)
    print("Example 3: CNN Encoder (ResNet-style)")
    print("=" * 60)
    
    # Create encoder
    encoder = CNNEncoder(
        in_channels=3,
        base_channels=32,
        num_layers=[1, 1, 1, 1],  # Small for demo
        embed_dim=128,
    )
    
    # Create input image
    x = Tensor(np.random.randn(2, 3, 64, 64).astype(np.float32), requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    out = encoder(x)
    print(f"Output embedding shape: {out.shape}")
    
    # Get normalized embeddings
    emb = encoder.encode_image(x, normalize=True)
    print(f"Normalized embedding shape: {emb.shape}")
    
    # Check normalization
    norms = np.sqrt(np.sum(emb.data ** 2, axis=-1))
    print(f"Embedding norms (should be ~1): {norms}")
    print()


def example_patch_embedding():
    """Example 4: Patch Embedding for ViT."""
    print("=" * 60)
    print("Example 4: Patch Embedding")
    print("=" * 60)
    
    # Create input image
    batch_size = 2
    image_size = 32
    patch_size = 8
    in_channels = 3
    embed_dim = 64
    
    x = Tensor(np.random.randn(batch_size, in_channels, image_size, image_size).astype(np.float32))
    print(f"Input image shape: {x.shape}")
    
    # Create patch embedding
    patch_embed = PatchEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )
    
    # Forward pass
    patches = patch_embed(x)
    num_patches = (image_size // patch_size) ** 2
    print(f"Number of patches: {num_patches}")
    print(f"Patch embeddings shape: {patches.shape}")  # (batch, num_patches, embed_dim)
    print()


def example_vit_block():
    """Example 5: Vision Transformer Block."""
    print("=" * 60)
    print("Example 5: ViT Block (Attention + MLP)")
    print("=" * 60)
    
    batch_size = 2
    num_patches = 16  # 4x4 patches
    embed_dim = 64
    num_heads = 4
    
    # Create input (batch, num_patches + 1 cls token, embed_dim)
    x = Tensor(np.random.randn(batch_size, num_patches + 1, embed_dim).astype(np.float32))
    print(f"Input shape: {x.shape}")
    
    # Create ViT block
    block = ViTBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0)
    
    # Forward pass
    out = block(x)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.data.size for p in block.parameters())}")
    print()


def example_vision_transformer():
    """Example 6: Full Vision Transformer."""
    print("=" * 60)
    print("Example 6: Vision Transformer (ViT)")
    print("=" * 60)
    
    # Create ViT
    vit = VisionTransformer(
        image_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    
    # Create input
    x = Tensor(np.random.randn(2, 3, 32, 32).astype(np.float32))
    print(f"Input image shape: {x.shape}")
    
    # Forward pass
    out = vit(x)
    print(f"Output shape: {out.shape}")
    
    # Get normalized embeddings
    emb = vit.encode_image(x, normalize=True)
    print(f"Normalized embedding shape: {emb.shape}")
    
    # Verify normalization
    norms = np.sqrt(np.sum(emb.data ** 2, axis=-1))
    print(f"Embedding norms (should be ~1): {norms}")
    print()


def example_vit_config():
    """Example 7: Using ViT Configuration."""
    print("=" * 60)
    print("Example 7: ViT Configuration Presets")
    print("=" * 60)
    
    # Tiny config for testing
    tiny_config = ViTConfig.tiny()
    print(f"Tiny config: {tiny_config.image_size}x{tiny_config.image_size}, "
          f"patch={tiny_config.patch_size}, dim={tiny_config.embed_dim}, "
          f"layers={tiny_config.num_layers}")
    
    # Small config
    small_config = ViTConfig.small()
    print(f"Small config: {small_config.image_size}x{small_config.image_size}, "
          f"patch={small_config.patch_size}, dim={small_config.embed_dim}, "
          f"layers={small_config.num_layers}")
    
    # Create model from config
    vit = VisionTransformer(
        image_size=tiny_config.image_size,
        patch_size=tiny_config.patch_size,
        embed_dim=tiny_config.embed_dim,
        num_layers=tiny_config.num_layers,
        num_heads=tiny_config.num_heads,
    )
    
    # Test forward pass
    x = Tensor(np.random.randn(1, 3, tiny_config.image_size, tiny_config.image_size).astype(np.float32))
    out = vit(x)
    print(f"Output from tiny ViT: {out.shape}")
    print()


def example_train_eval_modes():
    """Example 8: Training vs Evaluation modes."""
    print("=" * 60)
    print("Example 8: Training vs Evaluation Modes")
    print("=" * 60)
    
    # Create CNN encoder with BatchNorm
    encoder = CNNEncoder(
        in_channels=3,
        base_channels=16,
        num_layers=[1, 1, 1, 1],  # 4 stages like ResNet
        embed_dim=64,
    )
    
    # Create input
    x = Tensor(np.random.randn(4, 3, 32, 32).astype(np.float32))
    
    # Training mode
    encoder.train()
    out_train = encoder(x)
    print(f"Training mode output shape: {out_train.shape}")
    
    # Evaluation mode
    encoder.eval()
    out_eval = encoder(x)
    print(f"Eval mode output shape: {out_eval.shape}")
    
    # Outputs may differ due to BatchNorm behavior
    diff = np.mean(np.abs(out_train.data - out_eval.data))
    print(f"Mean absolute difference: {diff:.6f}")
    print()


def example_image_embedding():
    """Example 9: Image Embedding with Projection."""
    print("=" * 60)
    print("Example 9: Image Embedding with Projection")
    print("=" * 60)
    
    # Create ViT
    vit = VisionTransformer(
        image_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
    )
    
    # Set projection to different dimension
    projection_dim = 128
    vit.set_projection(projection_dim)
    print(f"Set projection to {projection_dim} dimensions")
    
    # Create input
    x = Tensor(np.random.randn(2, 3, 32, 32).astype(np.float32))
    
    # Get projected embeddings
    emb = vit.encode_image(x, normalize=True)
    print(f"Projected embedding shape: {emb.shape}")
    print()


def example_feature_visualization():
    """Example 10: Understanding Feature Dimensions."""
    print("=" * 60)
    print("Example 10: Feature Dimensions Through ViT")
    print("=" * 60)
    
    image_size = 64
    patch_size = 16
    embed_dim = 128
    num_layers = 3
    
    print(f"Image size: {image_size}x{image_size}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Embedding dim: {embed_dim}")
    
    num_patches = (image_size // patch_size) ** 2
    print(f"Number of patches: {num_patches} ({image_size // patch_size}x{image_size // patch_size})")
    print(f"Sequence length (with CLS): {num_patches + 1}")
    
    # Create model
    vit = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=8,
    )
    
    x = Tensor(np.random.randn(1, 3, image_size, image_size).astype(np.float32))
    out = vit(x)
    
    print(f"\nFinal output (CLS token): {out.shape}")
    print(f"Total parameters: {sum(p.data.size for p in vit.parameters()):,}")
    print()


def example_batch_processing():
    """Example 11: Batch Processing with ViT."""
    print("=" * 60)
    print("Example 11: Batch Processing")
    print("=" * 60)
    
    config = ViTConfig.tiny()
    vit = VisionTransformer(
        image_size=config.image_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
    )
    
    # Process different batch sizes
    for batch_size in [1, 4, 8]:
        x = Tensor(np.random.randn(batch_size, 3, config.image_size, config.image_size).astype(np.float32))
        out = vit.encode_image(x, normalize=True)
        print(f"Batch size {batch_size}: input {x.shape} -> output {out.shape}")
    
    print()


def example_cnn_vs_vit():
    """Example 12: Comparing CNN and ViT encoders."""
    print("=" * 60)
    print("Example 12: CNN vs ViT Comparison")
    print("=" * 60)
    
    image_size = 32
    embed_dim = 64
    
    # Create CNN encoder
    cnn = CNNEncoder(
        in_channels=3,
        base_channels=16,
        num_layers=[1, 1, 1, 1],
        embed_dim=embed_dim,
    )
    
    # Create ViT encoder
    vit = VisionTransformer(
        image_size=image_size,
        patch_size=8,
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=4,
    )
    
    # Same input
    x = Tensor(np.random.randn(2, 3, image_size, image_size).astype(np.float32))
    
    # Get embeddings
    cnn_emb = cnn.encode_image(x, normalize=True)
    vit_emb = vit.encode_image(x, normalize=True)
    
    print(f"CNN embedding shape: {cnn_emb.shape}")
    print(f"ViT embedding shape: {vit_emb.shape}")
    
    print(f"\nCNN parameters: {sum(p.data.size for p in cnn.parameters()):,}")
    print(f"ViT parameters: {sum(p.data.size for p in vit.parameters()):,}")
    
    # Compare embeddings (they will be different since random init)
    similarity = np.mean(cnn_emb.data * vit_emb.data)
    print(f"\nAverage embedding similarity: {similarity:.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Vision Encoder Examples")
    print("=" * 60 + "\n")
    
    example_conv_block()
    example_residual_block()
    example_cnn_encoder()
    example_patch_embedding()
    example_vit_block()
    example_vision_transformer()
    example_vit_config()
    example_train_eval_modes()
    example_image_embedding()
    example_feature_visualization()
    example_batch_processing()
    example_cnn_vs_vit()
    
    print("=" * 60)
    print("All vision encoder examples completed!")
    print("=" * 60)
