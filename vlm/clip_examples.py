"""
CLIP (Contrastive Language-Image Pre-training) Examples.

Demonstrates usage of:
- CLIP model creation
- Image and text encoding
- Contrastive loss
- Zero-shot classification
- Image-text retrieval
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnn.tensor import Tensor
from vlm.clip import (
    CLIP,
    CLIPConfig,
    create_clip,
    contrastive_loss,
    CLIPTrainer,
    compute_retrieval_metrics,
    create_text_embeddings_for_classes,
)
from vlm.vision_encoder import VisionTransformer, ViTConfig
from vlm.text_encoder import TextEncoder, TextEncoderConfig, create_text_encoder


def example_clip_creation():
    """Example 1: Creating a CLIP Model."""
    print("=" * 60)
    print("Example 1: CLIP Model Creation")
    print("=" * 60)
    
    # Create CLIP from config
    config = CLIPConfig.tiny(vocab_size=100)
    model = create_clip(config)
    
    print(f"Image size: {config.image_size}")
    print(f"Patch size: {config.patch_size}")
    print(f"Vision dim: {config.vision_dim}")
    print(f"Text dim: {config.text_dim}")
    print(f"Shared embed dim: {config.embed_dim}")
    print(f"Temperature: {model.temperature:.4f}")
    
    total_params = sum(p.data.size for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()


def example_encode_image():
    """Example 2: Encoding Images."""
    print("=" * 60)
    print("Example 2: Image Encoding")
    print("=" * 60)
    
    config = CLIPConfig.tiny()
    model = create_clip(config)
    
    # Create batch of images
    batch_size = 4
    images = np.random.randn(batch_size, 3, config.image_size, config.image_size).astype(np.float32)
    print(f"Input images shape: {images.shape}")
    
    # Encode images
    image_embeddings = model.encode_image(images, normalize=True)
    print(f"Image embeddings shape: {image_embeddings.shape}")
    
    # Verify normalization
    norms = np.sqrt(np.sum(image_embeddings.data ** 2, axis=-1))
    print(f"Embedding norms (should be ~1): {norms}")
    print()


def example_encode_text():
    """Example 3: Encoding Text."""
    print("=" * 60)
    print("Example 3: Text Encoding")
    print("=" * 60)
    
    config = CLIPConfig.tiny(vocab_size=100)
    model = create_clip(config)
    
    # Create batch of token IDs
    batch_size = 4
    seq_len = 16
    input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"Input token IDs shape: {input_ids.shape}")
    
    # Encode text
    text_embeddings = model.encode_text(input_ids, normalize=True)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Verify normalization
    norms = np.sqrt(np.sum(text_embeddings.data ** 2, axis=-1))
    print(f"Embedding norms (should be ~1): {norms}")
    print()


def example_forward_pass():
    """Example 4: Full Forward Pass."""
    print("=" * 60)
    print("Example 4: Full Forward Pass")
    print("=" * 60)
    
    config = CLIPConfig.tiny(vocab_size=100)
    model = create_clip(config)
    
    batch_size = 4
    images = np.random.randn(batch_size, 3, config.image_size, config.image_size).astype(np.float32)
    input_ids = np.random.randint(0, config.vocab_size, (batch_size, 16))
    
    # Forward pass
    image_emb, text_emb, logits_img, logits_txt = model(images, input_ids)
    
    print(f"Image embeddings: {image_emb.shape}")
    print(f"Text embeddings: {text_emb.shape}")
    print(f"Image-to-text logits: {logits_img.shape}")
    print(f"Text-to-image logits: {logits_txt.shape}")
    
    # Temperature scaling
    print(f"\nTemperature: {model.temperature:.4f}")
    print(f"Logits range: [{logits_img.data.min():.2f}, {logits_img.data.max():.2f}]")
    print()


def example_contrastive_loss():
    """Example 5: Contrastive Loss Computation."""
    print("=" * 60)
    print("Example 5: Contrastive Loss")
    print("=" * 60)
    
    batch_size = 8
    embed_dim = 64
    
    # Create random normalized embeddings
    image_emb = np.random.randn(batch_size, embed_dim).astype(np.float32)
    text_emb = np.random.randn(batch_size, embed_dim).astype(np.float32)
    
    # Normalize
    image_emb = image_emb / (np.linalg.norm(image_emb, axis=-1, keepdims=True) + 1e-8)
    text_emb = text_emb / (np.linalg.norm(text_emb, axis=-1, keepdims=True) + 1e-8)
    
    # Compute loss
    loss, logits_i2t, logits_t2i = contrastive_loss(image_emb, text_emb, temperature=0.07)
    
    print(f"Contrastive loss: {loss:.4f}")
    print(f"Image-to-text logits shape: {logits_i2t.shape}")
    print(f"Text-to-image logits shape: {logits_t2i.shape}")
    
    # The diagonal should have highest similarity (matching pairs)
    print(f"\nDiagonal logits (matching pairs):")
    print(f"  {np.diag(logits_i2t)[:4]}...")
    print()


def example_similarity():
    """Example 6: Image-Text Similarity."""
    print("=" * 60)
    print("Example 6: Image-Text Similarity")
    print("=" * 60)
    
    config = CLIPConfig.tiny(vocab_size=100)
    model = create_clip(config)
    
    # Create images and texts
    n_images = 3
    n_texts = 4
    
    images = np.random.randn(n_images, 3, config.image_size, config.image_size).astype(np.float32)
    input_ids = np.random.randint(0, config.vocab_size, (n_texts, 16))
    
    # Compute similarity
    similarity = model.similarity(images, input_ids)
    
    print(f"Images: {n_images}, Texts: {n_texts}")
    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"\nSimilarity matrix:")
    print(similarity)
    print()


def example_compute_loss():
    """Example 7: Training Loss with Metrics."""
    print("=" * 60)
    print("Example 7: Training Loss and Metrics")
    print("=" * 60)
    
    config = CLIPConfig.tiny(vocab_size=100)
    model = create_clip(config)
    model.train()
    
    batch_size = 8
    images = np.random.randn(batch_size, 3, config.image_size, config.image_size).astype(np.float32)
    input_ids = np.random.randint(0, config.vocab_size, (batch_size, 16))
    
    # Compute loss
    loss, metrics = model.compute_loss(images, input_ids)
    
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Image-to-text accuracy: {metrics['i2t_acc']:.2%}")
    print(f"Text-to-image accuracy: {metrics['t2i_acc']:.2%}")
    print(f"Temperature: {metrics['temperature']:.4f}")
    print()


def example_zero_shot_classification():
    """Example 8: Zero-Shot Classification."""
    print("=" * 60)
    print("Example 8: Zero-Shot Classification")
    print("=" * 60)
    
    config = CLIPConfig.tiny(vocab_size=100)
    model = create_clip(config)
    model.eval()
    
    # Create "class" embeddings (in real usage, these would be text embeddings)
    num_classes = 5
    class_embeddings = np.random.randn(num_classes, config.embed_dim).astype(np.float32)
    class_embeddings = class_embeddings / (np.linalg.norm(class_embeddings, axis=-1, keepdims=True) + 1e-8)
    
    # Create test images
    num_images = 10
    images = np.random.randn(num_images, 3, config.image_size, config.image_size).astype(np.float32)
    
    # Zero-shot classify
    predictions = model.zero_shot_classify(images, class_embeddings)
    
    print(f"Number of classes: {num_classes}")
    print(f"Number of images: {num_images}")
    print(f"Predictions: {predictions}")
    print(f"Prediction distribution: {np.bincount(predictions, minlength=num_classes)}")
    print()


def example_retrieval_metrics():
    """Example 9: Retrieval Metrics."""
    print("=" * 60)
    print("Example 9: Retrieval Metrics (Recall@K)")
    print("=" * 60)
    
    num_samples = 20
    embed_dim = 64
    
    # Create embeddings (in practice, from CLIP model)
    image_emb = np.random.randn(num_samples, embed_dim).astype(np.float32)
    text_emb = np.random.randn(num_samples, embed_dim).astype(np.float32)
    
    # Normalize
    image_emb = image_emb / (np.linalg.norm(image_emb, axis=-1, keepdims=True) + 1e-8)
    text_emb = text_emb / (np.linalg.norm(text_emb, axis=-1, keepdims=True) + 1e-8)
    
    # Compute metrics
    metrics = compute_retrieval_metrics(image_emb, text_emb, k_values=[1, 5, 10])
    
    print("Image-to-Text Retrieval:")
    print(f"  Recall@1: {metrics['i2t_recall@1']:.2%}")
    print(f"  Recall@5: {metrics['i2t_recall@5']:.2%}")
    print(f"  Recall@10: {metrics['i2t_recall@10']:.2%}")
    
    print("\nText-to-Image Retrieval:")
    print(f"  Recall@1: {metrics['t2i_recall@1']:.2%}")
    print(f"  Recall@5: {metrics['t2i_recall@5']:.2%}")
    print(f"  Recall@10: {metrics['t2i_recall@10']:.2%}")
    print()


def example_config_presets():
    """Example 10: CLIP Configuration Presets."""
    print("=" * 60)
    print("Example 10: Configuration Presets")
    print("=" * 60)
    
    configs = [
        ("Tiny", CLIPConfig.tiny()),
        ("Small", CLIPConfig.small()),
    ]
    
    for name, config in configs:
        print(f"\n{name} Config:")
        print(f"  Image: {config.image_size}x{config.image_size}, patch={config.patch_size}")
        print(f"  Vision: dim={config.vision_dim}, layers={config.vision_layers}")
        print(f"  Text: dim={config.text_dim}, layers={config.text_layers}")
        print(f"  Shared embed dim: {config.embed_dim}")
        
        # Create model and count params
        model = create_clip(config)
        params = sum(p.data.size for p in model.parameters())
        print(f"  Total parameters: {params:,}")
    print()


def example_trainer():
    """Example 11: CLIP Trainer."""
    print("=" * 60)
    print("Example 11: CLIP Trainer")
    print("=" * 60)
    
    config = CLIPConfig.tiny(vocab_size=100)
    model = create_clip(config)
    
    trainer = CLIPTrainer(
        model=model,
        learning_rate=1e-4,
        warmup_steps=100,
        max_steps=1000,
    )
    
    # Simulate a few training steps
    print("Training steps:")
    for step in range(5):
        images = np.random.randn(4, 3, config.image_size, config.image_size).astype(np.float32)
        input_ids = np.random.randint(0, config.vocab_size, (4, 16))
        
        metrics = trainer.train_step(images, input_ids)
        
        print(f"  Step {metrics['step']}: loss={metrics['loss']:.4f}, "
              f"lr={metrics['learning_rate']:.6f}, "
              f"i2t_acc={metrics['i2t_acc']:.2%}")
    print()


def example_train_eval_modes():
    """Example 12: Training vs Evaluation Modes."""
    print("=" * 60)
    print("Example 12: Train/Eval Modes")
    print("=" * 60)
    
    config = CLIPConfig.tiny(vocab_size=100)
    model = create_clip(config)
    
    images = np.random.randn(2, 3, config.image_size, config.image_size).astype(np.float32)
    
    # Training mode
    model.train()
    print(f"Training mode: {model.training}")
    emb_train = model.encode_image(images)
    
    # Evaluation mode
    model.eval()
    print(f"Eval mode: {not model.training}")
    emb_eval = model.encode_image(images)
    
    # Compare
    diff = np.mean(np.abs(emb_train.data - emb_eval.data))
    print(f"Mean difference: {diff:.6f}")
    print()


def example_custom_clip():
    """Example 13: Custom CLIP with Separate Encoders."""
    print("=" * 60)
    print("Example 13: Custom CLIP Configuration")
    print("=" * 60)
    
    # Create custom vision encoder (embed_dim must match shared dimension)
    embed_dim = 128
    vision_encoder = VisionTransformer(
        image_size=48,
        patch_size=8,
        embed_dim=embed_dim,
        num_layers=3,
        num_heads=4,
    )
    
    # Create custom text encoder (embed_dim must match shared dimension)
    text_config = TextEncoderConfig(
        vocab_size=500,
        max_seq_len=24,
        embed_dim=embed_dim,
        num_layers=3,
        num_heads=4,
    )
    text_encoder = create_text_encoder(text_config)
    
    # Create CLIP with custom encoders
    # Note: embed_dim should match encoder dimensions when no projection is used
    model = CLIP(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        embed_dim=embed_dim,  # Shared dimension matches encoder outputs
        temperature=0.1,
    )
    
    print(f"Vision encoder dim: {vision_encoder.embed_dim}")
    print(f"Text encoder dim: {text_config.embed_dim}")
    print(f"Shared embed dim: {model.embed_dim}")
    print(f"Temperature: {model.temperature:.2f}")
    
    # Test forward pass
    images = np.random.randn(2, 3, 48, 48).astype(np.float32)
    input_ids = np.random.randint(0, 500, (2, 16))
    
    img_emb, txt_emb, _, _ = model(images, input_ids)
    print(f"\nImage embedding: {img_emb.shape}")
    print(f"Text embedding: {txt_emb.shape}")
    print()


def example_learnable_temperature():
    """Example 14: Learnable Temperature."""
    print("=" * 60)
    print("Example 14: Learnable Temperature Parameter")
    print("=" * 60)
    
    # With learnable temperature
    config = CLIPConfig.tiny(vocab_size=100)
    model_learnable = create_clip(config)
    
    print(f"Initial temperature: {model_learnable.temperature:.4f}")
    print(f"Log temperature requires_grad: {model_learnable.log_temperature.requires_grad}")
    
    # Temperature is in the parameters
    params = model_learnable.parameters()
    temp_param = [p for p in params if p.data.shape == (1,)]
    print(f"Temperature parameter found in model.parameters(): {len(temp_param) > 0}")
    print()


def example_embedding_space():
    """Example 15: Shared Embedding Space Visualization."""
    print("=" * 60)
    print("Example 15: Shared Embedding Space")
    print("=" * 60)
    
    config = CLIPConfig.tiny(vocab_size=100)
    model = create_clip(config)
    model.eval()
    
    # Generate embeddings for images and texts
    n_samples = 10
    images = np.random.randn(n_samples, 3, config.image_size, config.image_size).astype(np.float32)
    input_ids = np.random.randint(0, config.vocab_size, (n_samples, 16))
    
    img_emb = model.encode_image(images, normalize=True)
    txt_emb = model.encode_text(input_ids, normalize=True)
    
    # All embeddings are in the same space
    print(f"Image embeddings: {img_emb.shape}")
    print(f"Text embeddings: {txt_emb.shape}")
    
    # Cross-modal similarity
    cross_sim = img_emb.data @ txt_emb.data.T
    
    print(f"\nCross-modal similarity matrix: {cross_sim.shape}")
    print(f"Similarity range: [{cross_sim.min():.3f}, {cross_sim.max():.3f}]")
    print(f"Mean similarity: {cross_sim.mean():.3f}")
    
    # In a trained CLIP, diagonal would be high (matching pairs)
    # With random init, all similarities are similar
    print(f"Diagonal mean: {np.mean(np.diag(cross_sim)):.3f}")
    print(f"Off-diagonal mean: {np.mean(cross_sim - np.diag(np.diag(cross_sim))):.3f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CLIP Examples")
    print("=" * 60 + "\n")
    
    example_clip_creation()
    example_encode_image()
    example_encode_text()
    example_forward_pass()
    example_contrastive_loss()
    example_similarity()
    example_compute_loss()
    example_zero_shot_classification()
    example_retrieval_metrics()
    example_config_presets()
    example_trainer()
    example_train_eval_modes()
    example_custom_clip()
    example_learnable_temperature()
    example_embedding_space()
    
    print("=" * 60)
    print("All CLIP examples completed!")
    print("=" * 60)
