"""
VLM (Vision-Language Model) module.

A complete implementation of vision-language models from scratch.

Components:
- Vision Encoders: CNN (ResNet-style), Vision Transformer (ViT)
- Text Encoder: Transformer-based text encoder
- CLIP: Contrastive Language-Image Pre-training

Example usage:
    from vlm import CLIP, CLIPConfig, create_clip
    
    # Create CLIP model
    config = CLIPConfig.tiny(vocab_size=1000)
    model = create_clip(config)
    
    # Encode images and text
    image_emb = model.encode_image(images)
    text_emb = model.encode_text(input_ids)
    
    # Compute similarity
    similarity = model.similarity(images, input_ids)
    
    # Zero-shot classification
    predictions = model.zero_shot_classify(images, text_embeddings)
"""

# Vision Encoder
from .vision_encoder import (
    # CNN components
    ConvBlock,
    ResidualBlock,
    CNNEncoder,
    # ViT components
    PatchEmbedding,
    ViTBlock,
    VisionTransformer,
    # Configuration
    ViTConfig,
    create_vision_encoder,
)

# Text Encoder
from .text_encoder import (
    TextTransformerBlock,
    TextEncoder,
    TextEncoderConfig,
    create_text_encoder,
)

# CLIP
from .clip import (
    # Loss functions
    contrastive_loss,
    # CLIP model
    CLIP,
    CLIPConfig,
    create_clip,
    # Training utilities
    CLIPTrainer,
    compute_retrieval_metrics,
    create_text_embeddings_for_classes,
)

__all__ = [
    # Vision Encoder
    'ConvBlock',
    'ResidualBlock',
    'CNNEncoder',
    'PatchEmbedding',
    'ViTBlock',
    'VisionTransformer',
    'ViTConfig',
    'create_vision_encoder',
    # Text Encoder
    'TextTransformerBlock',
    'TextEncoder',
    'TextEncoderConfig',
    'create_text_encoder',
    # CLIP
    'contrastive_loss',
    'CLIP',
    'CLIPConfig',
    'create_clip',
    'CLIPTrainer',
    'compute_retrieval_metrics',
    'create_text_embeddings_for_classes',
]
