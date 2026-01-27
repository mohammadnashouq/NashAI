"""
CLIP (Contrastive Language-Image Pre-training) Implementation.

Implements:
- CLIP model combining vision and text encoders
- Contrastive learning loss (InfoNCE)
- Image-text similarity computation
- Zero-shot classification
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Union
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnn.tensor import Tensor
from .vision_encoder import VisionTransformer, CNNEncoder, create_vision_encoder, ViTConfig
from .text_encoder import TextEncoder, create_text_encoder, TextEncoderConfig


# =============================================================================
# Contrastive Loss
# =============================================================================

def contrastive_loss(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    temperature: float = 0.07,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute symmetric contrastive loss (InfoNCE) for CLIP.
    
    Args:
        image_embeddings: Normalized image embeddings (batch, embed_dim)
        text_embeddings: Normalized text embeddings (batch, embed_dim)
        temperature: Temperature parameter for scaling logits
    
    Returns:
        loss: Scalar contrastive loss
        logits_per_image: Image-to-text similarity scores
        logits_per_text: Text-to-image similarity scores
    
    The loss encourages:
    - Matching image-text pairs to have high similarity
    - Non-matching pairs to have low similarity
    """
    batch_size = image_embeddings.shape[0]
    
    # Compute cosine similarity (already normalized, so just dot product)
    # logits_per_image[i, j] = similarity of image i with text j
    logits_per_image = (image_embeddings @ text_embeddings.T) / temperature
    logits_per_text = logits_per_image.T
    
    # Labels: diagonal elements are positive pairs
    labels = np.arange(batch_size)
    
    # Cross-entropy loss for both directions
    # Image -> Text direction
    image_loss = _cross_entropy(logits_per_image, labels)
    
    # Text -> Image direction
    text_loss = _cross_entropy(logits_per_text, labels)
    
    # Symmetric loss
    loss = (image_loss + text_loss) / 2
    
    return loss, logits_per_image, logits_per_text


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Raw logits of shape (batch, num_classes)
        labels: Ground truth labels of shape (batch,)
    
    Returns:
        Scalar loss value
    """
    batch_size = logits.shape[0]
    
    # Stable softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-9)
    
    # Get probability of correct class
    correct_probs = probs[np.arange(batch_size), labels]
    
    # Negative log likelihood
    loss = -np.mean(np.log(correct_probs + 1e-9))
    
    return loss


def _cross_entropy_backward(
    logits: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute gradient of cross-entropy loss.
    
    Args:
        logits: Raw logits of shape (batch, num_classes)
        labels: Ground truth labels of shape (batch,)
    
    Returns:
        Gradient with respect to logits
    """
    batch_size = logits.shape[0]
    
    # Softmax probabilities
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-9)
    
    # Gradient: softmax - one_hot
    grad = probs.copy()
    grad[np.arange(batch_size), labels] -= 1
    grad /= batch_size
    
    return grad


# =============================================================================
# CLIP Model
# =============================================================================

class CLIP:
    """
    CLIP (Contrastive Language-Image Pre-training) Model.
    
    Combines a vision encoder and text encoder in a shared embedding space
    through contrastive learning.
    
    Architecture:
    - Vision Encoder: ViT or CNN that produces image embeddings
    - Text Encoder: Transformer that produces text embeddings
    - Projection: Both encoders project to shared embedding space
    - Contrastive Loss: InfoNCE loss on image-text pairs
    """
    
    def __init__(
        self,
        vision_encoder: Union[VisionTransformer, CNNEncoder],
        text_encoder: TextEncoder,
        embed_dim: int = 512,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        """
        Args:
            vision_encoder: Vision encoder (ViT or CNN)
            text_encoder: Text encoder (Transformer)
            embed_dim: Shared embedding dimension
            temperature: Initial temperature for contrastive loss
            learnable_temperature: Whether temperature is learnable
        """
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.embed_dim = embed_dim
        self.training = True
        
        # Temperature parameter (log scale for numerical stability)
        self.log_temperature = Tensor(
            np.array([np.log(temperature)], dtype=np.float32),
            requires_grad=learnable_temperature
        )
        
        # Set up projections if needed
        self._setup_projections(embed_dim)
    
    def _setup_projections(self, embed_dim: int):
        """Set up projection layers to shared embedding space."""
        # Vision projection
        if hasattr(self.vision_encoder, 'set_projection'):
            self.vision_encoder.set_projection(embed_dim)
        
        # Text projection
        if hasattr(self.text_encoder, 'set_projection'):
            self.text_encoder.set_projection(embed_dim)
    
    @property
    def temperature(self) -> float:
        """Get current temperature value."""
        return np.exp(self.log_temperature.data[0])
    
    def encode_image(
        self,
        images: np.ndarray,
        normalize: bool = True,
    ) -> Tensor:
        """
        Encode images to embedding space.
        
        Args:
            images: Image batch of shape (batch, channels, height, width)
            normalize: Whether to L2 normalize embeddings
        
        Returns:
            Image embeddings of shape (batch, embed_dim)
        """
        # Convert to Tensor if needed
        if not isinstance(images, Tensor):
            images = Tensor(images.astype(np.float32), requires_grad=False)
        
        # Get embeddings from vision encoder
        if hasattr(self.vision_encoder, 'encode_image'):
            embeddings = self.vision_encoder.encode_image(images, normalize=normalize)
        else:
            embeddings = self.vision_encoder(images)
            if normalize:
                norm = np.sqrt(np.sum(embeddings.data ** 2, axis=-1, keepdims=True) + 1e-8)
                embeddings = Tensor(embeddings.data / norm, requires_grad=embeddings.requires_grad)
        
        return embeddings
    
    def encode_text(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        normalize: bool = True,
    ) -> Tensor:
        """
        Encode text to embedding space.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
            normalize: Whether to L2 normalize embeddings
        
        Returns:
            Text embeddings of shape (batch, embed_dim)
        """
        # Get embeddings from text encoder
        if hasattr(self.text_encoder, 'encode_text'):
            embeddings = self.text_encoder.encode_text(
                input_ids, attention_mask, normalize=normalize
            )
        else:
            embeddings = self.text_encoder(input_ids, attention_mask)
            if normalize:
                norm = np.sqrt(np.sum(embeddings.data ** 2, axis=-1, keepdims=True) + 1e-8)
                embeddings = Tensor(embeddings.data / norm, requires_grad=embeddings.requires_grad)
        
        return embeddings
    
    def forward(
        self,
        images: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass computing image and text embeddings.
        
        Args:
            images: Image batch (batch, channels, height, width)
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional attention mask
        
        Returns:
            image_embeddings: Normalized image embeddings
            text_embeddings: Normalized text embeddings
            logits_per_image: Image-to-text similarity
            logits_per_text: Text-to-image similarity
        """
        # Encode both modalities
        image_embeddings = self.encode_image(images, normalize=True)
        text_embeddings = self.encode_text(input_ids, attention_mask, normalize=True)
        
        # Compute similarity logits
        temperature = self.temperature
        logits_per_image = Tensor(
            (image_embeddings.data @ text_embeddings.data.T) / temperature,
            requires_grad=True
        )
        logits_per_text = Tensor(
            logits_per_image.data.T,
            requires_grad=True
        )
        
        return image_embeddings, text_embeddings, logits_per_image, logits_per_text
    
    def __call__(
        self,
        images: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass."""
        return self.forward(images, input_ids, attention_mask)
    
    def compute_loss(
        self,
        images: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute contrastive loss for a batch.
        
        Args:
            images: Image batch (batch, channels, height, width)
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional attention mask
        
        Returns:
            loss: Contrastive loss as Tensor
            metrics: Dictionary with loss components
        """
        # Forward pass
        image_emb, text_emb, logits_img, logits_txt = self.forward(
            images, input_ids, attention_mask
        )
        
        # Compute contrastive loss
        loss, _, _ = contrastive_loss(
            image_emb.data,
            text_emb.data,
            self.temperature
        )
        
        # Compute accuracy metrics
        batch_size = images.shape[0]
        labels = np.arange(batch_size)
        
        # Image -> Text accuracy
        i2t_preds = np.argmax(logits_img.data, axis=-1)
        i2t_acc = np.mean(i2t_preds == labels)
        
        # Text -> Image accuracy
        t2i_preds = np.argmax(logits_txt.data, axis=-1)
        t2i_acc = np.mean(t2i_preds == labels)
        
        metrics = {
            'loss': float(loss),
            'i2t_acc': float(i2t_acc),
            't2i_acc': float(t2i_acc),
            'temperature': float(self.temperature),
        }
        
        return Tensor(np.array([loss], dtype=np.float32), requires_grad=True), metrics
    
    def similarity(
        self,
        images: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute cosine similarity between images and texts.
        
        Args:
            images: Image batch (N, channels, height, width)
            input_ids: Token IDs (M, seq_len)
            attention_mask: Optional attention mask
        
        Returns:
            Similarity matrix of shape (N, M)
        """
        image_emb = self.encode_image(images, normalize=True)
        text_emb = self.encode_text(input_ids, attention_mask, normalize=True)
        
        return image_emb.data @ text_emb.data.T
    
    def zero_shot_classify(
        self,
        images: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Zero-shot image classification using precomputed text embeddings.
        
        Args:
            images: Image batch (batch, channels, height, width)
            text_embeddings: Precomputed text embeddings for classes (num_classes, embed_dim)
        
        Returns:
            Predicted class indices of shape (batch,)
        """
        image_emb = self.encode_image(images, normalize=True)
        
        # Compute similarities
        similarities = image_emb.data @ text_embeddings.T
        
        # Return argmax
        return np.argmax(similarities, axis=-1)
    
    def parameters(self) -> List[Tensor]:
        """Get all trainable parameters."""
        params = []
        params.extend(self.vision_encoder.parameters())
        params.extend(self.text_encoder.parameters())
        params.append(self.log_temperature)
        return params
    
    def train(self):
        """Set to training mode."""
        self.training = True
        self.vision_encoder.train()
        self.text_encoder.train()
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
        self.vision_encoder.eval()
        self.text_encoder.eval()


# =============================================================================
# CLIP Configuration
# =============================================================================

class CLIPConfig:
    """Configuration for CLIP model."""
    
    def __init__(
        self,
        # Vision encoder
        image_size: int = 224,
        patch_size: int = 16,
        vision_layers: int = 12,
        vision_heads: int = 12,
        vision_dim: int = 768,
        # Text encoder
        vocab_size: int = 49408,
        max_seq_len: int = 77,
        text_layers: int = 12,
        text_heads: int = 8,
        text_dim: int = 512,
        # Shared
        embed_dim: int = 512,
        temperature: float = 0.07,
        vision_type: str = 'vit',  # 'vit' or 'cnn'
    ):
        # Vision
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_layers = vision_layers
        self.vision_heads = vision_heads
        self.vision_dim = vision_dim
        self.vision_type = vision_type
        
        # Text
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.text_layers = text_layers
        self.text_heads = text_heads
        self.text_dim = text_dim
        
        # Shared
        self.embed_dim = embed_dim
        self.temperature = temperature
    
    @classmethod
    def vit_base(cls) -> 'CLIPConfig':
        """CLIP ViT-B/16 configuration."""
        return cls(
            image_size=224,
            patch_size=16,
            vision_layers=12,
            vision_heads=12,
            vision_dim=768,
            vocab_size=49408,
            max_seq_len=77,
            text_layers=12,
            text_heads=8,
            text_dim=512,
            embed_dim=512,
            vision_type='vit',
        )
    
    @classmethod
    def vit_large(cls) -> 'CLIPConfig':
        """CLIP ViT-L/14 configuration."""
        return cls(
            image_size=224,
            patch_size=14,
            vision_layers=24,
            vision_heads=16,
            vision_dim=1024,
            vocab_size=49408,
            max_seq_len=77,
            text_layers=12,
            text_heads=12,
            text_dim=768,
            embed_dim=768,
            vision_type='vit',
        )
    
    @classmethod
    def tiny(cls, vocab_size: int = 1000) -> 'CLIPConfig':
        """Tiny configuration for testing."""
        return cls(
            image_size=32,
            patch_size=8,
            vision_layers=2,
            vision_heads=4,
            vision_dim=64,
            vocab_size=vocab_size,
            max_seq_len=32,
            text_layers=2,
            text_heads=4,
            text_dim=64,
            embed_dim=64,
            vision_type='vit',
        )
    
    @classmethod
    def small(cls, vocab_size: int = 10000) -> 'CLIPConfig':
        """Small configuration for experiments."""
        return cls(
            image_size=64,
            patch_size=8,
            vision_layers=4,
            vision_heads=8,
            vision_dim=256,
            vocab_size=vocab_size,
            max_seq_len=64,
            text_layers=4,
            text_heads=8,
            text_dim=256,
            embed_dim=256,
            vision_type='vit',
        )


def create_clip(config: CLIPConfig) -> CLIP:
    """
    Create CLIP model from configuration.
    
    Args:
        config: CLIP configuration
    
    Returns:
        CLIP model instance
    """
    # Create vision encoder
    if config.vision_type == 'vit':
        vision_encoder = VisionTransformer(
            image_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.vision_dim,
            num_layers=config.vision_layers,
            num_heads=config.vision_heads,
            in_channels=3,
        )
    else:
        # CNN encoder
        vision_encoder = CNNEncoder(
            in_channels=3,
            base_channels=64,
            num_blocks=[2, 2, 2, 2],
            embed_dim=config.vision_dim,
        )
    
    # Create text encoder
    text_config = TextEncoderConfig(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        embed_dim=config.text_dim,
        num_layers=config.text_layers,
        num_heads=config.text_heads,
    )
    text_encoder = create_text_encoder(text_config)
    
    # Create CLIP model
    clip = CLIP(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        embed_dim=config.embed_dim,
        temperature=config.temperature,
    )
    
    return clip


# =============================================================================
# Training Utilities
# =============================================================================

class CLIPTrainer:
    """Training utilities for CLIP model."""
    
    def __init__(
        self,
        model: CLIP,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 10000,
        max_steps: int = 100000,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.step = 0
        
        # Initialize optimizer state (AdamW-style)
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.eps = 1e-6
    
    def get_lr(self) -> float:
        """Get learning rate with warmup and cosine decay."""
        if self.step < self.warmup_steps:
            # Linear warmup
            return self.learning_rate * self.step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
    
    def train_step(
        self,
        images: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            images: Image batch
            input_ids: Token IDs
            attention_mask: Optional attention mask
        
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Forward pass and compute loss
        loss, metrics = self.model.compute_loss(images, input_ids, attention_mask)
        
        # For demonstration, we simulate gradient update
        # In a real implementation, autograd would compute gradients
        lr = self.get_lr()
        metrics['learning_rate'] = lr
        metrics['step'] = self.step
        
        self.step += 1
        
        return metrics


def compute_retrieval_metrics(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute retrieval metrics (Recall@K).
    
    Args:
        image_embeddings: Image embeddings (N, D)
        text_embeddings: Text embeddings (N, D)
        k_values: List of K values for Recall@K
    
    Returns:
        Dictionary with retrieval metrics
    """
    n_samples = image_embeddings.shape[0]
    
    # Compute similarity matrix
    similarity = image_embeddings @ text_embeddings.T  # (N, N)
    
    # Ground truth: diagonal is correct
    labels = np.arange(n_samples)
    
    metrics = {}
    
    # Image -> Text retrieval
    i2t_ranks = np.argsort(-similarity, axis=1)  # Sorted by descending similarity
    for k in k_values:
        correct = np.sum([labels[i] in i2t_ranks[i, :k] for i in range(n_samples)])
        metrics[f'i2t_recall@{k}'] = correct / n_samples
    
    # Text -> Image retrieval
    t2i_ranks = np.argsort(-similarity.T, axis=1)
    for k in k_values:
        correct = np.sum([labels[i] in t2i_ranks[i, :k] for i in range(n_samples)])
        metrics[f't2i_recall@{k}'] = correct / n_samples
    
    return metrics


def create_text_embeddings_for_classes(
    model: CLIP,
    class_names: List[str],
    templates: List[str] = None,
    tokenizer = None,
) -> np.ndarray:
    """
    Create text embeddings for zero-shot classification.
    
    Args:
        model: CLIP model
        class_names: List of class names
        templates: List of prompt templates (default: ["a photo of a {}"])
        tokenizer: Tokenizer for encoding text
    
    Returns:
        Averaged text embeddings (num_classes, embed_dim)
    """
    if templates is None:
        templates = [
            "a photo of a {}",
            "a picture of a {}",
            "an image of a {}",
        ]
    
    model.eval()
    
    all_embeddings = []
    
    for class_name in class_names:
        class_embeddings = []
        
        for template in templates:
            text = template.format(class_name)
            
            # Tokenize (simple character-level if no tokenizer provided)
            if tokenizer is not None:
                tokens = tokenizer.encode(text)
            else:
                # Simple tokenization for demonstration
                tokens = [ord(c) % 1000 for c in text]
            
            # Pad/truncate
            max_len = model.text_encoder.max_seq_len
            if len(tokens) < max_len:
                tokens = tokens + [0] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
            
            input_ids = np.array([tokens], dtype=np.int64)
            
            # Get embedding
            embedding = model.encode_text(input_ids, normalize=True)
            class_embeddings.append(embedding.data[0])
        
        # Average embeddings for this class
        avg_embedding = np.mean(class_embeddings, axis=0)
        
        # Renormalize
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        
        all_embeddings.append(avg_embedding)
    
    return np.array(all_embeddings)
