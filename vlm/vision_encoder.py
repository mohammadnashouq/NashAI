"""
Vision Encoders for VLMs.

Implements:
- CNN-based vision encoders (ResNet-style)
- Vision Transformer (ViT)
- Image patch embedding
- Image feature extraction
"""

import numpy as np
from typing import Optional, List, Tuple, Union
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnn.tensor import Tensor
from nnn.conv import Conv2D, MaxPool2D, AvgPool2D, BatchNorm2D, Flatten
from nnn.activations import relu, gelu


# =============================================================================
# Utility Functions
# =============================================================================

def _im2col(input_data, kernel_h, kernel_w, stride, padding):
    """Transform input patches to columns for efficient convolution."""
    N, C, H, W = input_data.shape
    
    # Add padding
    if padding > 0:
        input_padded = np.pad(
            input_data,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        input_padded = input_data
    
    H_padded, W_padded = input_padded.shape[2], input_padded.shape[3]
    
    out_h = (H_padded - kernel_h) // stride + 1
    out_w = (W_padded - kernel_w) // stride + 1
    
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=input_data.dtype)
    
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = input_padded[:, :, y:y_max:stride, x:x_max:stride]
    
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


# =============================================================================
# CNN Building Blocks
# =============================================================================

class ConvBlock:
    """
    Convolutional block: Conv2D -> BatchNorm -> ReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
    ):
        self.conv = Conv2D(in_channels, out_channels, kernel_size, stride, padding)
        self.use_bn = use_bn
        if use_bn:
            self.bn = BatchNorm2D(out_channels)
        self.training = True
    
    def __call__(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        out = relu(out)
        return out
    
    def parameters(self) -> List[Tensor]:
        params = self.conv.parameters()
        if self.use_bn:
            params.extend(self.bn.parameters())
        return params
    
    def train(self):
        self.training = True
        if self.use_bn:
            self.bn.train()
    
    def eval(self):
        self.training = False
        if self.use_bn:
            self.bn.eval()


class ResidualBlock:
    """
    Residual block with skip connection.
    
    x -> Conv -> BN -> ReLU -> Conv -> BN -> + -> ReLU
    |                                        |
    +------------- (shortcut) ---------------+
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride, 1)
        self.bn1 = BatchNorm2D(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, 3, 1, 1)
        self.bn2 = BatchNorm2D(out_channels)
        
        # Shortcut connection
        self.use_shortcut = (in_channels != out_channels) or (stride != 1)
        if self.use_shortcut:
            self.shortcut_conv = Conv2D(in_channels, out_channels, 1, stride, 0)
            self.shortcut_bn = BatchNorm2D(out_channels)
        
        self.training = True
    
    def __call__(self, x: Tensor) -> Tensor:
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut
        if self.use_shortcut:
            identity = self.shortcut_conv(identity)
            identity = self.shortcut_bn(identity)
        
        # Add residual
        out_data = out.data + identity.data
        out = Tensor(out_data, requires_grad=out.requires_grad or identity.requires_grad)
        
        # Final activation
        out = relu(out)
        return out
    
    def parameters(self) -> List[Tensor]:
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.bn1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.bn2.parameters())
        if self.use_shortcut:
            params.extend(self.shortcut_conv.parameters())
            params.extend(self.shortcut_bn.parameters())
        return params
    
    def train(self):
        self.training = True
        self.bn1.train()
        self.bn2.train()
        if self.use_shortcut:
            self.shortcut_bn.train()
    
    def eval(self):
        self.training = False
        self.bn1.eval()
        self.bn2.eval()
        if self.use_shortcut:
            self.shortcut_bn.eval()


# =============================================================================
# CNN Vision Encoder
# =============================================================================

class CNNEncoder:
    """
    CNN-based vision encoder (ResNet-style).
    
    Produces image embeddings from raw pixel inputs.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 512,
        num_layers: List[int] = [2, 2, 2, 2],
        base_channels: int = 64,
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Output embedding dimension
            num_layers: Number of residual blocks per stage
            base_channels: Base number of channels
        """
        self.embed_dim = embed_dim
        self.training = True
        
        # Initial convolution
        self.conv1 = Conv2D(in_channels, base_channels, 7, 2, 3)
        self.bn1 = BatchNorm2D(base_channels)
        self.pool1 = MaxPool2D(3, 2, 1)
        
        # Residual stages
        self.stages = []
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        
        in_ch = base_channels
        for stage_idx, num_blocks in enumerate(num_layers):
            out_ch = channels[stage_idx]
            stride = 1 if stage_idx == 0 else 2
            
            stage = []
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                block_in_ch = in_ch if block_idx == 0 else out_ch
                stage.append(ResidualBlock(block_in_ch, out_ch, block_stride))
            
            self.stages.append(stage)
            in_ch = out_ch
        
        # Global average pooling + projection
        self.final_channels = channels[-1]
        limit = np.sqrt(6.0 / (self.final_channels + embed_dim))
        self.fc = Tensor(
            np.random.uniform(-limit, limit, (self.final_channels, embed_dim)).astype(np.float32),
            requires_grad=True
        )
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Image embeddings of shape (batch, embed_dim)
        """
        # Initial layers
        out = self.conv1(x)
        out = self.bn1(out)
        out = relu(out)
        out = self.pool1(out)
        
        # Residual stages
        for stage in self.stages:
            for block in stage:
                out = block(out)
        
        # Global average pooling
        out_data = np.mean(out.data, axis=(2, 3))  # (batch, channels)
        
        # Project to embedding dimension
        embeddings = out_data @ self.fc.data  # (batch, embed_dim)
        
        return Tensor(embeddings, requires_grad=out.requires_grad)
    
    def parameters(self) -> List[Tensor]:
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.bn1.parameters())
        
        for stage in self.stages:
            for block in stage:
                params.extend(block.parameters())
        
        params.append(self.fc)
        return params
    
    def train(self):
        self.training = True
        self.bn1.train()
        for stage in self.stages:
            for block in stage:
                block.train()
    
    def eval(self):
        self.training = False
        self.bn1.eval()
        for stage in self.stages:
            for block in stage:
                block.eval()
    
    def encode_image(self, x: Tensor, normalize: bool = True) -> Tensor:
        """
        Encode images with optional L2 normalization.
        
        Args:
            x: Input images of shape (batch, channels, height, width)
            normalize: Whether to L2 normalize embeddings
        
        Returns:
            Image embeddings of shape (batch, embed_dim)
        """
        embeddings = self(x)
        
        if normalize:
            norm = np.sqrt(np.sum(embeddings.data ** 2, axis=-1, keepdims=True) + 1e-8)
            embeddings = Tensor(embeddings.data / norm, requires_grad=embeddings.requires_grad)
        
        return embeddings


# =============================================================================
# Vision Transformer Components
# =============================================================================

class PatchEmbedding:
    """
    Split image into patches and embed them.
    
    Image (B, C, H, W) -> Patches (B, num_patches, embed_dim)
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        
        # Linear projection of flattened patches
        limit = np.sqrt(6.0 / (patch_dim + embed_dim))
        self.projection = Tensor(
            np.random.uniform(-limit, limit, (patch_dim, embed_dim)).astype(np.float32),
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(embed_dim, dtype=np.float32), requires_grad=True)
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Extract patches
        patches = self._extract_patches(x.data)  # (batch, num_patches, patch_dim)
        
        # Project to embedding dimension
        embeddings = patches @ self.projection.data + self.bias.data
        
        return Tensor(embeddings, requires_grad=x.requires_grad)
    
    def _extract_patches(self, x: np.ndarray) -> np.ndarray:
        """Extract non-overlapping patches from images."""
        batch_size, C, H, W = x.shape
        P = self.patch_size
        
        # Reshape to extract patches
        # (B, C, H, W) -> (B, C, H/P, P, W/P, P) -> (B, H/P, W/P, C, P, P)
        x = x.reshape(batch_size, C, H // P, P, W // P, P)
        x = x.transpose(0, 2, 4, 1, 3, 5)  # (B, H/P, W/P, C, P, P)
        
        # Flatten patches
        patches = x.reshape(batch_size, self.num_patches, -1)  # (B, num_patches, C*P*P)
        
        return patches.astype(np.float32)
    
    def parameters(self) -> List[Tensor]:
        return [self.projection, self.bias]


class ViTBlock:
    """
    Vision Transformer block.
    
    LayerNorm -> Multi-Head Self-Attention -> Residual
    LayerNorm -> MLP -> Residual
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
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
        
        # Multi-head self-attention
        limit = np.sqrt(6.0 / (embed_dim + embed_dim))
        self.W_qkv = Tensor(
            np.random.uniform(-limit, limit, (embed_dim, 3 * embed_dim)).astype(np.float32),
            requires_grad=True
        )
        self.W_o = Tensor(
            np.random.uniform(-limit, limit, (embed_dim, embed_dim)).astype(np.float32),
            requires_grad=True
        )
        
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
    
    def _layer_norm(self, x: np.ndarray, ln: dict, eps: float = 1e-5) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return ln['gamma'].data * x_norm + ln['beta'].data
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
        
        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Self-attention
        x_norm = self._layer_norm(x.data, self.ln1)
        attn_out = self._multi_head_attention(x_norm)
        x_data = x.data + attn_out  # Residual
        
        # MLP
        x_norm = self._layer_norm(x_data, self.ln2)
        mlp_out = self._mlp(x_norm)
        out_data = x_data + mlp_out  # Residual
        
        return Tensor(out_data, requires_grad=x.requires_grad)
    
    def _multi_head_attention(self, x: np.ndarray) -> np.ndarray:
        """Multi-head self-attention."""
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = x @ self.W_qkv.data  # (B, S, 3*D)
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        
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


class VisionTransformer:
    """
    Vision Transformer (ViT) for image encoding.
    
    Splits images into patches, adds positional embeddings,
    and processes through transformer blocks.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cls_token: bool = True,
    ):
        """
        Args:
            image_size: Input image size
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            use_cls_token: Whether to use CLS token for pooling
        """
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.training = True
        self.projection = None  # Optional projection layer

        # Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token
        if use_cls_token:
            self.cls_token = Tensor(
                np.random.randn(1, 1, embed_dim).astype(np.float32) * 0.02,
                requires_grad=True
            )
            num_positions = num_patches + 1
        else:
            self.cls_token = None
            num_positions = num_patches
        
        # Position embeddings
        self.pos_embed = Tensor(
            np.random.randn(1, num_positions, embed_dim).astype(np.float32) * 0.02,
            requires_grad=True
        )
        
        # Transformer blocks
        self.blocks = [
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.ln_final = {
            'gamma': Tensor(np.ones(embed_dim, dtype=np.float32), requires_grad=True),
            'beta': Tensor(np.zeros(embed_dim, dtype=np.float32), requires_grad=True),
        }
    
    def __call__(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Image embeddings of shape (batch, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        if self.use_cls_token:
            cls_tokens = np.broadcast_to(self.cls_token.data, (batch_size, 1, self.embed_dim))
            x_data = np.concatenate([cls_tokens, x.data], axis=1)
        else:
            x_data = x.data
        
        # Add position embeddings
        x_data = x_data + self.pos_embed.data
        
        # Apply transformer blocks
        for block in self.blocks:
            x = Tensor(x_data, requires_grad=True)
            x = block(x)
            x_data = x.data
        
        # Final layer norm
        mean = np.mean(x_data, axis=-1, keepdims=True)
        var = np.var(x_data, axis=-1, keepdims=True)
        x_norm = (x_data - mean) / np.sqrt(var + 1e-5)
        x_data = self.ln_final['gamma'].data * x_norm + self.ln_final['beta'].data
        
        # Pool: use CLS token or mean pool
        if self.use_cls_token:
            embeddings = x_data[:, 0, :]  # CLS token
        else:
            embeddings = np.mean(x_data, axis=1)  # Mean pool
        
        return Tensor(embeddings, requires_grad=True)
    
    def parameters(self) -> List[Tensor]:
        params = self.patch_embed.parameters()
        
        if self.use_cls_token:
            params.append(self.cls_token)
        
        params.append(self.pos_embed)
        
        for block in self.blocks:
            params.extend(block.parameters())
        
        params.extend([self.ln_final['gamma'], self.ln_final['beta']])
        
        return params
    
    def train(self):
        self.training = True
        for block in self.blocks:
            block.train()
    
    def eval(self):
        self.training = False
        for block in self.blocks:
            block.eval()
    
    def encode_image(self, x: Tensor, normalize: bool = True) -> Tensor:
        """
        Encode images with optional L2 normalization.
        
        Args:
            x: Input images of shape (batch, channels, height, width)
            normalize: Whether to L2 normalize embeddings
        
        Returns:
            Image embeddings of shape (batch, embed_dim) or (batch, projection_dim)
        """
        embeddings = self(x)
        
        # Apply projection if set
        if self.projection is not None:
            embeddings = Tensor(
                embeddings.data @ self.projection.data,
                requires_grad=embeddings.requires_grad
            )
        
        if normalize:
            norm = np.sqrt(np.sum(embeddings.data ** 2, axis=-1, keepdims=True) + 1e-8)
            embeddings = Tensor(embeddings.data / norm, requires_grad=embeddings.requires_grad)
        
        return embeddings
    
    def set_projection(self, projection_dim: int):
        """Set up projection layer to a different embedding dimension."""
        limit = np.sqrt(6.0 / (self.embed_dim + projection_dim))
        self.projection = Tensor(
            np.random.uniform(-limit, limit, (self.embed_dim, projection_dim)).astype(np.float32),
            requires_grad=True
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_vision_encoder(
    encoder_type: str = 'vit',
    **kwargs
) -> Union[CNNEncoder, VisionTransformer]:
    """
    Factory function to create vision encoders.
    
    Args:
        encoder_type: 'cnn' or 'vit'
        **kwargs: Encoder-specific arguments
    
    Returns:
        Vision encoder instance
    """
    if encoder_type == 'cnn':
        return CNNEncoder(**kwargs)
    elif encoder_type == 'vit':
        return VisionTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


class ViTConfig:
    """Configuration for Vision Transformer."""
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
    
    @classmethod
    def vit_tiny(cls) -> 'ViTConfig':
        """ViT-Tiny configuration."""
        return cls(
            image_size=224,
            patch_size=16,
            embed_dim=192,
            num_layers=12,
            num_heads=3,
        )
    
    @classmethod
    def vit_small(cls) -> 'ViTConfig':
        """ViT-Small configuration."""
        return cls(
            image_size=224,
            patch_size=16,
            embed_dim=384,
            num_layers=12,
            num_heads=6,
        )
    
    @classmethod
    def vit_base(cls) -> 'ViTConfig':
        """ViT-Base configuration."""
        return cls(
            image_size=224,
            patch_size=16,
            embed_dim=768,
            num_layers=12,
            num_heads=12,
        )
    
    @classmethod
    def vit_tiny_patch4(cls, image_size: int = 32) -> 'ViTConfig':
        """Tiny ViT with small patches for testing."""
        return cls(
            image_size=image_size,
            patch_size=4,
            embed_dim=64,
            num_layers=4,
            num_heads=4,
            mlp_ratio=2.0,
        )
    
    @classmethod
    def tiny(cls) -> 'ViTConfig':
        """Tiny configuration for testing (alias for vit_tiny_patch4)."""
        return cls(
            image_size=32,
            patch_size=8,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
        )
    
    @classmethod
    def small(cls) -> 'ViTConfig':
        """Small configuration for experiments."""
        return cls(
            image_size=64,
            patch_size=8,
            embed_dim=128,
            num_layers=4,
            num_heads=4,
            mlp_ratio=2.0,
        )
