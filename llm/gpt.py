"""
GPT-like Language Model.

Implements:
- GPT model architecture (decoder-only transformer)
- Token and position embeddings
- Language modeling head
- Pretraining loop
- Text generation
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Any
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnn.tensor import Tensor
from nnn.optim import Adam

from .transformer import TransformerBlock, LayerNorm, LearnedPositionalEncoding
from .tokenizer import BaseTokenizer


class GPTConfig:
    """Configuration for GPT model."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 512,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = None,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            max_seq_len: Maximum sequence length
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feed-forward dimension (default: 4 * d_model)
            dropout: Dropout probability
            tie_weights: Whether to tie embedding and output weights
        """
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff or 4 * d_model
        self.dropout = dropout
        self.tie_weights = tie_weights
    
    @classmethod
    def gpt2_small(cls) -> 'GPTConfig':
        """GPT-2 Small configuration (117M parameters)."""
        return cls(
            vocab_size=50257,
            max_seq_len=1024,
            d_model=768,
            num_heads=12,
            num_layers=12,
        )
    
    @classmethod
    def gpt2_medium(cls) -> 'GPTConfig':
        """GPT-2 Medium configuration (345M parameters)."""
        return cls(
            vocab_size=50257,
            max_seq_len=1024,
            d_model=1024,
            num_heads=16,
            num_layers=24,
        )
    
    @classmethod
    def tiny(cls, vocab_size: int = 1000) -> 'GPTConfig':
        """Tiny configuration for testing."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=128,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256,
        )
    
    @classmethod
    def small(cls, vocab_size: int = 10000) -> 'GPTConfig':
        """Small configuration for experiments."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=256,
            d_model=256,
            num_heads=8,
            num_layers=6,
            d_ff=1024,
        )


class GPT:
    """
    GPT-like decoder-only transformer for language modeling.
    
    Architecture:
    - Token embeddings
    - Learned positional embeddings
    - N x TransformerBlock (with causal masking)
    - Final LayerNorm
    - Linear projection to vocabulary (optionally tied with embeddings)
    """
    
    def __init__(self, config: GPTConfig):
        self.config = config
        self.training = True
        
        # Token embedding
        self.token_embedding = Tensor(
            np.random.randn(config.vocab_size, config.d_model).astype(np.float32) * 0.02,
            requires_grad=True
        )
        
        # Position embedding (learned)
        self.position_embedding = Tensor(
            np.random.randn(config.max_seq_len, config.d_model).astype(np.float32) * 0.02,
            requires_grad=True
        )
        
        # Embedding dropout
        self.embed_dropout = config.dropout
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len
            )
            for _ in range(config.num_layers)
        ]
        
        # Final layer norm
        self.ln_f = LayerNorm(config.d_model)
        
        # Language model head
        if config.tie_weights:
            # Share weights with token embedding
            self.lm_head = None
        else:
            self.lm_head = Tensor(
                np.random.randn(config.d_model, config.vocab_size).astype(np.float32) * 0.02,
                requires_grad=True
            )
    
    def forward(self, input_ids: np.ndarray, targets: Optional[np.ndarray] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through GPT.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            targets: Target token IDs for computing loss (shifted input_ids)
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        batch_size, seq_len = input_ids.shape
        
        assert seq_len <= self.config.max_seq_len, \
            f"Sequence length {seq_len} exceeds maximum {self.config.max_seq_len}"
        
        # Get token embeddings
        tok_emb = self.token_embedding.data[input_ids]  # (batch, seq, d_model)
        
        # Get position embeddings
        positions = np.arange(seq_len)
        pos_emb = self.position_embedding.data[positions]  # (seq, d_model)
        pos_emb = pos_emb.reshape(1, seq_len, self.config.d_model)  # (1, seq, d_model)
        
        # Combine embeddings
        x_data = tok_emb + pos_emb
        
        # Apply embedding dropout
        if self.training and self.embed_dropout > 0:
            mask = (np.random.rand(*x_data.shape) > self.embed_dropout).astype(np.float32)
            scale = 1.0 / (1.0 - self.embed_dropout)
            x_data = x_data * mask * scale
        
        x = Tensor(x_data, requires_grad=True)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        if self.config.tie_weights:
            logits_data = x.data @ self.token_embedding.data.T
        else:
            logits_data = x.data @ self.lm_head.data
        
        logits = Tensor(logits_data, requires_grad=True)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self._compute_loss(logits, targets)
        
        return logits, loss
    
    def _compute_loss(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        """
        Compute cross-entropy loss for language modeling.
        
        Args:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs of shape (batch_size, seq_len)
        
        Returns:
            Scalar loss tensor
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for cross-entropy
        logits_flat = logits.data.reshape(-1, vocab_size)  # (batch * seq, vocab)
        targets_flat = targets.reshape(-1)  # (batch * seq,)
        
        # Stable softmax
        logits_max = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - logits_max)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-9)
        
        # Get probabilities of target tokens
        n_samples = len(targets_flat)
        target_probs = probs[np.arange(n_samples), targets_flat]
        
        # Compute negative log likelihood
        # Mask padding tokens (assuming padding is -100)
        mask = (targets_flat != -100).astype(np.float32)
        log_probs = np.log(target_probs + 1e-9)
        loss_value = -np.sum(log_probs * mask) / (np.sum(mask) + 1e-9)
        
        # Ensure loss is a scalar (0-dimensional array)
        loss_data = np.float32(loss_value)
        
        loss = Tensor(loss_data, requires_grad=True)
        
        # Setup backward pass
        def _backward(grad):
            # Gradient of cross-entropy with softmax
            grad_logits_flat = probs.copy()
            grad_logits_flat[np.arange(n_samples), targets_flat] -= 1
            grad_logits_flat = grad_logits_flat * mask.reshape(-1, 1) / (np.sum(mask) + 1e-9)
            grad_logits_flat = grad_logits_flat * grad
            
            # Reshape back
            grad_logits = grad_logits_flat.reshape(batch_size, seq_len, vocab_size)
            
            # Propagate gradient through logits
            logits.backward(grad_logits)
            
            return ()
        
        loss._backward_fn = _backward
        loss._children = ()
        
        return loss
    
    def __call__(self, input_ids: np.ndarray, targets: Optional[np.ndarray] = None) -> Tuple[Tensor, Optional[Tensor]]:
        return self.forward(input_ids, targets)
    
    def parameters(self) -> List[Tensor]:
        """Get all trainable parameters."""
        params = [self.token_embedding, self.position_embedding]
        
        for block in self.blocks:
            params.extend(block.parameters())
        
        params.extend(self.ln_f.parameters())
        
        if not self.config.tie_weights and self.lm_head is not None:
            params.append(self.lm_head)
        
        return params
    
    def num_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.data.size for p in self.parameters())
    
    def train(self):
        """Set model to training mode."""
        self.training = True
        for block in self.blocks:
            block.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.training = False
        for block in self.blocks:
            block.eval()
    
    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> np.ndarray:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling with this probability mass
        
        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        # Import sampling functions
        from .sampling import sample_token
        
        generated = input_ids.copy()
        
        for _ in range(max_new_tokens):
            # Get context (crop if needed)
            context = generated[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _ = self.forward(context)
            
            # Get logits for next token (last position)
            next_logits = logits.data[:, -1, :]  # (batch, vocab)
            
            # Sample next token
            next_token = sample_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Append to generated sequence
            generated = np.concatenate([generated, next_token.reshape(-1, 1)], axis=1)
        
        return generated


# =============================================================================
# Pretraining
# =============================================================================

class TextDataset:
    """Simple text dataset for language modeling."""
    
    def __init__(self, texts: List[str], tokenizer: BaseTokenizer, max_seq_len: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Tokenize all texts
        self.tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            self.tokens.extend(tokens)
        
        self.tokens = np.array(self.tokens, dtype=np.int64)
        print(f"Dataset: {len(self.tokens)} tokens")
    
    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.max_seq_len)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a training example (input_ids, target_ids)."""
        x = self.tokens[idx:idx + self.max_seq_len]
        y = self.tokens[idx + 1:idx + 1 + self.max_seq_len]
        return x, y


class DataLoader:
    """Simple data loader for batching."""
    
    def __init__(self, dataset: TextDataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, n, self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            
            batch_x = []
            batch_y = []
            
            for idx in batch_indices:
                x, y = self.dataset[idx]
                batch_x.append(x)
                batch_y.append(y)
            
            yield np.array(batch_x), np.array(batch_y)
    
    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def pretrain(
    model: GPT,
    train_texts: List[str],
    tokenizer: BaseTokenizer,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    max_seq_len: int = 128,
    log_interval: int = 10,
    eval_interval: int = 100,
    eval_texts: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    """
    Pretrain GPT model on text data.
    
    Args:
        model: GPT model to train
        train_texts: Training texts
        tokenizer: Tokenizer for encoding texts
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_seq_len: Maximum sequence length
        log_interval: Steps between logging
        eval_interval: Steps between evaluation
        eval_texts: Optional validation texts
    
    Returns:
        Dictionary with training history
    """
    print(f"Pretraining GPT model with {model.num_parameters():,} parameters")
    
    # Create dataset and dataloader
    train_dataset = TextDataset(train_texts, tokenizer, max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    
    eval_dataset = None
    if eval_texts:
        eval_dataset = TextDataset(eval_texts, tokenizer, max_seq_len)
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'eval_loss': [],
        'steps': [],
    }
    
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        start_time = time.time()
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits, loss = model(input_ids, target_ids)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (max norm = 1.0)
            max_norm = 1.0
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = np.sum(p.grad ** 2)
                    total_norm += param_norm
            total_norm = np.sqrt(total_norm)
            
            if total_norm > max_norm:
                clip_coef = max_norm / (total_norm + 1e-6)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad *= clip_coef
            
            # Update parameters
            optimizer.step()
            
            loss_val = float(loss.data.item())
            epoch_losses.append(loss_val)
            
            # Logging
            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} | Step {global_step} | "
                      f"Loss: {loss_val:.4f} | Time: {elapsed:.2f}s")
            
            # Evaluation
            if eval_dataset and global_step % eval_interval == 0 and global_step > 0:
                eval_loss = evaluate(model, eval_dataset, batch_size)
                history['eval_loss'].append(eval_loss)
                history['steps'].append(global_step)
                print(f"  Eval Loss: {eval_loss:.4f}")
            
            global_step += 1
        
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_loss)
        print(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")
    
    return history


def evaluate(model: GPT, dataset: TextDataset, batch_size: int = 32) -> float:
    """Evaluate model on dataset."""
    model.eval()
    
    loader = DataLoader(dataset, batch_size, shuffle=False)
    losses = []
    
    for input_ids, target_ids in loader:
        logits, loss = model(input_ids, target_ids)
        losses.append(float(loss.data.item()))
    
    return np.mean(losses)


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model: GPT) -> Dict[str, int]:
    """Count parameters by component."""
    counts = {}
    
    counts['token_embedding'] = model.token_embedding.data.size
    counts['position_embedding'] = model.position_embedding.data.size
    
    block_params = sum(sum(p.data.size for p in block.parameters()) for block in model.blocks)
    counts['transformer_blocks'] = block_params
    
    counts['final_layernorm'] = sum(p.data.size for p in model.ln_f.parameters())
    
    if not model.config.tie_weights and model.lm_head is not None:
        counts['lm_head'] = model.lm_head.data.size
    else:
        counts['lm_head'] = 0
    
    counts['total'] = sum(counts.values())
    
    return counts
