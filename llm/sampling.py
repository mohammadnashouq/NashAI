"""
Sampling strategies for text generation.

Implements:
- Greedy sampling (argmax)
- Temperature scaling
- Top-k sampling
- Nucleus (Top-p) sampling
- Combined sampling strategies
- Beam search (basic)
"""

import numpy as np
from typing import Optional, List, Tuple


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Apply softmax with temperature scaling.
    
    Args:
        logits: Raw logits of shape (..., vocab_size)
        temperature: Temperature for scaling (higher = more random)
    
    Returns:
        Probability distribution
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Stable softmax
    max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(scaled_logits - max_logits)
    probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-9)
    
    return probs


def greedy_sample(logits: np.ndarray) -> np.ndarray:
    """
    Greedy sampling - select the most probable token.
    
    Args:
        logits: Logits of shape (batch_size, vocab_size)
    
    Returns:
        Selected token IDs of shape (batch_size,)
    """
    return np.argmax(logits, axis=-1)


def temperature_sample(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Sample from temperature-scaled probability distribution.
    
    Args:
        logits: Logits of shape (batch_size, vocab_size)
        temperature: Temperature for scaling
            - temperature < 1: More deterministic (sharper distribution)
            - temperature = 1: Standard sampling
            - temperature > 1: More random (flatter distribution)
    
    Returns:
        Selected token IDs of shape (batch_size,)
    """
    probs = softmax(logits, temperature)
    
    batch_size = logits.shape[0]
    samples = np.zeros(batch_size, dtype=np.int64)
    
    for i in range(batch_size):
        samples[i] = np.random.choice(len(probs[i]), p=probs[i])
    
    return samples


def top_k_sample(logits: np.ndarray, k: int, temperature: float = 1.0) -> np.ndarray:
    """
    Top-k sampling - sample only from the k most probable tokens.
    
    Args:
        logits: Logits of shape (batch_size, vocab_size)
        k: Number of top tokens to consider
        temperature: Temperature for scaling
    
    Returns:
        Selected token IDs of shape (batch_size,)
    
    Algorithm:
    1. Find the k largest logits
    2. Set all other logits to -inf
    3. Apply softmax and sample
    """
    batch_size, vocab_size = logits.shape
    
    # Clamp k to vocab size
    k = min(k, vocab_size)
    
    # Apply temperature first
    scaled_logits = logits / temperature
    
    # For each sample in batch, keep only top-k
    filtered_logits = np.full_like(scaled_logits, -np.inf)
    
    for i in range(batch_size):
        # Get indices of top-k logits
        top_k_indices = np.argpartition(scaled_logits[i], -k)[-k:]
        filtered_logits[i, top_k_indices] = scaled_logits[i, top_k_indices]
    
    # Apply softmax
    probs = softmax(filtered_logits, temperature=1.0)  # Already scaled
    
    # Sample
    samples = np.zeros(batch_size, dtype=np.int64)
    for i in range(batch_size):
        samples[i] = np.random.choice(vocab_size, p=probs[i])
    
    return samples


def top_p_sample(logits: np.ndarray, p: float, temperature: float = 1.0) -> np.ndarray:
    """
    Nucleus (Top-p) sampling - sample from tokens whose cumulative probability >= p.
    
    Args:
        logits: Logits of shape (batch_size, vocab_size)
        p: Probability mass threshold (e.g., 0.9 means top 90% probability mass)
        temperature: Temperature for scaling
    
    Returns:
        Selected token IDs of shape (batch_size,)
    
    Algorithm:
    1. Sort tokens by probability
    2. Find smallest set of tokens whose cumulative probability >= p
    3. Sample from this set
    
    Advantages over top-k:
    - Adapts dynamically to probability distribution
    - Works well when distribution is sharp (few tokens) or flat (many tokens)
    """
    batch_size, vocab_size = logits.shape
    
    # Apply temperature and get probabilities
    probs = softmax(logits, temperature)
    
    samples = np.zeros(batch_size, dtype=np.int64)
    
    for i in range(batch_size):
        # Sort probabilities in descending order
        sorted_indices = np.argsort(probs[i])[::-1]
        sorted_probs = probs[i, sorted_indices]
        
        # Find cumulative probabilities
        cumsum_probs = np.cumsum(sorted_probs)
        
        # Find cutoff index (smallest set with cumsum >= p)
        cutoff_idx = np.searchsorted(cumsum_probs, p) + 1
        cutoff_idx = min(cutoff_idx, vocab_size)
        
        # Get tokens in nucleus
        nucleus_indices = sorted_indices[:cutoff_idx]
        nucleus_probs = sorted_probs[:cutoff_idx]
        
        # Renormalize probabilities
        nucleus_probs = nucleus_probs / (np.sum(nucleus_probs) + 1e-9)
        
        # Sample from nucleus
        chosen_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
        samples[i] = nucleus_indices[chosen_idx]
    
    return samples


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> np.ndarray:
    """
    Combined sampling with multiple strategies.
    
    Args:
        logits: Logits of shape (batch_size, vocab_size)
        temperature: Temperature for scaling
        top_k: If set, use top-k filtering
        top_p: If set, use nucleus sampling
    
    Returns:
        Selected token IDs of shape (batch_size,)
    
    Priority:
    1. If temperature == 0, use greedy
    2. Apply top-k if specified
    3. Apply top-p if specified
    4. Otherwise, use temperature sampling
    """
    if temperature == 0 or temperature < 1e-6:
        return greedy_sample(logits)
    
    # Apply top-k filtering first (if specified)
    if top_k is not None and top_k > 0:
        batch_size, vocab_size = logits.shape
        
        # Clamp top_k to vocab size
        effective_k = min(top_k, vocab_size)
        
        # Filter to top-k
        filtered_logits = np.full_like(logits, -np.inf)
        for i in range(batch_size):
            top_k_indices = np.argpartition(logits[i], -effective_k)[-effective_k:]
            filtered_logits[i, top_k_indices] = logits[i, top_k_indices]
        
        logits = filtered_logits
    
    # Apply top-p (nucleus) filtering (if specified)
    if top_p is not None and 0 < top_p < 1.0:
        return top_p_sample(logits, top_p, temperature)
    
    # Standard temperature sampling
    return temperature_sample(logits, temperature)


def beam_search(
    logits_fn,
    input_ids: np.ndarray,
    max_length: int,
    beam_width: int = 5,
    length_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Beam search for text generation.
    
    Args:
        logits_fn: Function that takes input_ids and returns logits
        input_ids: Starting token IDs of shape (1, seq_len)
        max_length: Maximum sequence length
        beam_width: Number of beams
        length_penalty: Penalty for sequence length (>1 favors longer, <1 favors shorter)
        eos_token_id: End of sequence token ID
    
    Returns:
        best_sequence: Best sequence of token IDs
        best_score: Score of best sequence
    
    Algorithm:
    1. Start with input sequence
    2. At each step, keep top beam_width candidates
    3. Score = log probability / length^length_penalty
    4. Stop when EOS or max_length reached
    """
    # Initialize beams: [(sequence, score)]
    beams = [(input_ids.copy(), 0.0)]
    
    completed = []
    
    for step in range(max_length - input_ids.shape[1]):
        all_candidates = []
        
        for seq, score in beams:
            # Check if already complete
            if eos_token_id is not None and seq[0, -1] == eos_token_id:
                completed.append((seq, score))
                continue
            
            # Get next token logits
            logits = logits_fn(seq)  # (1, vocab_size)
            log_probs = np.log(softmax(logits, temperature=1.0) + 1e-9)
            
            # Get top-k candidates
            top_indices = np.argsort(log_probs[0])[-beam_width:]
            
            for idx in top_indices:
                new_seq = np.concatenate([seq, [[idx]]], axis=1)
                new_score = score + log_probs[0, idx]
                
                # Apply length penalty
                length = new_seq.shape[1]
                penalized_score = new_score / (length ** length_penalty)
                
                all_candidates.append((new_seq, penalized_score, new_score))
        
        if not all_candidates:
            break
        
        # Keep top beam_width candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = [(seq, raw_score) for seq, _, raw_score in all_candidates[:beam_width]]
    
    # Add remaining beams to completed
    completed.extend(beams)
    
    if not completed:
        return input_ids, 0.0
    
    # Return best sequence
    best = max(completed, key=lambda x: x[1] / (x[0].shape[1] ** length_penalty))
    return best[0], best[1]


def repetition_penalty(
    logits: np.ndarray,
    input_ids: np.ndarray,
    penalty: float = 1.2
) -> np.ndarray:
    """
    Apply repetition penalty to logits.
    
    Reduces probability of tokens that have already appeared.
    
    Args:
        logits: Logits of shape (batch_size, vocab_size)
        input_ids: Previous token IDs of shape (batch_size, seq_len)
        penalty: Penalty factor (> 1.0 discourages repetition)
    
    Returns:
        Penalized logits
    """
    batch_size = logits.shape[0]
    penalized_logits = logits.copy()
    
    for i in range(batch_size):
        # Get unique tokens in context
        unique_tokens = np.unique(input_ids[i])
        
        for token in unique_tokens:
            # Apply penalty
            if penalized_logits[i, token] > 0:
                penalized_logits[i, token] = penalized_logits[i, token] / penalty
            else:
                penalized_logits[i, token] = penalized_logits[i, token] * penalty
    
    return penalized_logits


def typical_sampling(logits: np.ndarray, mass: float = 0.9, temperature: float = 1.0) -> np.ndarray:
    """
    Typical sampling - sample tokens with typical information content.
    
    Based on "Typical Decoding for Natural Language Generation" (Meister et al., 2022)
    
    Args:
        logits: Logits of shape (batch_size, vocab_size)
        mass: Target probability mass
        temperature: Temperature for scaling
    
    Returns:
        Selected token IDs of shape (batch_size,)
    
    Algorithm:
    1. Compute entropy of distribution
    2. Compute "surprise" (negative log prob) of each token
    3. Keep tokens whose surprise is close to entropy
    4. Sample from this "typical" set
    """
    batch_size, vocab_size = logits.shape
    
    # Get probabilities
    probs = softmax(logits, temperature)
    
    samples = np.zeros(batch_size, dtype=np.int64)
    
    for i in range(batch_size):
        # Compute entropy
        entropy = -np.sum(probs[i] * np.log(probs[i] + 1e-9))
        
        # Compute surprise for each token
        surprise = -np.log(probs[i] + 1e-9)
        
        # Compute deviation from entropy
        deviation = np.abs(surprise - entropy)
        
        # Sort by deviation
        sorted_indices = np.argsort(deviation)
        sorted_probs = probs[i, sorted_indices]
        
        # Find typical set (cumsum >= mass)
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, mass) + 1
        cutoff = min(cutoff, vocab_size)
        
        # Get typical tokens
        typical_indices = sorted_indices[:cutoff]
        typical_probs = probs[i, typical_indices]
        
        # Renormalize and sample
        typical_probs = typical_probs / (np.sum(typical_probs) + 1e-9)
        chosen = np.random.choice(len(typical_probs), p=typical_probs)
        samples[i] = typical_indices[chosen]
    
    return samples


class SamplingConfig:
    """Configuration for sampling parameters."""
    
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        typical_p: Optional[float] = None,
    ):
        """
        Args:
            temperature: Temperature for scaling (0 = greedy)
            top_k: Top-k filtering (None = disabled)
            top_p: Nucleus sampling threshold (None = disabled)
            repetition_penalty: Penalty for repeated tokens (1.0 = disabled)
            typical_p: Typical sampling mass (None = disabled)
        """
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.typical_p = typical_p
    
    @classmethod
    def greedy(cls) -> 'SamplingConfig':
        """Greedy decoding."""
        return cls(temperature=0)
    
    @classmethod
    def default(cls) -> 'SamplingConfig':
        """Default sampling with temperature."""
        return cls(temperature=0.8, top_p=0.9)
    
    @classmethod
    def creative(cls) -> 'SamplingConfig':
        """More creative/random sampling."""
        return cls(temperature=1.2, top_p=0.95)
    
    @classmethod
    def deterministic(cls) -> 'SamplingConfig':
        """More deterministic sampling."""
        return cls(temperature=0.7, top_k=50)


def sample_with_config(
    logits: np.ndarray,
    config: SamplingConfig,
    input_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample tokens using a sampling configuration.
    
    Args:
        logits: Logits of shape (batch_size, vocab_size)
        config: Sampling configuration
        input_ids: Previous tokens for repetition penalty
    
    Returns:
        Selected token IDs
    """
    # Apply repetition penalty if needed
    if config.repetition_penalty > 1.0 and input_ids is not None:
        logits = repetition_penalty(logits, input_ids, config.repetition_penalty)
    
    # Typical sampling
    if config.typical_p is not None:
        return typical_sampling(logits, config.typical_p, config.temperature)
    
    # Standard sampling with top-k and top-p
    return sample_token(
        logits,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
    )
