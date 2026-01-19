"""
LLM (Large Language Model) module.

A complete implementation of transformer-based language models from scratch.

Components:
- Tokenizers: BPE, WordPiece, Character-level
- Transformer: Positional encoding, LayerNorm, Self-attention, Transformer blocks
- GPT: GPT-like decoder model with pretraining loop
- Sampling: Top-k, Nucleus (top-p), Beam search, and more

Example usage:
    from llm import GPT, GPTConfig, BPETokenizer
    
    # Create tokenizer
    tokenizer = BPETokenizer(vocab_size=5000)
    tokenizer.fit(texts)
    
    # Create model
    config = GPTConfig.tiny(vocab_size=tokenizer.vocab_size)
    model = GPT(config)
    
    # Train
    from llm import pretrain
    pretrain(model, texts, tokenizer, epochs=10)
    
    # Generate
    input_ids = tokenizer.encode("Once upon a time")
    output = model.generate(input_ids, max_new_tokens=50)
    print(tokenizer.decode(output))
"""

# Tokenizers
from .tokenizer import (
    BaseTokenizer,
    CharTokenizer,
    BPETokenizer,
    WordPieceTokenizer,
    create_tokenizer,
)

# Transformer components
from .transformer import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    LayerNorm,
    FeedForward,
    CausalSelfAttention,
    TransformerBlock,
    create_causal_mask,
    create_padding_mask,
)

# GPT model
from .gpt import (
    GPT,
    GPTConfig,
    TextDataset,
    DataLoader,
    pretrain,
    evaluate,
    count_parameters,
)

# Sampling strategies
from .sampling import (
    softmax,
    greedy_sample,
    temperature_sample,
    top_k_sample,
    top_p_sample,
    sample_token,
    beam_search,
    repetition_penalty,
    typical_sampling,
    SamplingConfig,
    sample_with_config,
)

__all__ = [
    # Tokenizers
    'BaseTokenizer',
    'CharTokenizer',
    'BPETokenizer',
    'WordPieceTokenizer',
    'create_tokenizer',
    # Transformer
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEncoding',
    'LayerNorm',
    'FeedForward',
    'CausalSelfAttention',
    'TransformerBlock',
    'create_causal_mask',
    'create_padding_mask',
    # GPT
    'GPT',
    'GPTConfig',
    'TextDataset',
    'DataLoader',
    'pretrain',
    'evaluate',
    'count_parameters',
    # Sampling
    'softmax',
    'greedy_sample',
    'temperature_sample',
    'top_k_sample',
    'top_p_sample',
    'sample_token',
    'beam_search',
    'repetition_penalty',
    'typical_sampling',
    'SamplingConfig',
    'sample_with_config',
]
