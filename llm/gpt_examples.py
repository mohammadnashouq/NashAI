"""
Examples for GPT model and pretraining.

Demonstrates:
- GPT model configuration
- Forward pass
- Text generation
- Pretraining loop
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnn.tensor import Tensor


def example_gpt_config():
    """Example 1: GPT model configurations."""
    print("=" * 60)
    print("Example 1: GPT Model Configurations")
    print("=" * 60)
    
    from llm.gpt import GPTConfig
    
    # Different configurations
    configs = {
        'Tiny': GPTConfig.tiny(vocab_size=1000),
        'Small': GPTConfig.small(vocab_size=10000),
        'GPT-2 Small': GPTConfig.gpt2_small(),
        'GPT-2 Medium': GPTConfig.gpt2_medium(),
    }
    
    print(f"{'Config':<15} {'Vocab':>8} {'d_model':>8} {'Heads':>6} {'Layers':>7} {'d_ff':>8} {'Max Seq':>8}")
    print("-" * 65)
    
    for name, cfg in configs.items():
        print(f"{name:<15} {cfg.vocab_size:>8} {cfg.d_model:>8} {cfg.num_heads:>6} "
              f"{cfg.num_layers:>7} {cfg.d_ff:>8} {cfg.max_seq_len:>8}")
    
    print("\nNote: GPT-2 configurations are shown for reference.")
    print("For training, use Tiny or Small configurations.")
    print()


def example_gpt_model():
    """Example 2: Create and inspect GPT model."""
    print("=" * 60)
    print("Example 2: GPT Model Creation")
    print("=" * 60)
    
    from llm.gpt import GPT, GPTConfig, count_parameters
    
    # Create tiny model
    config = GPTConfig.tiny(vocab_size=500)
    model = GPT(config)
    
    print(f"Model Configuration:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  tie_weights: {config.tie_weights}")
    
    # Count parameters
    param_counts = count_parameters(model)
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    print(f"\nTotal parameters: {model.num_parameters():,}")
    print()


def example_forward_pass():
    """Example 3: GPT forward pass."""
    print("=" * 60)
    print("Example 3: GPT Forward Pass")
    print("=" * 60)
    
    from llm.gpt import GPT, GPTConfig
    
    # Create model
    config = GPTConfig.tiny(vocab_size=100)
    model = GPT(config)
    model.eval()
    
    # Create input
    batch_size = 2
    seq_len = 16
    input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Input sample: {input_ids[0, :10]}...")
    
    # Forward pass without targets
    logits, loss = model(input_ids)
    
    print(f"\nOutput (no targets):")
    print(f"  Logits shape: {logits.shape}")  # (batch, seq, vocab)
    print(f"  Loss: {loss}")  # None when no targets
    
    # Forward pass with targets (for language modeling)
    target_ids = np.roll(input_ids, -1, axis=1)  # Shift left
    target_ids[:, -1] = 0  # Last token targets padding
    
    logits, loss = model(input_ids, target_ids)
    
    print(f"\nOutput (with targets):")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.data.item():.4f}")
    print()


def example_next_token_prediction():
    """Example 4: Next token prediction."""
    print("=" * 60)
    print("Example 4: Next Token Prediction")
    print("=" * 60)
    
    from llm.gpt import GPT, GPTConfig
    from llm.sampling import softmax, greedy_sample, top_k_sample
    
    # Create model
    vocab_size = 50
    config = GPTConfig.tiny(vocab_size=vocab_size)
    model = GPT(config)
    model.eval()
    
    # Input sequence
    input_ids = np.array([[1, 5, 10, 15, 20]])  # (1, 5)
    
    print(f"Input sequence: {input_ids[0]}")
    
    # Get logits
    logits, _ = model(input_ids)
    
    # Get logits for next token (last position)
    next_logits = logits.data[:, -1, :]  # (1, vocab)
    
    # Compute probabilities
    probs = softmax(next_logits, temperature=1.0)
    
    print(f"\nTop 5 predicted tokens:")
    top5_indices = np.argsort(probs[0])[-5:][::-1]
    for idx in top5_indices:
        print(f"  Token {idx}: {probs[0, idx]:.4f}")
    
    # Sample next token
    greedy_next = greedy_sample(next_logits)
    topk_next = top_k_sample(next_logits, k=5)
    
    print(f"\nGreedy next token: {greedy_next[0]}")
    print(f"Top-k (k=5) sample: {topk_next[0]}")
    print()


def example_text_generation():
    """Example 5: Text generation (token-level)."""
    print("=" * 60)
    print("Example 5: Text Generation")
    print("=" * 60)
    
    from llm.gpt import GPT, GPTConfig
    from llm.sampling import sample_token
    
    # Create model
    vocab_size = 100
    config = GPTConfig.tiny(vocab_size=vocab_size)
    model = GPT(config)
    model.eval()
    
    # Start with a prompt
    prompt = np.array([[5, 10, 15]])  # Starting tokens
    
    print(f"Prompt: {prompt[0]}")
    
    # Generate tokens
    generated = prompt.copy()
    num_tokens = 20
    
    for i in range(num_tokens):
        # Forward pass
        logits, _ = model(generated[:, -config.max_seq_len:])
        
        # Get next token logits
        next_logits = logits.data[:, -1, :]
        
        # Sample
        next_token = sample_token(next_logits, temperature=0.8, top_k=10)
        
        # Append
        generated = np.concatenate([generated, next_token.reshape(1, 1)], axis=1)
    
    print(f"Generated sequence ({len(generated[0])} tokens):")
    print(f"  {generated[0]}")
    print()


def example_simple_training():
    """Example 6: Simple training loop."""
    print("=" * 60)
    print("Example 6: Simple Training Loop")
    print("=" * 60)
    
    from llm.gpt import GPT, GPTConfig
    from nnn.optim import Adam
    
    # Create small model
    vocab_size = 50
    config = GPTConfig(
        vocab_size=vocab_size,
        max_seq_len=32,
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=128,
        dropout=0.0,
    )
    model = GPT(config)
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    # Create dummy training data
    batch_size = 4
    seq_len = 16
    
    print(f"\nTraining for 5 steps...")
    
    for step in range(5):
        # Generate random batch
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        target_ids = np.roll(input_ids, -1, axis=1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, loss = model(input_ids, target_ids)
        
        # Backward pass
        loss.backward()
        
        # Update
        optimizer.step()
        
        print(f"  Step {step + 1}: loss = {loss.data.item():.4f}")
    
    print()


def example_with_tokenizer():
    """Example 7: GPT with tokenizer."""
    print("=" * 60)
    print("Example 7: GPT with Tokenizer")
    print("=" * 60)
    
    from llm.gpt import GPT, GPTConfig
    from llm.tokenizer import CharTokenizer
    from llm.sampling import sample_token
    
    # Training texts
    texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a bird flew over the tree",
    ]
    
    # Create and fit tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(texts)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=128,
    )
    model = GPT(config)
    model.eval()
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Encode prompt
    prompt_text = "the cat"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = np.array([prompt_ids])
    
    print(f"\nPrompt: '{prompt_text}'")
    print(f"Encoded: {input_ids[0]}")
    
    # Generate
    for i in range(20):
        logits, _ = model(input_ids[:, -config.max_seq_len:])
        next_logits = logits.data[:, -1, :]
        next_token = sample_token(next_logits, temperature=0.8)
        input_ids = np.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)
    
    # Decode
    generated_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    print(f"Generated: '{generated_text}'")
    print()


def example_pretrain_function():
    """Example 8: Using the pretrain function."""
    print("=" * 60)
    print("Example 8: Pretraining Function")
    print("=" * 60)
    
    from llm.gpt import GPT, GPTConfig, pretrain
    from llm.tokenizer import CharTokenizer
    
    # Training texts
    texts = [
        "hello world how are you today",
        "the quick brown fox jumps over the lazy dog",
        "machine learning is very interesting",
        "neural networks process information",
    ] * 10  # Repeat for more data
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(texts)
    
    # Create small model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=32,
        d_model=32,
        num_heads=2,
        num_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    model = GPT(config)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Training on {len(texts)} text samples")
    
    # Pretrain (just 2 epochs for demo)
    history = pretrain(
        model=model,
        train_texts=texts,
        tokenizer=tokenizer,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        max_seq_len=32,
        log_interval=5,
    )
    
    print(f"\nTraining history:")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print()


def example_text_dataset():
    """Example 9: TextDataset and DataLoader."""
    print("=" * 60)
    print("Example 9: TextDataset and DataLoader")
    print("=" * 60)
    
    from llm.gpt import TextDataset, DataLoader
    from llm.tokenizer import CharTokenizer
    
    # Create tokenizer
    texts = ["hello world this is a test"]
    tokenizer = CharTokenizer()
    tokenizer.fit(texts)
    
    # Create dataset
    dataset = TextDataset(texts, tokenizer, max_seq_len=8)
    
    print(f"Total tokens: {len(dataset.tokens)}")
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"\nSample 0:")
        print(f"  Input (x): {x}")
        print(f"  Target (y): {y}")
        print(f"  Note: y is x shifted by 1 (next token prediction)")
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(f"\nDataLoader batches: {len(loader)}")
    
    # Iterate through one batch
    for batch_x, batch_y in loader:
        print(f"Batch shapes: x={batch_x.shape}, y={batch_y.shape}")
        break
    print()


def example_parameter_breakdown():
    """Example 10: Detailed parameter breakdown."""
    print("=" * 60)
    print("Example 10: Parameter Breakdown")
    print("=" * 60)
    
    from llm.gpt import GPT, GPTConfig, count_parameters
    
    # Create model with known dimensions
    config = GPTConfig(
        vocab_size=1000,
        max_seq_len=128,
        d_model=64,
        num_heads=4,
        num_layers=4,
        d_ff=256,
        tie_weights=True,
    )
    model = GPT(config)
    
    counts = count_parameters(model)
    
    print(f"Parameter breakdown:")
    print(f"  Token embedding: {counts['token_embedding']:,}")
    print(f"    = vocab_size × d_model = {config.vocab_size} × {config.d_model}")
    print(f"  Position embedding: {counts['position_embedding']:,}")
    print(f"    = max_seq_len × d_model = {config.max_seq_len} × {config.d_model}")
    print(f"  Transformer blocks: {counts['transformer_blocks']:,}")
    print(f"    = {config.num_layers} layers × params_per_layer")
    print(f"  Final LayerNorm: {counts['final_layernorm']:,}")
    print(f"  LM head: {counts['lm_head']:,}")
    print(f"    (0 because weights tied with token embedding)")
    print(f"\n  Total: {counts['total']:,}")
    print()


if __name__ == "__main__":
    example_gpt_config()
    example_gpt_model()
    example_forward_pass()
    example_next_token_prediction()
    example_text_generation()
    example_simple_training()
    example_with_tokenizer()
    example_pretrain_function()
    example_text_dataset()
    example_parameter_breakdown()
    
    print("All GPT examples completed!")
