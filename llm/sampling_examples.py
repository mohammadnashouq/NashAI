"""
Examples for sampling strategies.

Demonstrates:
- Greedy sampling
- Temperature scaling
- Top-k sampling
- Nucleus (Top-p) sampling
- Typical sampling
- Combined strategies
- Repetition penalty
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_softmax_temperature():
    """Example 1: Temperature scaling effect."""
    print("=" * 60)
    print("Example 1: Temperature Scaling")
    print("=" * 60)
    
    from llm.sampling import softmax
    
    # Sample logits
    logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    
    print("Logits:", logits[0])
    print("\nProbabilities at different temperatures:")
    
    for temp in [0.5, 1.0, 2.0, 5.0]:
        probs = softmax(logits, temperature=temp)
        print(f"  T={temp}: {np.round(probs[0], 3)}")
    
    print("\nObservation:")
    print("  - Lower temperature -> sharper distribution (more deterministic)")
    print("  - Higher temperature -> flatter distribution (more random)")
    print()


def example_greedy_sampling():
    """Example 2: Greedy sampling."""
    print("=" * 60)
    print("Example 2: Greedy Sampling")
    print("=" * 60)
    
    from llm.sampling import greedy_sample, softmax
    
    # Sample logits
    logits = np.array([
        [1.0, 3.0, 2.0, 0.5, 4.0],
        [5.0, 1.0, 2.0, 3.0, 0.5],
    ])
    
    print("Logits:")
    print(logits)
    
    probs = softmax(logits, temperature=1.0)
    print("\nProbabilities:")
    print(np.round(probs, 3))
    
    # Greedy selection
    selected = greedy_sample(logits)
    
    print(f"\nGreedy selected tokens: {selected}")
    print(f"  Sample 0: token {selected[0]} (highest logit = {logits[0, selected[0]]})")
    print(f"  Sample 1: token {selected[1]} (highest logit = {logits[1, selected[1]]})")
    print()


def example_temperature_sampling():
    """Example 3: Temperature-based sampling."""
    print("=" * 60)
    print("Example 3: Temperature Sampling")
    print("=" * 60)
    
    from llm.sampling import temperature_sample
    
    np.random.seed(42)
    
    # Logits with clear winner
    logits = np.array([[1.0, 2.0, 5.0, 1.5, 0.5]])  # Token 2 has highest
    
    print("Logits:", logits[0])
    print(f"Highest logit: token 2 with value {logits[0, 2]}")
    
    # Sample multiple times at different temperatures
    for temp in [0.5, 1.0, 2.0]:
        samples = [temperature_sample(logits, temperature=temp)[0] for _ in range(100)]
        unique, counts = np.unique(samples, return_counts=True)
        
        print(f"\nT={temp} (100 samples):")
        for token, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"  Token {token}: {count}%")
    print()


def example_top_k_sampling():
    """Example 4: Top-k sampling."""
    print("=" * 60)
    print("Example 4: Top-K Sampling")
    print("=" * 60)
    
    from llm.sampling import top_k_sample, softmax
    
    np.random.seed(42)
    
    # Logits with spread distribution
    logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1]])
    
    print("Logits:", logits[0])
    print("Probabilities:", np.round(softmax(logits)[0], 3))
    
    # Sample with different k values
    for k in [2, 3, 5]:
        samples = [top_k_sample(logits, k=k, temperature=1.0)[0] for _ in range(100)]
        unique, counts = np.unique(samples, return_counts=True)
        
        print(f"\nTop-k (k={k}) - 100 samples:")
        print(f"  Only tokens 0-{k-1} are eligible")
        for token, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"  Token {token}: {count}%")
    print()


def example_nucleus_sampling():
    """Example 5: Nucleus (Top-p) sampling."""
    print("=" * 60)
    print("Example 5: Nucleus (Top-P) Sampling")
    print("=" * 60)
    
    from llm.sampling import top_p_sample, softmax
    
    np.random.seed(42)
    
    # Logits with varied distribution
    logits = np.array([[5.0, 4.0, 3.0, 1.0, 0.5, 0.2]])
    
    probs = softmax(logits)[0]
    cumsum = np.cumsum(np.sort(probs)[::-1])
    
    print("Logits:", logits[0])
    print("Probabilities:", np.round(probs, 3))
    print("Sorted cumulative sum:", np.round(cumsum, 3))
    
    # Sample with different p values
    for p in [0.5, 0.9, 0.95]:
        samples = [top_p_sample(logits, p=p, temperature=1.0)[0] for _ in range(100)]
        unique, counts = np.unique(samples, return_counts=True)
        
        # Find nucleus size
        nucleus_size = np.searchsorted(cumsum, p) + 1
        
        print(f"\nTop-p (p={p}) - Nucleus size ~ {nucleus_size} tokens:")
        for token, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"  Token {token}: {count}%")
    print()


def example_combined_sampling():
    """Example 6: Combined top-k and top-p."""
    print("=" * 60)
    print("Example 6: Combined Sampling Strategies")
    print("=" * 60)
    
    from llm.sampling import sample_token
    
    np.random.seed(42)
    
    logits = np.array([[5.0, 4.5, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2]])
    
    print("Logits:", logits[0])
    
    # Different combinations
    configs = [
        ("Greedy (T=0)", {"temperature": 0}),
        ("T=1.0 only", {"temperature": 1.0}),
        ("Top-k=3, T=1.0", {"temperature": 1.0, "top_k": 3}),
        ("Top-p=0.9, T=1.0", {"temperature": 1.0, "top_p": 0.9}),
        ("Top-k=5, Top-p=0.9, T=0.8", {"temperature": 0.8, "top_k": 5, "top_p": 0.9}),
    ]
    
    for name, params in configs:
        samples = [sample_token(logits, **params)[0] for _ in range(50)]
        unique, counts = np.unique(samples, return_counts=True)
        
        print(f"\n{name}:")
        for token, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:5]:
            print(f"  Token {token}: {count*2}%")
    print()


def example_repetition_penalty():
    """Example 7: Repetition penalty."""
    print("=" * 60)
    print("Example 7: Repetition Penalty")
    print("=" * 60)
    
    from llm.sampling import repetition_penalty, softmax
    
    # Logits and previous tokens
    logits = np.array([[3.0, 3.0, 3.0, 3.0, 3.0]])  # Uniform
    previous_tokens = np.array([[0, 1, 2]])  # Tokens 0, 1, 2 already appeared
    
    print("Original logits:", logits[0])
    print("Previous tokens:", previous_tokens[0])
    
    for penalty in [1.0, 1.2, 1.5, 2.0]:
        penalized = repetition_penalty(logits, previous_tokens, penalty=penalty)
        probs = softmax(penalized)[0]
        
        print(f"\nPenalty = {penalty}:")
        print(f"  Penalized logits: {np.round(penalized[0], 2)}")
        print(f"  Probabilities: {np.round(probs, 3)}")
        print(f"  Note: Tokens 0,1,2 have lower prob with higher penalty")
    print()


def example_typical_sampling():
    """Example 8: Typical sampling."""
    print("=" * 60)
    print("Example 8: Typical Sampling")
    print("=" * 60)
    
    from llm.sampling import typical_sampling, softmax
    
    np.random.seed(42)
    
    # Logits with varied distribution
    logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.5]])
    
    probs = softmax(logits)[0]
    
    print("Logits:", logits[0])
    print("Probabilities:", np.round(probs, 3))
    
    # Compute entropy
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    print(f"Entropy: {entropy:.3f}")
    
    # Compute surprise for each token
    surprise = -np.log(probs + 1e-9)
    print(f"Surprise (per token): {np.round(surprise, 2)}")
    print(f"Tokens close to entropy are 'typical'")
    
    # Sample
    samples = [typical_sampling(logits, mass=0.9)[0] for _ in range(100)]
    unique, counts = np.unique(samples, return_counts=True)
    
    print(f"\nTypical sampling (mass=0.9) - 100 samples:")
    for token, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  Token {token}: {count}%")
    print()


def example_sampling_config():
    """Example 9: SamplingConfig presets."""
    print("=" * 60)
    print("Example 9: Sampling Configuration Presets")
    print("=" * 60)
    
    from llm.sampling import SamplingConfig, sample_with_config
    
    np.random.seed(42)
    
    logits = np.array([[5.0, 4.0, 3.5, 2.0, 1.0]])
    
    print("Logits:", logits[0])
    
    # Different presets
    presets = {
        'greedy': SamplingConfig.greedy(),
        'default': SamplingConfig.default(),
        'creative': SamplingConfig.creative(),
        'deterministic': SamplingConfig.deterministic(),
    }
    
    for name, config in presets.items():
        print(f"\n{name.upper()} preset:")
        print(f"  temperature={config.temperature}, "
              f"top_k={config.top_k}, top_p={config.top_p}")
        
        samples = [sample_with_config(logits, config)[0] for _ in range(50)]
        unique, counts = np.unique(samples, return_counts=True)
        
        for token, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:3]:
            print(f"    Token {token}: {count*2}%")
    print()


def example_generation_quality():
    """Example 10: Compare generation quality."""
    print("=" * 60)
    print("Example 10: Generation Quality Comparison")
    print("=" * 60)
    
    from llm.sampling import sample_token
    
    np.random.seed(42)
    
    # Simulate generating a sequence
    vocab_size = 10
    seq_length = 20
    
    def generate_sequence(strategy_fn):
        """Generate a sequence with given sampling strategy."""
        seq = []
        for _ in range(seq_length):
            # Simulate logits (some tokens more likely)
            logits = np.random.randn(1, vocab_size).astype(np.float32)
            logits[0, 0] += 2  # Token 0 is common
            logits[0, 1] += 1.5  # Token 1 is somewhat common
            
            token = strategy_fn(logits)
            seq.append(token[0])
        return seq
    
    strategies = {
        'Greedy': lambda l: sample_token(l, temperature=0),
        'T=0.5': lambda l: sample_token(l, temperature=0.5),
        'T=1.0': lambda l: sample_token(l, temperature=1.0),
        'Top-k=3': lambda l: sample_token(l, temperature=1.0, top_k=3),
        'Top-p=0.9': lambda l: sample_token(l, temperature=1.0, top_p=0.9),
    }
    
    print("Generated sequences (20 tokens each):")
    print(f"Vocab size: {vocab_size}, Token 0 & 1 are common\n")
    
    for name, strategy in strategies.items():
        seq = generate_sequence(strategy)
        unique_tokens = len(set(seq))
        token_0_count = seq.count(0)
        
        print(f"{name}:")
        print(f"  Sequence: {seq}")
        print(f"  Unique tokens: {unique_tokens}, Token 0 count: {token_0_count}")
    print()


def example_probability_distribution():
    """Example 11: Visualizing probability distributions."""
    print("=" * 60)
    print("Example 11: Probability Distribution Visualization")
    print("=" * 60)
    
    from llm.sampling import softmax
    
    # Create logits with known pattern
    vocab_size = 20
    logits = np.zeros((1, vocab_size))
    logits[0, 0] = 5.0  # Very likely
    logits[0, 1] = 3.0  # Likely
    logits[0, 2] = 2.0  # Somewhat likely
    logits[0, 3:6] = 1.0  # Possible
    
    print("Probability distribution at different temperatures:")
    print(f"Vocab size: {vocab_size}")
    print(f"Logits[0:6]: {logits[0, :6]}")
    print()
    
    for temp in [0.5, 1.0, 2.0]:
        probs = softmax(logits, temperature=temp)[0]
        
        # ASCII visualization
        print(f"Temperature = {temp}:")
        for i in range(10):
            bar_len = int(probs[i] * 50)
            bar = "#" * bar_len
            print(f"  Token {i:2d}: {bar} {probs[i]:.3f}")
        print()


def example_beam_search():
    """Example 12: Beam search (basic)."""
    print("=" * 60)
    print("Example 12: Beam Search (Basic)")
    print("=" * 60)
    
    from llm.sampling import beam_search, softmax
    
    np.random.seed(42)
    vocab_size = 10
    
    # Simple logits function
    def logits_fn(input_ids):
        """Return random logits favoring lower token IDs."""
        logits = np.random.randn(1, vocab_size).astype(np.float32)
        # Make lower tokens more likely
        for i in range(vocab_size):
            logits[0, i] -= i * 0.3
        return logits
    
    # Starting sequence
    input_ids = np.array([[1, 2]])
    
    print(f"Starting sequence: {input_ids[0]}")
    print(f"Vocab size: {vocab_size}")
    
    # Beam search
    for beam_width in [1, 3, 5]:
        best_seq, best_score = beam_search(
            logits_fn=logits_fn,
            input_ids=input_ids,
            max_length=8,
            beam_width=beam_width,
        )
        
        print(f"\nBeam width = {beam_width}:")
        print(f"  Best sequence: {best_seq[0]}")
        print(f"  Score: {best_score:.4f}")
    
    print("\nNote: beam_width=1 is equivalent to greedy search")
    print()


if __name__ == "__main__":
    example_softmax_temperature()
    example_greedy_sampling()
    example_temperature_sampling()
    example_top_k_sampling()
    example_nucleus_sampling()
    example_combined_sampling()
    example_repetition_penalty()
    example_typical_sampling()
    example_sampling_config()
    example_generation_quality()
    example_probability_distribution()
    example_beam_search()
    
    print("All sampling examples completed!")
