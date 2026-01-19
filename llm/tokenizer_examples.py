"""
Examples for tokenizer implementations.

Demonstrates:
- Character-level tokenization
- BPE (Byte Pair Encoding)
- WordPiece tokenization
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_char_tokenizer():
    """Example 1: Character-level tokenizer."""
    print("=" * 60)
    print("Example 1: Character-Level Tokenizer")
    print("=" * 60)
    
    from llm.tokenizer import CharTokenizer
    
    # Sample texts
    texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
    ]
    
    # Create and fit tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(texts)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sample vocabulary: {dict(list(tokenizer.vocab.items())[:20])}")
    
    # Encode and decode
    text = "Hello, AI!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: '{text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print()


def example_bpe_tokenizer():
    """Example 2: BPE tokenizer."""
    print("=" * 60)
    print("Example 2: BPE (Byte Pair Encoding) Tokenizer")
    print("=" * 60)
    
    from llm.tokenizer import BPETokenizer
    
    # Training corpus
    texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a quick brown fox jumps over a lazy dog",
        "machine learning models learn patterns from data",
        "neural networks process information layer by layer",
        "transformers use attention mechanisms for context",
        "language models predict the next word in a sequence",
    ] * 10  # Repeat for more training data
    
    # Create and train BPE tokenizer
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.fit(texts)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Number of merges: {len(tokenizer.merges)}")
    print(f"First 10 merges: {tokenizer.merges[:10]}")
    
    # Show some vocabulary
    print(f"\nSample tokens in vocabulary:")
    for token, idx in list(tokenizer.vocab.items())[:15]:
        print(f"  '{token}': {idx}")
    
    # Encode and decode
    test_text = "the neural network learns"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Tokens: {[tokenizer.inverse_vocab[i] for i in encoded]}")
    print(f"Decoded: '{decoded}'")
    print()


def example_wordpiece_tokenizer():
    """Example 3: WordPiece tokenizer."""
    print("=" * 60)
    print("Example 3: WordPiece Tokenizer")
    print("=" * 60)
    
    from llm.tokenizer import WordPieceTokenizer
    
    # Training corpus
    texts = [
        "unbelievable achievement in science",
        "preprocessing and tokenization are important",
        "transformer models have revolutionized NLP",
        "understanding machine learning requires practice",
        "embedding layers convert tokens to vectors",
        "attention mechanisms capture relationships",
    ] * 10
    
    # Create and train WordPiece tokenizer
    tokenizer = WordPieceTokenizer(vocab_size=150)
    tokenizer.fit(texts)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Show tokens with ## prefix (subword continuations)
    subword_tokens = [t for t in tokenizer.vocab.keys() if t.startswith('##')]
    print(f"Subword tokens (## prefix): {subword_tokens[:15]}")
    
    # Encode and decode
    test_text = "understanding transformers"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Tokens: {[tokenizer.inverse_vocab[i] for i in encoded]}")
    print(f"Decoded: '{decoded}'")
    print()


def example_special_tokens():
    """Example 4: Special tokens handling."""
    print("=" * 60)
    print("Example 4: Special Tokens")
    print("=" * 60)
    
    from llm.tokenizer import CharTokenizer
    
    tokenizer = CharTokenizer()
    tokenizer.fit(["hello world"])
    
    print("Special tokens:")
    print(f"  <pad>: {tokenizer.pad_token_id}")
    print(f"  <unk>: {tokenizer.unk_token_id}")
    print(f"  <bos>: {tokenizer.bos_token_id}")
    print(f"  <eos>: {tokenizer.eos_token_id}")
    
    # Encode with and without special tokens
    text = "hi"
    with_special = tokenizer.encode(text, add_special_tokens=True)
    without_special = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"\nText: '{text}'")
    print(f"With special tokens: {with_special}")
    print(f"Without special tokens: {without_special}")
    
    # Decode with and without skipping special tokens
    decoded_skip = tokenizer.decode(with_special, skip_special_tokens=True)
    decoded_keep = tokenizer.decode(with_special, skip_special_tokens=False)
    
    print(f"\nDecoded (skip special): '{decoded_skip}'")
    print(f"Decoded (keep special): '{decoded_keep}'")
    print()


def example_unknown_tokens():
    """Example 5: Handling unknown tokens."""
    print("=" * 60)
    print("Example 5: Unknown Token Handling")
    print("=" * 60)
    
    from llm.tokenizer import CharTokenizer
    
    # Train on limited vocabulary
    tokenizer = CharTokenizer()
    tokenizer.fit(["abc"])  # Only knows a, b, c
    
    # Try to encode text with unknown characters
    text = "abcxyz"
    encoded = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"Vocabulary: {[k for k in tokenizer.vocab.keys() if k not in tokenizer.special_tokens]}")
    print(f"Text: '{text}'")
    print(f"Encoded: {encoded}")
    print(f"Note: {tokenizer.unk_token_id} is the <unk> token ID for 'x', 'y', 'z'")
    print()


def example_tokenizer_factory():
    """Example 6: Tokenizer factory function."""
    print("=" * 60)
    print("Example 6: Tokenizer Factory")
    print("=" * 60)
    
    from llm.tokenizer import create_tokenizer
    
    texts = ["hello world", "machine learning", "neural networks"]
    
    # Create different tokenizers using factory
    for tokenizer_type in ['char', 'bpe', 'wordpiece']:
        tokenizer = create_tokenizer(tokenizer_type, vocab_size=50)
        tokenizer.fit(texts)
        
        print(f"{tokenizer_type.upper()} Tokenizer:")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        
        encoded = tokenizer.encode("hello", add_special_tokens=False)
        print(f"  'hello' -> {encoded}")
        print()


def example_compare_tokenizers():
    """Example 7: Compare tokenization strategies."""
    print("=" * 60)
    print("Example 7: Tokenizer Comparison")
    print("=" * 60)
    
    from llm.tokenizer import CharTokenizer, BPETokenizer, WordPieceTokenizer
    
    # Training corpus
    texts = [
        "the neural network learns patterns from data",
        "deep learning models process complex information",
        "attention mechanisms revolutionized language models",
    ] * 20
    
    # Create tokenizers
    char_tok = CharTokenizer()
    char_tok.fit(texts)
    
    bpe_tok = BPETokenizer(vocab_size=80)
    bpe_tok.fit(texts)
    
    wp_tok = WordPieceTokenizer(vocab_size=80)
    wp_tok.fit(texts)
    
    # Compare on test sentence
    test = "learning neural patterns"
    
    print(f"Test sentence: '{test}'")
    print()
    
    for name, tok in [("Character", char_tok), ("BPE", bpe_tok), ("WordPiece", wp_tok)]:
        encoded = tok.encode(test, add_special_tokens=False)
        tokens = [tok.inverse_vocab.get(i, '<unk>') for i in encoded]
        print(f"{name}:")
        print(f"  Vocab size: {tok.vocab_size}")
        print(f"  Num tokens: {len(encoded)}")
        print(f"  Tokens: {tokens}")
        print()


def example_batch_encoding():
    """Example 8: Batch encoding with padding."""
    print("=" * 60)
    print("Example 8: Batch Encoding with Padding")
    print("=" * 60)
    
    from llm.tokenizer import CharTokenizer
    
    tokenizer = CharTokenizer()
    tokenizer.fit(["hello world programming"])
    
    # Multiple texts of different lengths
    texts = ["hi", "hello", "hello world"]
    
    # Encode each
    encoded = [tokenizer.encode(t, add_special_tokens=True) for t in texts]
    
    # Find max length and pad
    max_len = max(len(e) for e in encoded)
    padded = []
    
    for e in encoded:
        pad_len = max_len - len(e)
        padded.append(e + [tokenizer.pad_token_id] * pad_len)
    
    print("Encoded and padded batch:")
    for i, (text, enc, pad) in enumerate(zip(texts, encoded, padded)):
        print(f"  '{text}':")
        print(f"    Encoded: {enc}")
        print(f"    Padded:  {pad}")
    
    # Convert to numpy array
    batch = np.array(padded)
    print(f"\nBatch shape: {batch.shape}")
    print()


if __name__ == "__main__":
    example_char_tokenizer()
    example_bpe_tokenizer()
    example_wordpiece_tokenizer()
    example_special_tokens()
    example_unknown_tokens()
    example_tokenizer_factory()
    example_compare_tokenizers()
    example_batch_encoding()
    
    print("All tokenizer examples completed!")
