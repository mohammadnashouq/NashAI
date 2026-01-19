"""
Tokenization implementations for LLMs.

Implements:
- Byte Pair Encoding (BPE)
- WordPiece tokenization
- Basic character-level tokenizer
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import json


class BaseTokenizer:
    """Base class for tokenizers."""
    
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }
    
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    @property
    def pad_token_id(self) -> int:
        return self.special_tokens['<pad>']
    
    @property
    def unk_token_id(self) -> int:
        return self.special_tokens['<unk>']
    
    @property
    def bos_token_id(self) -> int:
        return self.special_tokens['<bos>']
    
    @property
    def eos_token_id(self) -> int:
        return self.special_tokens['<eos>']
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        raise NotImplementedError
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        raise NotImplementedError
    
    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'type': self.__class__.__name__
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BaseTokenizer':
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.vocab = data['vocab']
        tokenizer.special_tokens = data['special_tokens']
        tokenizer.inverse_vocab = {int(v): k for k, v in tokenizer.vocab.items()}
        return tokenizer


class CharTokenizer(BaseTokenizer):
    """
    Simple character-level tokenizer.
    
    Useful for understanding tokenization basics and small experiments.
    """
    
    def __init__(self):
        super().__init__()
        self._init_special_tokens()
    
    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Add characters to vocabulary
        idx = len(self.special_tokens)
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
                idx += 1
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to character token IDs."""
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        
        for char in text:
            ids.append(self.vocab.get(char, self.unk_token_id))
        
        if add_special_tokens:
            ids.append(self.eos_token_id)
        
        return ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        chars = []
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()
        
        for idx in token_ids:
            if idx in special_ids:
                continue
            chars.append(self.inverse_vocab.get(idx, '<unk>'))
        
        return ''.join(chars)


class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer.
    
    BPE iteratively merges the most frequent pair of consecutive tokens.
    Used in GPT-2, RoBERTa, and many other models.
    
    Algorithm:
    1. Start with character-level tokens
    2. Count frequency of all adjacent pairs
    3. Merge most frequent pair into new token
    4. Repeat until vocabulary size reached
    """
    
    def __init__(self, vocab_size: int = 1000):
        super().__init__()
        self.target_vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []  # Ordered list of merges
        self._init_special_tokens()
    
    def _get_pairs(self, word: Tuple[str, ...]) -> Dict[Tuple[str, str], int]:
        """Get frequency of adjacent pairs in a word."""
        pairs = defaultdict(int)
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
        return pairs
    
    def _get_corpus_pairs(self, word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Get frequency of all pairs across corpus."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            word_pairs = self._get_pairs(word)
            for pair, count in word_pairs.items():
                pairs[pair] += count * freq
        return pairs
    
    def _merge_pair(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        """Merge a pair in a word."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)
    
    def fit(self, texts: List[str]):
        """
        Learn BPE merges from corpus.
        
        Args:
            texts: List of training texts
        """
        # Tokenize into words and add end-of-word marker
        word_freqs: Dict[Tuple[str, ...], int] = defaultdict(int)
        
        for text in texts:
            # Simple word tokenization (split on whitespace and punctuation)
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            for word in words:
                # Add end-of-word marker and split into characters
                word_tuple = tuple(word) + ('</w>',)
                word_freqs[word_tuple] += 1
        
        # Initialize vocabulary with characters
        vocab_set: Set[str] = set()
        for word in word_freqs.keys():
            vocab_set.update(word)
        
        # Add characters to vocabulary
        idx = len(self.special_tokens)
        for char in sorted(vocab_set):
            self.vocab[char] = idx
            self.inverse_vocab[idx] = char
            idx += 1
        
        # BPE merge loop
        while len(self.vocab) < self.target_vocab_size:
            # Get most frequent pair
            pairs = self._get_corpus_pairs(word_freqs)
            if not pairs:
                break
            
            best_pair = max(pairs.keys(), key=lambda p: pairs[p])
            
            # Merge the pair in all words
            new_word_freqs: Dict[Tuple[str, ...], int] = {}
            for word, freq in word_freqs.items():
                new_word = self._merge_pair(word, best_pair)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
            
            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = idx
                self.inverse_vocab[idx] = merged_token
                idx += 1
            
            # Record the merge
            self.merges.append(best_pair)
        
        print(f"BPE training complete. Vocabulary size: {len(self.vocab)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using learned merges."""
        # Start with characters
        word = tuple(word) + ('</w>',)
        
        # Apply merges in order
        for merge in self.merges:
            word = self._merge_pair(word, merge)
        
        return list(word)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to BPE token IDs."""
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        
        # Tokenize words
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        for word in words:
            tokens = self._tokenize_word(word)
            for token in tokens:
                ids.append(self.vocab.get(token, self.unk_token_id))
        
        if add_special_tokens:
            ids.append(self.eos_token_id)
        
        return ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode BPE token IDs to text."""
        tokens = []
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()
        
        for idx in token_ids:
            if idx in special_ids:
                continue
            token = self.inverse_vocab.get(idx, '<unk>')
            tokens.append(token)
        
        # Join tokens and handle end-of-word markers
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def save(self, path: str):
        """Save BPE tokenizer with merges."""
        data = {
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'merges': self.merges,
            'target_vocab_size': self.target_vocab_size,
            'type': 'BPETokenizer'
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load BPE tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['target_vocab_size'])
        tokenizer.vocab = data['vocab']
        tokenizer.special_tokens = data['special_tokens']
        tokenizer.merges = [tuple(m) for m in data['merges']]
        tokenizer.inverse_vocab = {int(v): k for k, v in tokenizer.vocab.items()}
        return tokenizer


class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece tokenizer.
    
    Similar to BPE but uses likelihood instead of frequency for merging.
    Used in BERT and related models.
    
    Key difference from BPE:
    - Uses ## prefix for subword continuations
    - Merges based on likelihood: score(ab) = freq(ab) / (freq(a) * freq(b))
    """
    
    def __init__(self, vocab_size: int = 1000):
        super().__init__()
        self.target_vocab_size = vocab_size
        self.continuing_subword_prefix = '##'
        self._init_special_tokens()
    
    def fit(self, texts: List[str]):
        """
        Learn WordPiece vocabulary from corpus.
        
        Args:
            texts: List of training texts
        """
        # Count word frequencies
        word_freqs: Dict[str, int] = defaultdict(int)
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            for word in words:
                word_freqs[word] += 1
        
        # Initialize with characters
        char_freqs: Dict[str, int] = defaultdict(int)
        for word, freq in word_freqs.items():
            for i, char in enumerate(word):
                if i == 0:
                    char_freqs[char] += freq
                else:
                    char_freqs[self.continuing_subword_prefix + char] += freq
        
        # Add characters to vocabulary
        idx = len(self.special_tokens)
        for char in sorted(char_freqs.keys()):
            self.vocab[char] = idx
            self.inverse_vocab[idx] = char
            idx += 1
        
        # Split words into subwords
        splits: Dict[str, List[str]] = {}
        for word in word_freqs.keys():
            split = [word[0]] + [self.continuing_subword_prefix + c for c in word[1:]]
            splits[word] = split
        
        # WordPiece merge loop
        while len(self.vocab) < self.target_vocab_size:
            # Compute pair scores
            pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
            token_freqs: Dict[str, int] = defaultdict(int)
            
            for word, freq in word_freqs.items():
                split = splits[word]
                for token in split:
                    token_freqs[token] += freq
                for i in range(len(split) - 1):
                    pair_freqs[(split[i], split[i + 1])] += freq
            
            if not pair_freqs:
                break
            
            # Compute scores (likelihood-based)
            scores = {}
            for pair, freq in pair_freqs.items():
                score = freq / (token_freqs[pair[0]] * token_freqs[pair[1]])
                scores[pair] = score
            
            # Get best pair
            best_pair = max(scores.keys(), key=lambda p: scores[p])
            
            # Create merged token
            merged = best_pair[0] + best_pair[1].replace(self.continuing_subword_prefix, '')
            
            # Add to vocabulary
            if merged not in self.vocab:
                self.vocab[merged] = idx
                self.inverse_vocab[idx] = merged
                idx += 1
            
            # Update splits
            for word in splits:
                split = splits[word]
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and split[i] == best_pair[0] and split[i + 1] == best_pair[1]:
                        new_split.append(merged)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[word] = new_split
        
        print(f"WordPiece training complete. Vocabulary size: {len(self.vocab)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using WordPiece."""
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            found = False
            
            while start < end:
                # Create subword with proper prefix
                if start == 0:
                    subword = word[start:end]
                else:
                    subword = self.continuing_subword_prefix + word[start:end]
                
                if subword in self.vocab:
                    tokens.append(subword)
                    found = True
                    break
                end -= 1
            
            if not found:
                # Unknown character
                if start == 0:
                    tokens.append(word[start])
                else:
                    tokens.append(self.continuing_subword_prefix + word[start])
                start += 1
            else:
                start = end
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to WordPiece token IDs."""
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        for word in words:
            tokens = self._tokenize_word(word)
            for token in tokens:
                ids.append(self.vocab.get(token, self.unk_token_id))
        
        if add_special_tokens:
            ids.append(self.eos_token_id)
        
        return ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode WordPiece token IDs to text."""
        tokens = []
        special_ids = set(self.special_tokens.values()) if skip_special_tokens else set()
        
        for idx in token_ids:
            if idx in special_ids:
                continue
            token = self.inverse_vocab.get(idx, '<unk>')
            tokens.append(token)
        
        # Join tokens, handling ## prefix
        if not tokens:
            return ''
        
        text_parts = []
        current_word = []
        
        for token in tokens:
            if token.startswith(self.continuing_subword_prefix):
                current_word.append(token[2:])  # Remove ##
            else:
                if current_word:
                    text_parts.append(''.join(current_word))
                current_word = [token]
        
        if current_word:
            text_parts.append(''.join(current_word))
        
        return ' '.join(text_parts)


def create_tokenizer(tokenizer_type: str = 'bpe', vocab_size: int = 1000) -> BaseTokenizer:
    """
    Factory function to create tokenizers.
    
    Args:
        tokenizer_type: 'char', 'bpe', or 'wordpiece'
        vocab_size: Target vocabulary size (for BPE and WordPiece)
    
    Returns:
        Tokenizer instance
    """
    if tokenizer_type == 'char':
        return CharTokenizer()
    elif tokenizer_type == 'bpe':
        return BPETokenizer(vocab_size=vocab_size)
    elif tokenizer_type == 'wordpiece':
        return WordPieceTokenizer(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
