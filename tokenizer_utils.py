# tokenizer_utils.py
"""
Shared utility functions for tokenizer evaluation and comparison.

This module contains common functions reused by evaluate_tokenizer.py and
compare_tokenizers.py to avoid code duplication.
"""

import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    from tqdm import tqdm
except ImportError:
    print("Error: Missing required package. Please install dependencies:")
    print("  pip install datasets tokenizers transformers tqdm numpy")
    sys.exit(1)


# SentencePiece-style byte fallback token pattern: <0xAB>
_BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def preprocess_text(text: str) -> str:
    """
    Preprocess text for tokenization.

    - Apply NFC normalization
    - Unify various apostrophe characters to standard '
    """
    if not text or not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)

    apostrophe_variants = [
        "ʼ",  # U+02BC
        "’",  # U+2019 (common)
        "`",
        "´",
        "ʻ",
        "‛",
        "′",
        "ˈ",
        "ˊ",
    ]
    for variant in apostrophe_variants:
        text = text.replace(variant, "'")

    return text


def load_hf_tokenizer(model_id: str):
    """
    Load tokenizer from HuggingFace model.

    Returns:
        (tokenizer, model_id) or (None, None)
    """
    try:
        print(f"  Loading {model_id}...", end="", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # We only evaluate tokenization; avoid max-length warnings from transformers.
        try:
            tokenizer.model_max_length = 10**12
        except Exception:
            pass

        vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else 0
        print(" ✓")
        if vocab_size > 0:
            print(f"    Vocabulary size: {vocab_size:,}")
        return tokenizer, model_id
    except Exception as e:
        print(" ✗")
        print(f"    Warning: Could not load {model_id}: {e}")
        return None, None


def load_local_tokenizer(tokenizer_path: str = "ukr_bpe_tokenizer"):
    """
    Load local BPE tokenizer from disk.

    Returns:
        (tokenizer, "Local BPE Tokenizer") or (None, None)
    """
    try:
        tokenizer_path = Path(tokenizer_path)
        if tokenizer_path.is_dir():
            tokenizer_file = tokenizer_path / "tokenizer.json"
        else:
            tokenizer_file = tokenizer_path

        if not tokenizer_file.exists():
            print(f"Error: Tokenizer file not found at {tokenizer_file}")
            return None, None

        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        print(f"Loaded tokenizer from: {tokenizer_file}")
        print(f"  Vocabulary size: {tokenizer.get_vocab_size():,}")
        return tokenizer, "Local BPE Tokenizer"
    except Exception as e:
        print(f"Error loading local tokenizer: {e}")
        return None, None


def load_corpus_kobza(num_samples: int) -> List[str]:
    """
    Load texts from Kobza dataset (HuggingFace, streaming mode).
    """
    print("\nLoading Kobza dataset (Goader/kobza)...")

    try:
        dataset = load_dataset("Goader/kobza", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have internet connection and the dataset exists.")
        return []

    texts: List[str] = []
    pbar = tqdm(total=num_samples, desc="Loading texts", unit="text")

    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        text = item.get("text", "")
        processed = preprocess_text(text)
        if processed.strip():
            texts.append(processed)
            pbar.update(1)

    pbar.close()
    print(f"Loaded {len(texts):,} texts from Kobza")
    return texts


def load_corpus_file(file_path: str, num_samples: int) -> List[str]:
    """
    Load texts from a local file, one sample per line.
    """
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return []

    print(f"\nLoading texts from file: {file_path}")

    texts: List[str] = []
    pbar = tqdm(total=num_samples, desc="Loading texts", unit="text")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(texts) >= num_samples:
                    break
                processed = preprocess_text(line.strip())
                if processed:
                    texts.append(processed)
                    pbar.update(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    pbar.close()
    print(f"Loaded {len(texts):,} texts from file")
    return texts


def split_text_into_words(text: str) -> List[str]:
    """
    Split text into words using whitespace boundaries.
    """
    return [w for w in re.split(r"\s+", text) if w]


def tokenize_text(tokenizer, text: str) -> List[int]:
    """
    Tokenize text with the given tokenizer.
    For HF tokenizers, special tokens are disabled for fair comparison.
    """
    try:
        if isinstance(tokenizer, Tokenizer):
            return tokenizer.encode(text).ids
        return tokenizer.encode(text, add_special_tokens=False, truncation=False)
    except Exception as e:
        print(f"Tokenization error: {e}")
        return []


def decode_ids(tokenizer, ids: List[int]) -> Optional[str]:
    """
    Decode token IDs back to text. Returns None if decoding is unsupported or fails.
    """
    if not ids:
        return ""

    try:
        if isinstance(tokenizer, Tokenizer):
            # tokenizers.Tokenizer.decode expects a list of ids
            return tokenizer.decode(ids)
        # HF: keep special tokens off; we don't add them anyway
        return tokenizer.decode(ids, skip_special_tokens=False)
    except Exception:
        return None


def tokenizer_supports_decode(tokenizer) -> bool:
    """
    Best-effort check if tokenizer supports decoding.
    """
    try:
        _ = decode_ids(tokenizer, [0])
        return _ is not None
    except Exception:
        return False


def get_unk_id(tokenizer) -> Optional[int]:
    """
    Return unk token id for HF tokenizers, or None if not supported / unknown.
    """
    if isinstance(tokenizer, Tokenizer):
        return None
    return getattr(tokenizer, "unk_token_id", None)


def get_vocab_map(tokenizer) -> Dict[str, int]:
    """
    Return token->id vocab map for either local or HF tokenizers.
    """
    try:
        if isinstance(tokenizer, Tokenizer):
            return tokenizer.get_vocab()
        if hasattr(tokenizer, "get_vocab"):
            return tokenizer.get_vocab()
    except Exception:
        pass
    return {}


def get_byte_fallback_ids(tokenizer) -> Tuple[set, bool]:
    """
    Identify SentencePiece-style byte fallback token IDs by scanning vocab for tokens like <0xAB>.
    Returns: (ids_set, supported_flag)
    """
    vocab = get_vocab_map(tokenizer)
    if not vocab:
        return set(), False

    byte_ids = {tid for tok, tid in vocab.items() if _BYTE_FALLBACK_RE.match(tok)}
    return byte_ids, bool(byte_ids)
