"""
Evaluate a BPE tokenizer on Ukrainian text using key metrics:
- Token fertility (tokens per word, target <2)
- OOV coverage (fraction of <unk> tokens, target ~0)

This script:
1. Loads tokenizer from disk or trains new one
2. Loads evaluation corpus (Kobza or local file)
3. Calculates fertility and OOV metrics
4. Generates formatted evaluation report
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    from tokenizers import Tokenizer
    from tqdm import tqdm
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package. Please install dependencies:")
    print("  pip install datasets tokenizers transformers tqdm numpy")
    sys.exit(1)

from tokenizer_utils import (
    load_hf_tokenizer,
    load_local_tokenizer,
    load_corpus_kobza,
    load_corpus_file,
    split_text_into_words,
    tokenize_word,
    preprocess_text,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a BPE tokenizer on Ukrainian text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="ukr_bpe_tokenizer",
        help="Path to tokenizer directory or tokenizer.json file",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        help="HuggingFace model ID (e.g., 'meta-llama/Llama-3.1-8B-Instruct'). If provided, --tokenizer-path is ignored",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        choices=["kobza", "file"],
        default="kobza",
        help="Corpus source: 'kobza' (HuggingFace) or 'file' (local)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to local text file (required if --corpus file)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50000,
        help="Number of samples to load from corpus",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        help="Path to save evaluation report (JSON)",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Number of high-fertility word examples to show",
    )
    parser.add_argument(
        "--show-oov-examples",
        type=int,
        default=5,
        help="Number of OOV word examples to show",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.corpus == "file" and not args.input_file:
        parser.error("--input-file is required when --corpus is 'file'")

    return args



def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """
    Load tokenizer from disk.

    Args:
        tokenizer_path: Path to tokenizer directory or tokenizer.json file

    Returns:
        Loaded tokenizer
    """
    tokenizer_path = Path(tokenizer_path)

    # Check if path is directory or file
    if tokenizer_path.is_dir():
        tokenizer_file = tokenizer_path / "tokenizer.json"
    else:
        tokenizer_file = tokenizer_path

    if not tokenizer_file.exists():
        print(f"Error: Tokenizer file not found at {tokenizer_file}")
        sys.exit(1)

    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        print(f"Loaded tokenizer from: {tokenizer_file}")
        print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)



def calculate_fertility(
    tokenizer: Tokenizer,
    texts: List[str],
) -> Dict:
    """
    Calculate token fertility metrics (tokens per word).

    Args:
        tokenizer: Loaded tokenizer
        texts: List of evaluation texts

    Returns:
        Dictionary with fertility statistics
    """
    print("\nCalculating token fertility (tokens per word)...")

    tokens_per_word = []
    word_examples = {}  # Map word -> token_count for examples
    total_words = 0
    total_tokens = 0

    pbar = tqdm(total=len(texts), desc="Processing texts", unit="text")

    for text in texts:
        words = split_text_into_words(text)

        for word in words:
            # Tokenize word individually using helper function
            tokens = tokenize_word(tokenizer, word)
            num_tokens = len(tokens)

            tokens_per_word.append(num_tokens)
            total_words += 1
            total_tokens += num_tokens

            # Store examples of high-fertility words
            if num_tokens >= 5 and word not in word_examples:
                word_examples[word] = num_tokens

        pbar.update(1)

    pbar.close()

    # Calculate percentiles
    tokens_per_word = np.array(tokens_per_word)

    result = {
        "total_words": int(total_words),
        "total_tokens": int(total_tokens),
        "mean_tokens_per_word": float(np.mean(tokens_per_word)),
        "median_tokens_per_word": float(np.median(tokens_per_word)),
        "std_tokens_per_word": float(np.std(tokens_per_word)),
        "p50_tokens_per_word": float(np.percentile(tokens_per_word, 50)),
        "p75_tokens_per_word": float(np.percentile(tokens_per_word, 75)),
        "p95_tokens_per_word": float(np.percentile(tokens_per_word, 95)),
        "p99_tokens_per_word": float(np.percentile(tokens_per_word, 99)),
        "min_tokens_per_word": int(np.min(tokens_per_word)),
        "max_tokens_per_word": int(np.max(tokens_per_word)),
        "high_fertility_words": dict(sorted(
            word_examples.items(),
            key=lambda x: x[1],
            reverse=True
        )),
    }

    return result


def calculate_oov(
    tokenizer: Tokenizer,
    texts: List[str],
) -> Dict:
    """
    Calculate OOV (out-of-vocabulary) metrics.

    Args:
        tokenizer: Loaded tokenizer
        texts: List of evaluation texts

    Returns:
        Dictionary with OOV statistics
    """
    print("\nCalculating OOV metrics...")

    # Get UNK token ID based on tokenizer type
    unk_id = None

    if isinstance(tokenizer, Tokenizer):
        # Local BPE tokenizer
        vocab = tokenizer.get_vocab()
        unk_token = "<unk>"
        if unk_token in vocab:
            unk_id = vocab[unk_token]
    else:
        # HuggingFace tokenizer - try common UNK token IDs
        # Most HF tokenizers use 0 or have unk_token attribute
        if hasattr(tokenizer, 'unk_token_id'):
            unk_id = tokenizer.unk_token_id
        else:
            # Fallback to common UNK IDs
            unk_id = 0  # Most transformers use 0 as UNK

    total_tokens = 0
    unk_count = 0
    unk_words = {}  # Map word -> count

    pbar = tqdm(total=len(texts), desc="Encoding texts", unit="text")

    for text in texts:
        # Encode full text
        if isinstance(tokenizer, Tokenizer):
            # Local tokenizer
            encoding = tokenizer.encode(text)
            token_ids = encoding.ids
            token_strs = encoding.tokens
        else:
            # HuggingFace tokenizer
            token_ids = tokenizer.encode(text)
            # Try to get token strings if available
            try:
                token_strs = tokenizer.convert_ids_to_tokens(token_ids)
            except:
                token_strs = [str(tid) for tid in token_ids]

        for token_id, token_str in zip(token_ids, token_strs):
            total_tokens += 1

            if unk_id is not None and token_id == unk_id:
                unk_count += 1

                # Track UNK words (extract the word part)
                clean_word = str(token_str).replace("▁", "").replace("Ġ", "")
                if clean_word and clean_word not in ['<unk>', '[UNK]']:
                    unk_words[clean_word] = unk_words.get(clean_word, 0) + 1

        pbar.update(1)

    pbar.close()

    oov_fraction = unk_count / total_tokens if total_tokens > 0 else 0
    oov_percentage = oov_fraction * 100

    result = {
        "total_tokens": int(total_tokens),
        "unk_count": int(unk_count),
        "oov_fraction": float(oov_fraction),
        "oov_percentage": float(oov_percentage),
        "unk_token_id": int(unk_id) if unk_id is not None else None,
        "unk_words": dict(sorted(
            unk_words.items(),
            key=lambda x: x[1],
            reverse=True
        )),
    }

    return result


def tokenize_word(tokenizer, word: str) -> List[int]:
    """
    Tokenize a word using either local or HuggingFace tokenizer.

    Args:
        tokenizer: Either a local Tokenizer or HuggingFace AutoTokenizer
        word: Word to tokenize

    Returns:
        List of token IDs
    """
    if isinstance(tokenizer, Tokenizer):
        # Local BPE tokenizer
        encoding = tokenizer.encode(word)
        return encoding.ids
    else:
        # HuggingFace tokenizer
        tokens = tokenizer.encode(word)
        return tokens


def print_report(
    fertility: Dict,
    oov: Dict,
    tokenizer_path: str,
    corpus_type: str,
    num_samples: int,
    show_examples: int,
    show_oov_examples: int,
) -> None:
    """
    Print formatted evaluation report.

    Args:
        fertility: Fertility metrics dictionary
        oov: OOV metrics dictionary
        tokenizer_path: Path to tokenizer
        corpus_type: Corpus type used (kobza/file)
        num_samples: Number of samples evaluated
        show_examples: Number of high-fertility examples to show
        show_oov_examples: Number of OOV examples to show
    """
    print("\n" + "=" * 70)
    print("TOKENIZER EVALUATION REPORT")
    print("=" * 70)

    print(f"\nSetup:")
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Corpus: {corpus_type}")
    print(f"  Samples evaluated: {num_samples:,}")

    # Fertility metrics
    print(f"\n{'-' * 70}")
    print("TOKEN FERTILITY (tokens per word)")
    print(f"{'-' * 70}")
    print(f"  Total words: {fertility['total_words']:,}")
    print(f"  Total tokens: {fertility['total_tokens']:,}")
    print(f"  Mean tokens/word: {fertility['mean_tokens_per_word']:.3f}")
    print(f"  Median tokens/word: {fertility['median_tokens_per_word']:.3f}")
    print(f"  Std Dev: {fertility['std_tokens_per_word']:.3f}")
    print(f"\n  Percentiles:")
    print(f"    p50: {fertility['p50_tokens_per_word']:.3f} (50th)")
    print(f"    p75: {fertility['p75_tokens_per_word']:.3f} (75th)")
    print(f"    p95: {fertility['p95_tokens_per_word']:.3f} (95th)")
    print(f"    p99: {fertility['p99_tokens_per_word']:.3f} (99th)")
    print(f"\n  Range: [{fertility['min_tokens_per_word']}, {fertility['max_tokens_per_word']}]")

    # Fertility evaluation
    goal_fertility = 2.0
    if fertility['mean_tokens_per_word'] < goal_fertility:
        status = "✓ GOOD"
    elif fertility['mean_tokens_per_word'] < goal_fertility + 0.5:
        status = "~ OK"
    else:
        status = "✗ NEEDS IMPROVEMENT"
    print(f"\n  Target: < {goal_fertility} tokens/word")
    print(f"  Status: {status}")

    # High fertility examples
    if fertility['high_fertility_words']:
        print(f"\n  Examples of high-fertility words (>= 5 tokens):")
        for i, (word, count) in enumerate(
            sorted(fertility['high_fertility_words'].items(),
                   key=lambda x: x[1], reverse=True)[:show_examples],
            1
        ):
            print(f"    {i}. '{word}' → {count} tokens")

    # OOV metrics
    print(f"\n{'-' * 70}")
    print("OUT-OF-VOCABULARY (OOV) COVERAGE")
    print(f"{'-' * 70}")
    print(f"  Total tokens encoded: {oov['total_tokens']:,}")
    print(f"  UNK tokens: {oov['unk_count']:,}")
    print(f"  OOV fraction: {oov['oov_fraction']:.6f}")
    print(f"  OOV percentage: {oov['oov_percentage']:.3f}%")

    # OOV evaluation
    goal_oov = 1.0  # 1% or less
    if oov['oov_percentage'] < 0.5:
        status = "✓ EXCELLENT"
    elif oov['oov_percentage'] < goal_oov:
        status = "✓ GOOD"
    elif oov['oov_percentage'] < 2.0:
        status = "~ OK"
    else:
        status = "✗ NEEDS IMPROVEMENT"
    print(f"\n  Target: < {goal_oov}% OOV")
    print(f"  Status: {status}")

    # OOV examples
    if oov['unk_words']:
        print(f"\n  Examples of OOV words:")
        for i, (word, count) in enumerate(
            sorted(oov['unk_words'].items(),
                   key=lambda x: x[1], reverse=True)[:show_oov_examples],
            1
        ):
            print(f"    {i}. '{word}' (UNK count: {count})")
    else:
        print(f"\n  No OOV words found in evaluation corpus!")

    print(f"\n" + "=" * 70)


def save_report(
    fertility: Dict,
    oov: Dict,
    output_path: str,
    tokenizer_path: str,
    corpus_type: str,
    num_samples: int,
) -> None:
    """
    Save evaluation report to JSON file.

    Args:
        fertility: Fertility metrics dictionary
        oov: OOV metrics dictionary
        output_path: Path to save report
        tokenizer_path: Path to tokenizer
        corpus_type: Corpus type used
        num_samples: Number of samples evaluated
    """
    report = {
        "metadata": {
            "tokenizer_path": str(tokenizer_path),
            "corpus_type": corpus_type,
            "samples_evaluated": num_samples,
        },
        "fertility": fertility,
        "oov": oov,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to: {output_path}")
    except Exception as e:
        print(f"Error saving report: {e}")


def main():
    """Main entry point."""
    args = parse_args()

    # Load tokenizer
    if args.model_id:
        tokenizer = load_hf_tokenizer(args.model_id)
        tokenizer_name = args.model_id
    else:
        tokenizer = load_tokenizer(args.tokenizer_path)
        tokenizer_name = args.tokenizer_path

    # Load corpus
    if args.corpus == "kobza":
        texts = load_corpus_kobza(args.samples)
    else:  # file
        texts = load_corpus_file(args.input_file, args.samples)

    if not texts:
        print("Error: No texts loaded for evaluation")
        sys.exit(1)

    # Calculate metrics
    fertility = calculate_fertility(tokenizer, texts)
    oov = calculate_oov(tokenizer, texts)

    # Print report
    print_report(
        fertility,
        oov,
        tokenizer_name,
        args.corpus,
        len(texts),
        args.show_examples,
        args.show_oov_examples,
    )

    # Save report if requested
    if args.output_report:
        save_report(
            fertility,
            oov,
            args.output_report,
            tokenizer_name,
            args.corpus,
            len(texts),
        )


if __name__ == "__main__":
    main()

