"""
Train a BPE tokenizer on the Ukrainian Kobza dataset.

This script:
1. Loads a subsample of the Kobza dataset from HuggingFace
2. Preprocesses text (NFC normalization, unify apostrophes)
3. Trains a BPE tokenizer with Metaspace pretokenizer
4. Saves tokenizer, merge pairs, and vocabulary
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

try:
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required package. Please install dependencies:")
    print("  pip install datasets tokenizers tqdm")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on the Ukrainian Kobza dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100000,
        help="Number of texts to use for training",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Target vocabulary size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ukr_bpe_tokenizer",
        help="Output directory for tokenizer files",
    )
    return parser.parse_args()


def preprocess_text(text: str) -> str:
    """
    Preprocess text for tokenization.
    
    - Apply NFC normalization
    - Unify various apostrophe characters to standard '
    """
    if not text or not isinstance(text, str):
        return ""
    
    # NFC normalization
    text = unicodedata.normalize("NFC", text)
    
    # Unify apostrophes: various apostrophe-like characters to standard '
    # Common variants: ', ʼ, `, ´, ʻ, ', ', ‛, ′
    apostrophe_variants = ["ʼ", "`", "´", "ʻ", "'", "'", "‛", "′", "ˈ", "ˊ"]
    for variant in apostrophe_variants:
        text = text.replace(variant, "'")
    
    return text


def load_kobza_dataset(num_samples: int) -> list[str]:
    """
    Load and preprocess texts from the Kobza dataset.
    
    Args:
        num_samples: Maximum number of texts to load
        
    Returns:
        List of preprocessed text strings
    """
    print(f"Loading Kobza dataset (Goader/kobza)...")
    
    try:
        # Load the dataset in streaming mode for efficiency
        dataset = load_dataset("Goader/kobza", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have internet connection and the dataset exists.")
        sys.exit(1)
    
    texts = []
    print(f"Processing up to {num_samples:,} samples...")
    
    # Use tqdm for progress bar
    pbar = tqdm(total=num_samples, desc="Loading texts")
    
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        
        # The Kobza dataset has 'text' field
        text = item.get("text", "")
        processed = preprocess_text(text)
        
        if processed.strip():
            texts.append(processed)
            pbar.update(1)
    
    pbar.close()
    
    return texts


def create_bpe_tokenizer(vocab_size: int) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """
    Create a BPE tokenizer with the specified configuration.
    
    Args:
        vocab_size: Target vocabulary size
        
    Returns:
        Tuple of (tokenizer, trainer)
    """
    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Set up pretokenizer: Metaspace + Digits split
    # Metaspace replaces spaces with ▁ and adds prefix space
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True),
        pre_tokenizers.Digits(individual_digits=True),
    ])
    
    # Special tokens
    special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
    
    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    return tokenizer, trainer


def train_tokenizer(
    tokenizer: Tokenizer,
    trainer: trainers.BpeTrainer,
    texts: list[str],
) -> Tokenizer:
    """
    Train the tokenizer on the provided texts.
    
    Args:
        tokenizer: The tokenizer to train
        trainer: The trainer configuration
        texts: List of training texts
        
    Returns:
        Trained tokenizer
    """
    print(f"\nTraining BPE tokenizer on {len(texts):,} texts...")
    
    # Train from iterator for memory efficiency
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, output_dir: str) -> None:
    """
    Save tokenizer files: tokenizer.json, vocab.json, merges.txt
    
    Args:
        tokenizer: Trained tokenizer
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full tokenizer
    tokenizer_path = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer to: {tokenizer_path}")
    
    # Extract and save vocabulary
    vocab = tokenizer.get_vocab()
    vocab_path = output_path / "vocab.json"
    
    # Sort by token ID for consistency
    sorted_vocab = dict(sorted(vocab.items(), key=lambda x: x[1]))
    
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(sorted_vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary ({len(vocab):,} tokens) to: {vocab_path}")
    
    # Extract and save merge pairs from the model
    merges_path = output_path / "merges.txt"
    
    # Load the tokenizer JSON to extract merges
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    
    merges = tokenizer_data.get("model", {}).get("merges", [])
    
    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            # Merges are stored as "token1 token2" strings
            f.write(f"{merge}\n")
    
    print(f"Saved {len(merges):,} merge pairs to: {merges_path}")


def contains_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters."""
    return bool(re.search(r"[\u0400-\u04FF]", text))


def analyze_vocabulary(tokenizer: Tokenizer) -> None:
    """
    Analyze and print vocabulary statistics.
    
    Args:
        tokenizer: Trained tokenizer
    """
    vocab = tokenizer.get_vocab()
    
    # Filter tokens containing Cyrillic characters
    cyrillic_tokens = {
        token: idx for token, idx in vocab.items() 
        if contains_cyrillic(token)
    }
    
    print(f"\n{'='*60}")
    print("VOCABULARY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total vocabulary size: {len(vocab):,}")
    print(f"Tokens with Cyrillic characters: {len(cyrillic_tokens):,}")
    
    # Top 20 Cyrillic tokens (sorted by ID, which roughly corresponds to frequency for BPE)
    # Lower IDs typically mean more frequent tokens (earlier in training)
    print(f"\nTop 20 most common Ukrainian tokens (by merge order):")
    print("-" * 40)
    
    sorted_cyrillic = sorted(cyrillic_tokens.items(), key=lambda x: x[1])[:20]
    for i, (token, idx) in enumerate(sorted_cyrillic, 1):
        # Display the token, handling the metaspace character
        display_token = token.replace("▁", "▁ (space prefix)")
        print(f"  {i:2}. [{idx:5}] {repr(token):30} -> {display_token}")


def show_example_tokenizations(tokenizer: Tokenizer) -> None:
    """
    Show example tokenizations of Ukrainian sentences.
    
    Args:
        tokenizer: Trained tokenizer
    """
    examples = [
        "Україна — незалежна європейська держава.",
        "Київ — столиця України.",
        "Сьогодні гарна погода, сонце світить яскраво.",
        "Доброго дня! Як справи?",
        "Програмування — це мистецтво створення алгоритмів.",
    ]
    
    print(f"\n{'='*60}")
    print("EXAMPLE TOKENIZATIONS")
    print(f"{'='*60}")
    
    for i, text in enumerate(examples, 1):
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        
        print(f"\n{i}. Input: {text}")
        print(f"   Tokens ({len(tokens)}): {tokens}")
        print(f"   IDs: {encoding.ids}")


def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("BPE TOKENIZER TRAINING FOR UKRAINIAN")
    print(f"{'='*60}")
    print(f"Settings:")
    print(f"  - Samples: {args.samples:,}")
    print(f"  - Vocab size: {args.vocab_size:,}")
    print(f"  - Output dir: {args.output}")
    print(f"{'='*60}\n")
    
    try:
        # Step 1: Load dataset
        texts = load_kobza_dataset(args.samples)
        print(f"\nLoaded {len(texts):,} texts for training")
        
        if len(texts) == 0:
            print("Error: No texts loaded. Check dataset availability.")
            sys.exit(1)
        
        # Step 2: Create tokenizer
        tokenizer, trainer = create_bpe_tokenizer(args.vocab_size)
        
        # Step 3: Train tokenizer
        tokenizer = train_tokenizer(tokenizer, trainer, texts)
        
        # Step 4: Save tokenizer and related files
        print(f"\nSaving tokenizer files...")
        save_tokenizer(tokenizer, args.output)
        
        # Step 5: Print statistics
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Number of texts processed: {len(texts):,}")
        print(f"Final vocabulary size: {tokenizer.get_vocab_size():,}")
        
        # Analyze vocabulary
        analyze_vocabulary(tokenizer)
        
        # Show example tokenizations
        show_example_tokenizations(tokenizer)
        
        print(f"\n{'='*60}")
        print(f"Output files saved to: {args.output}/")
        print(f"  - tokenizer.json (full tokenizer)")
        print(f"  - vocab.json (vocabulary)")
        print(f"  - merges.txt (BPE merge pairs)")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

