"""
Train an Aya-compatible donor tokenizer on a Ukrainian corpus.

This script keeps the tokenizer family of a base Aya tokenizer by retraining it
with `train_new_from_iterator(...)` on a Ukrainian corpus. The result can then
be used as a donor tokenizer for build_hybrid_tokenizer.py.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: Missing required package. Please install dependencies:")
    print("  pip install transformers")
    sys.exit(1)

from tokenizer_utils import load_corpus_file, load_corpus_kobza, preprocess_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Aya-compatible donor tokenizer on a Ukrainian corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-tokenizer",
        type=str,
        default="CohereLabs/aya-expanse-8b",
        help="Base tokenizer path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        choices=["kobza", "file"],
        default="kobza",
        help="Corpus source: 'kobza' (Hugging Face) or 'file' (local).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to local UTF-8 text file, one sample per line (required for --corpus file).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100000,
        help="Number of samples to load from corpus.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for train_new_from_iterator.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Target vocabulary size. Default: keep the base tokenizer vocabulary size.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aya_uk_donor",
        help="Output directory for the trained donor tokenizer.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the base tokenizer.",
    )
    args = parser.parse_args()

    if args.corpus == "file" and not args.input_file:
        parser.error("--input-file is required when --corpus is 'file'")
    if args.samples <= 0:
        parser.error("--samples must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.vocab_size is not None and args.vocab_size <= 0:
        parser.error("--vocab-size must be > 0")

    return args


def load_corpus(corpus: str, samples: int, input_file: str | None) -> List[str]:
    if corpus == "file":
        texts = load_corpus_file(input_file, samples)
    else:
        texts = load_corpus_kobza(samples)

    cleaned = [preprocess_text(text) for text in texts if preprocess_text(text).strip()]
    if not cleaned:
        print("Error: No usable texts were loaded from the corpus.")
        sys.exit(1)
    return cleaned


def batch_iterator(texts: List[str], batch_size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def save_vocab_dump(tokenizer, output_dir: Path) -> None:
    vocab = tokenizer.get_vocab()
    sorted_vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(sorted_vocab, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()

    print(f"Loading base tokenizer: {args.base_tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_tokenizer,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    base_vocab_size = len(tokenizer)
    target_vocab_size = args.vocab_size or base_vocab_size

    print(f"Base tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Base vocabulary size: {base_vocab_size:,}")
    print(f"Target vocabulary size: {target_vocab_size:,}")

    texts = load_corpus(args.corpus, args.samples, args.input_file)
    print(f"Loaded {len(texts):,} training samples")

    print("Training Aya-compatible donor tokenizer...")
    donor_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(texts, args.batch_size),
        vocab_size=target_vocab_size,
        length=len(texts),
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    donor_tokenizer.save_pretrained(output_dir)
    save_vocab_dump(donor_tokenizer, output_dir)

    tokenizer_json_path = output_dir / "tokenizer.json"
    if tokenizer_json_path.exists():
        print(f"Saved tokenizer to: {tokenizer_json_path}")
    print(f"Saved donor tokenizer directory to: {output_dir}")
    print(f"Final vocabulary size: {len(donor_tokenizer):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
