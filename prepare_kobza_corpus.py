"""
Prepare a fast local UTF-8 corpus file from Goader/kobza.

Why this exists:
- Remote streaming from Hugging Face is convenient but often slow and fragile.
- For repeated tokenizer training/evaluation, the fastest practical workflow is:
  1) download/prepare Kobza locally once
  2) preprocess it once
  3) write one sample per line to a local text file
  4) train/evaluate from --corpus file
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Error: Missing required package. Please install dependencies:")
    print("  pip install datasets tqdm")
    sys.exit(1)

from tokenizer_utils import preprocess_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, preprocess, and export the Kobza corpus to a local UTF-8 text file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100000,
        help="Maximum number of samples to export",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/kobza_uk.txt",
        help="Output UTF-8 text file path, one sample per line",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of worker processes for local dataset preparation and preprocessing",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Optional Hugging Face datasets cache directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    return parser.parse_args()


def normalize_record(example: dict) -> dict:
    return {"text": preprocess_text(example.get("text", ""))}


def main() -> int:
    args = parse_args()

    if args.samples <= 0:
        print("Error: --samples must be > 0")
        return 1
    if args.num_proc <= 0:
        print("Error: --num-proc must be > 0")
        return 1

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"Error: Output file already exists: {output_path}")
        print("Use --overwrite to replace it.")
        return 1

    print("Loading Kobza dataset locally (non-streaming)...")
    dataset = load_dataset(
        "Goader/kobza",
        split="train",
        streaming=False,
        num_proc=args.num_proc,
        cache_dir=args.cache_dir,
    )

    total_available = len(dataset)
    limit = min(args.samples, total_available)
    if limit < total_available:
        dataset = dataset.select(range(limit))

    print(f"Loaded {len(dataset):,} rows; preprocessing with {args.num_proc} worker(s)...")
    dataset = dataset.map(
        normalize_record,
        num_proc=args.num_proc,
        desc="Normalizing texts",
    )

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, total=len(dataset), desc="Writing corpus", unit="text"):
            text = item.get("text", "")
            if text and text.strip():
                f.write(text.replace("\n", " ").strip())
                f.write("\n")
                written += 1

    print(f"Wrote {written:,} texts to: {output_path}")
    print("Use this file with: --corpus file --input-file <path>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
