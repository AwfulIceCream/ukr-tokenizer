"""
Train an Aya-compatible donor tokenizer on a Ukrainian corpus.

This script keeps the tokenizer family of a base Aya tokenizer by retraining it
with `train_new_from_iterator(...)` on a Ukrainian corpus. The result can then
be used as a donor tokenizer for build_hybrid_tokenizer.py.
"""

import argparse
import codecs
import glob
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

try:
    from transformers import AutoTokenizer
    from tqdm import tqdm
except ImportError:
    print("Error: Missing required package. Please install dependencies:")
    print("  pip install transformers tqdm")
    sys.exit(1)

from tokenizer_utils import load_corpus_kobza, preprocess_text


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
        help="Path to local UTF-8 text file, one sample per line.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Path to a directory of local UTF-8 text shards.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        help="Glob for local UTF-8 text shards, for example 'data/hf_test/*.txt'.",
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

    if args.corpus == "file":
        file_inputs = [args.input_file, args.input_dir, args.input_glob]
        if sum(value is not None for value in file_inputs) != 1:
            parser.error(
                "When --corpus is 'file', provide exactly one of "
                "--input-file, --input-dir, or --input-glob."
            )
    if args.corpus != "file" and args.samples <= 0:
        parser.error("--samples must be > 0 unless --corpus file is used")
    if args.corpus == "file" and args.samples < 0:
        parser.error("--samples must be >= 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.vocab_size is not None and args.vocab_size <= 0:
        parser.error("--vocab-size must be > 0")

    return args


def resolve_input_paths(
    input_file: Optional[str],
    input_dir: Optional[str],
    input_glob: Optional[str],
) -> List[Path]:
    if input_file is not None:
        path = Path(input_file)
        if not path.exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
        return [path]

    if input_dir is not None:
        directory = Path(input_dir)
        if not directory.exists() or not directory.is_dir():
            print(f"Error: Directory not found: {directory}")
            sys.exit(1)
        paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
        if not paths:
            print(f"Error: No .txt files found in directory: {directory}")
            sys.exit(1)
        return paths

    assert input_glob is not None
    paths = sorted(Path(path) for path in glob.glob(input_glob))
    paths = [path for path in paths if path.is_file()]
    if not paths:
        print(f"Error: No files matched glob: {input_glob}")
        sys.exit(1)
    return paths


def load_kobza_corpus(samples: int) -> List[str]:
    texts = load_corpus_kobza(samples)
    cleaned = [preprocess_text(text) for text in texts if preprocess_text(text).strip()]
    if not cleaned:
        print("Error: No usable texts were loaded from the corpus.")
        sys.exit(1)
    return cleaned


class FileBatchIterator:
    def __init__(self, paths: List[Path], sample_limit: Optional[int], batch_size: int):
        self.paths = paths
        self.sample_limit = sample_limit
        self.batch_size = batch_size
        self.yielded_texts = 0
        self.yielded_batches = 0

    def _iter_lines(self) -> Iterator[str]:
        decoder = codecs.getincrementaldecoder("utf-8")()
        buffer = ""

        for path in self.paths:
            print(f"Reading shard: {path}")
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break

                    buffer += decoder.decode(chunk)

                    while True:
                        newline_pos = buffer.find("\n")
                        if newline_pos < 0:
                            break

                        line = buffer[:newline_pos]
                        buffer = buffer[newline_pos + 1 :]
                        yield line.rstrip("\r")

        buffer += decoder.decode(b"", final=True)
        if buffer:
            yield buffer.rstrip("\r")

    def __iter__(self) -> Iterator[List[str]]:
        batch: List[str] = []
        pbar = tqdm(total=self.sample_limit, desc="Tokenizer training texts", unit="text")
        try:
            for line in self._iter_lines():
                if self.sample_limit is not None and self.yielded_texts >= self.sample_limit:
                    break

                processed = preprocess_text(line.strip())
                if not processed:
                    continue

                batch.append(processed)
                self.yielded_texts += 1
                pbar.update(1)

                if len(batch) >= self.batch_size:
                    self.yielded_batches += 1
                    yield batch
                    batch = []

            if batch:
                self.yielded_batches += 1
                yield batch
        finally:
            pbar.close()


def list_batch_iterator(texts: List[str], batch_size: int) -> Iterable[List[str]]:
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, len(texts), batch_size),
        total=total_batches,
        desc="Tokenizer training batches",
        unit="batch",
    ):
        yield texts[start : start + batch_size]


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

    train_kwargs = {"vocab_size": target_vocab_size}
    actual_text_count: Optional[int] = None
    total_batches: Optional[int] = None

    if args.corpus == "file":
        paths = resolve_input_paths(args.input_file, args.input_dir, args.input_glob)
        print(f"Training from {len(paths):,} local text file(s)")
        sample_limit = args.samples if args.samples > 0 else None
        iterator = FileBatchIterator(paths, sample_limit=sample_limit, batch_size=args.batch_size)
        train_iterator: Iterable[List[str]] = iterator
    else:
        texts = load_kobza_corpus(args.samples)
        print(f"Loaded {len(texts):,} training samples")
        train_iterator = list_batch_iterator(texts, args.batch_size)
        train_kwargs["length"] = len(texts)
        actual_text_count = len(texts)
        total_batches = (len(texts) + args.batch_size - 1) // args.batch_size

    print("Training Aya-compatible donor tokenizer...")
    train_start = time.perf_counter()
    donor_tokenizer = tokenizer.train_new_from_iterator(
        train_iterator,
        **train_kwargs,
    )
    train_elapsed = time.perf_counter() - train_start

    if args.corpus == "file":
        actual_text_count = iterator.yielded_texts
        total_batches = iterator.yielded_batches
        print(f"Consumed {actual_text_count:,} training samples from local files")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    donor_tokenizer.save_pretrained(output_dir)
    save_vocab_dump(donor_tokenizer, output_dir)

    tokenizer_json_path = output_dir / "tokenizer.json"
    if tokenizer_json_path.exists():
        print(f"Saved tokenizer to: {tokenizer_json_path}")
    print(f"Saved donor tokenizer directory to: {output_dir}")
    print(f"Final vocabulary size: {len(donor_tokenizer):,}")
    print(f"Training time: {train_elapsed:.1f} sec")
    if train_elapsed > 0 and actual_text_count is not None and total_batches is not None:
        print(f"Average speed: {actual_text_count / train_elapsed:,.1f} texts/sec")
        print(f"Batch throughput: {total_batches / train_elapsed:,.2f} batches/sec")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
