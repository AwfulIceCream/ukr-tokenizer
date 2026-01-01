# compare_tokenizers.py
"""
Compare your Ukrainian BPE tokenizer against state-of-the-art LLM tokenizers.

Evaluates:
- Fertility (tokens/word, whitespace words) + percentiles
- UNK fraction (if tokenizer has unk token)
- Byte fallback fraction (SentencePiece <0xAB> tokens, when detectable)
- Round-trip fidelity: decode(encode(x)) == x (with agreed normalization)
"""

import argparse
import json
import sys
from typing import Dict, List

try:
    from tqdm import tqdm
    import numpy as np
except ImportError:
    print("Error: Missing required package. Please install dependencies:")
    print("  pip install datasets tokenizers transformers tqdm numpy")
    sys.exit(1)

from tokenizer_utils import (
    load_hf_tokenizer,
    load_local_tokenizer,
    load_corpus_kobza,
    load_corpus_file,
    split_text_into_words,
    tokenize_text,
    decode_ids,
    preprocess_text,
    get_unk_id,
    get_byte_fallback_ids,
)

COMPARISON_MODELS = [
    "CohereForAI/aya-expanse-8b",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-3-12b-it",
    "microsoft/Phi-4-mini-instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare your tokenizer against several LLM tokenizers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples from corpus")
    parser.add_argument("--output-report", type=str, default="comparison_results.json", help="Output JSON report path")
    parser.add_argument("--input-file", type=str, help="Path to input text file (if not using Kobza corpus)")
    return parser.parse_args()


def load_corpus(samples: int, input_file: str = None) -> List[str]:
    if input_file:
        return load_corpus_file(input_file, samples)
    return load_corpus_kobza(samples)


def calculate_fertility(tokenizer, texts: List[str]) -> Dict:
    tokens_per_word_list: List[int] = []

    for text in tqdm(texts, desc="  Calculating fertility", leave=False):
        words = split_text_into_words(text)
        for word in words:
            try:
                token_ids = tokenize_text(tokenizer, word)
                tokens_per_word_list.append(len(token_ids))
            except Exception:
                continue

    if not tokens_per_word_list:
        return {
            "mean_tokens_per_word": 0.0,
            "median_tokens_per_word": 0.0,
            "std_tokens_per_word": 0.0,
            "min_tokens_per_word": 0,
            "max_tokens_per_word": 0,
            "p50_tokens_per_word": 0.0,
            "p75_tokens_per_word": 0.0,
            "p95_tokens_per_word": 0.0,
            "p99_tokens_per_word": 0.0,
            "total_words": 0,
            "total_tokens": 0,
        }

    arr = np.array(tokens_per_word_list, dtype=np.int32)

    return {
        "mean_tokens_per_word": float(np.mean(arr)),
        "median_tokens_per_word": float(np.median(arr)),
        "std_tokens_per_word": float(np.std(arr)),
        "min_tokens_per_word": int(np.min(arr)),
        "max_tokens_per_word": int(np.max(arr)),
        "p50_tokens_per_word": float(np.percentile(arr, 50)),
        "p75_tokens_per_word": float(np.percentile(arr, 75)),
        "p95_tokens_per_word": float(np.percentile(arr, 95)),
        "p99_tokens_per_word": float(np.percentile(arr, 99)),
        "total_words": int(arr.size),
        "total_tokens": int(np.sum(arr)),
    }


def calculate_coverage(tokenizer, texts: List[str]) -> Dict:
    """
    Computes:
    - UNK fraction (if unk token exists)
    - Byte fallback fraction (SentencePiece-style <0xAB>, if present)
    Fractions are among all produced tokens on the corpus.
    """
    unk_id = get_unk_id(tokenizer)
    byte_ids, byte_supported = get_byte_fallback_ids(tokenizer)

    unk_tokens = 0
    byte_fallback_tokens = 0
    total_tokens = 0

    unk_examples: List[str] = []
    byte_examples: List[str] = []

    for text in tqdm(texts, desc="  Calculating UNK/byte fallback", leave=False):
        try:
            token_ids = tokenize_text(tokenizer, text)
        except Exception:
            continue

        if not token_ids:
            continue

        total_tokens += len(token_ids)

        if unk_id is not None:
            c_unk = sum(1 for tid in token_ids if tid == unk_id)
            unk_tokens += c_unk
            if c_unk > 0 and len(unk_examples) < 10:
                unk_examples.append(text[:200])

        if byte_supported:
            c_byte = sum(1 for tid in token_ids if tid in byte_ids)
            byte_fallback_tokens += c_byte
            if c_byte > 0 and len(byte_examples) < 10:
                byte_examples.append(text[:200])

    unk_fraction = (unk_tokens / total_tokens) if (total_tokens > 0 and unk_id is not None) else 0.0
    byte_fraction = (byte_fallback_tokens / total_tokens) if (total_tokens > 0 and byte_supported) else 0.0

    return {
        "total_tokens": int(total_tokens),
        "unk_supported": bool(unk_id is not None),
        "unk_token_id": int(unk_id) if unk_id is not None else None,
        "unk_fraction": float(unk_fraction),  # 0..1
        "unk_tokens": int(unk_tokens),
        "unk_examples": unk_examples[:5],
        "byte_fallback_supported": bool(byte_supported),
        "byte_fallback_fraction": float(byte_fraction) if byte_supported else None,  # 0..1 or None
        "byte_fallback_tokens": int(byte_fallback_tokens),
        "byte_fallback_examples": byte_examples[:5],
    }


def calculate_round_trip_fidelity(tokenizer, texts: List[str]) -> Dict:
    """
    Round-trip fidelity: fraction of cases where decode(encode(x)) == x,
    under an agreed normalization policy.

    Policy used:
      x0 = preprocess_text(x)
      y  = decode(encode(x0))
      compare preprocess_text(y) == x0

    Returns:
      exact_match_rate in [0,1], counts, and up to 5 mismatch examples.
    """
    tested = 0
    exact = 0
    mismatch_examples: List[Dict[str, str]] = []
    decode_failures = 0

    for text in tqdm(texts, desc="  Calculating round-trip fidelity", leave=False):
        x0 = preprocess_text(text)
        ids = tokenize_text(tokenizer, x0)
        y = decode_ids(tokenizer, ids)

        if y is None:
            decode_failures += 1
            continue

        y0 = preprocess_text(y)
        tested += 1

        if y0 == x0:
            exact += 1
        else:
            if len(mismatch_examples) < 5:
                mismatch_examples.append(
                    {
                        "original": x0[:300],
                        "decoded": y0[:300],
                    }
                )

    rate = (exact / tested) if tested > 0 else 0.0

    return {
        "supported": True,
        "tested": int(tested),
        "exact_matches": int(exact),
        "exact_match_rate": float(rate),  # 0..1
        "decode_failures": int(decode_failures),
        "mismatch_examples": mismatch_examples,
    }


def evaluate_tokenizer(tokenizer, tokenizer_name: str, texts: List[str]) -> Dict:
    print(f"\nEvaluating {tokenizer_name}...")

    fertility = calculate_fertility(tokenizer, texts)
    coverage = calculate_coverage(tokenizer, texts)
    round_trip = calculate_round_trip_fidelity(tokenizer, texts)

    return {
        "fertility": fertility,
        "coverage": coverage,
        "round_trip": round_trip,
    }


def generate_comparison_report(results: Dict, output_path: str) -> None:
    """
    Rank by mean fertility (lower is better).
    Print UNK%, ByteFB%, and RoundTrip%.
    """
    ranking = []
    for model_name, metrics in results.items():
        if model_name == "ranking":
            continue

        fert_mean = metrics["fertility"]["mean_tokens_per_word"]
        fert_p95 = metrics["fertility"]["p95_tokens_per_word"]
        unk_frac = metrics["coverage"]["unk_fraction"]
        byte_frac = metrics["coverage"]["byte_fallback_fraction"]
        rt_rate = metrics["round_trip"]["exact_match_rate"]

        ranking.append(
            {
                "model": model_name,
                "fertility_mean": fert_mean,
                "fertility_p95": fert_p95,
                "unk_fraction": unk_frac,            # 0..1
                "byte_fallback_fraction": byte_frac, # 0..1 or None
                "round_trip_rate": rt_rate,          # 0..1
            }
        )

    ranking.sort(key=lambda x: x["fertility_mean"])
    for i, item in enumerate(ranking, 1):
        item["rank"] = i

    results["ranking"] = ranking

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 125)
    print("TOKENIZER COMPARISON RESULTS (Ranked by Mean Token Fertility: tokens/word, whitespace words)")
    print("=" * 125)
    print(f"\n{'Rank':<6} {'Model':<40} {'Fert(mean)':<12} {'Fert(p95)':<10} {'UNK%':<10} {'ByteFB%':<10} {'RTF%':<10}")
    print("-" * 125)

    for item in ranking:
        unk_pct = item["unk_fraction"] * 100.0
        byte_pct = (item["byte_fallback_fraction"] * 100.0) if item["byte_fallback_fraction"] is not None else None
        rtf_pct = item["round_trip_rate"] * 100.0

        byte_str = f"{byte_pct:,.4f}" if byte_pct is not None else "n/a"

        print(
            f"{item['rank']:<6} {item['model']:<40} "
            f"{item['fertility_mean']:<12.3f} {item['fertility_p95']:<10.2f} "
            f"{unk_pct:<10.4f} {byte_str:<10} {rtf_pct:<10.2f}"
        )

    print("\n" + "=" * 125)
    print("Legend: Lower fertility is better. UNK%/ByteFB%/RTF% are percentages over the evaluated corpus.")
    print("RTF% = round-trip fidelity: preprocess_text(decode(encode(x))) == preprocess_text(x)")
    print("=" * 125)


def main():
    args = parse_args()

    texts = load_corpus(args.samples, args.input_file)
    if not texts:
        print("Error: Could not load corpus")
        sys.exit(1)

    print(f"Loaded {len(texts):,} text samples")

    tokenizers_to_evaluate = []

    print("\nLoading tokenizers...")
    local_tok, local_name = load_local_tokenizer()
    if local_tok:
        tokenizers_to_evaluate.append((local_tok, local_name))

    for model_id in COMPARISON_MODELS:
        hf_tok, hf_name = load_hf_tokenizer(model_id)
        if hf_tok:
            tokenizers_to_evaluate.append((hf_tok, hf_name))

    if not tokenizers_to_evaluate:
        print("Error: Could not load any tokenizers")
        sys.exit(1)

    print(f"\nSuccessfully loaded {len(tokenizers_to_evaluate)} tokenizer(s)")

    print("\n" + "=" * 110)
    print(f"EVALUATING {len(tokenizers_to_evaluate)} TOKENIZERS ON {len(texts):,} SAMPLES")
    print("=" * 110)

    results: Dict[str, Dict] = {}
    for tok, name in tokenizers_to_evaluate:
        results[name] = evaluate_tokenizer(tok, name, texts)

    generate_comparison_report(results, args.output_report)


if __name__ == "__main__":
    main()
