"""
Compare multiple tokenizer evaluation reports (JSON format).

This utility helps you compare fertility and OOV metrics across different
tokenizer versions to identify improvements.

Usage:
    python compare_reports.py report_v1.json report_v2.json report_v3.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_report(path: str) -> Dict:
    """Load evaluation report from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Report file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in report: {path}")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare multiple tokenizer evaluation reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "reports",
        nargs="+",
        help="Paths to evaluation report JSON files",
    )

    return parser.parse_args()


def format_label(path: str) -> str:
    """Create a short label from file path."""
    return Path(path).stem


def print_comparison(reports_data: List[tuple[str, Dict]]) -> None:
    """Print formatted comparison table."""

    print("\n" + "=" * 100)
    print("TOKENIZER EVALUATION COMPARISON")
    print("=" * 100)

    # Prepare data
    labels = [format_label(path) for path, _ in reports_data]
    max_label_len = max(len(label) for label in labels)

    # Token Fertility Comparison
    print(f"\n{'-' * 100}")
    print("TOKEN FERTILITY (tokens per word)")
    print(f"{'-' * 100}")

    print(f"\n{'Metric':<35} {' '.join(f'{label:>15}' for label in labels)}")
    print("-" * (35 + len(labels) * 16))

    metrics = [
        ("Mean tokens/word", "fertility", "mean_tokens_per_word"),
        ("Median tokens/word", "fertility", "median_tokens_per_word"),
        ("Std Dev", "fertility", "std_tokens_per_word"),
        ("p50", "fertility", "p50_tokens_per_word"),
        ("p75", "fertility", "p75_tokens_per_word"),
        ("p95", "fertility", "p95_tokens_per_word"),
        ("p99", "fertility", "p99_tokens_per_word"),
        ("Min tokens/word", "fertility", "min_tokens_per_word"),
        ("Max tokens/word", "fertility", "max_tokens_per_word"),
    ]

    for display_name, section, key in metrics:
        values = []
        for _, report in reports_data:
            try:
                value = report[section][key]
                if isinstance(value, float):
                    values.append(f"{value:>15.3f}")
                else:
                    values.append(f"{value:>15}")
            except (KeyError, TypeError):
                values.append(f"{'N/A':>15}")

        print(f"{display_name:<35} {' '.join(values)}")

    # OOV Comparison
    print(f"\n{'-' * 100}")
    print("OUT-OF-VOCABULARY (OOV) COVERAGE")
    print(f"{'-' * 100}")

    print(f"\n{'Metric':<35} {' '.join(f'{label:>15}' for label in labels)}")
    print("-" * (35 + len(labels) * 16))

    oov_metrics = [
        ("Total tokens", "oov", "total_tokens"),
        ("UNK tokens", "oov", "unk_count"),
        ("OOV fraction", "oov", "oov_fraction"),
        ("OOV percentage (%)", "oov", "oov_percentage"),
    ]

    for display_name, section, key in oov_metrics:
        values = []
        for _, report in reports_data:
            try:
                value = report[section][key]
                if isinstance(value, float):
                    if "percentage" in key:
                        values.append(f"{value:>15.3f}")
                    else:
                        values.append(f"{value:>15.6f}")
                else:
                    values.append(f"{value:>15,}")
            except (KeyError, TypeError):
                values.append(f"{'N/A':>15}")

        print(f"{display_name:<35} {' '.join(values)}")

    # Metadata
    print(f"\n{'-' * 100}")
    print("METADATA")
    print(f"{'-' * 100}")

    print(f"\n{'Property':<35} {' '.join(f'{label:>15}' for label in labels)}")
    print("-" * (35 + len(labels) * 16))

    metadata_fields = [
        ("Corpus", "metadata", "corpus_type"),
        ("Samples", "metadata", "samples_evaluated"),
    ]

    for display_name, section, key in metadata_fields:
        values = []
        for _, report in reports_data:
            try:
                value = report[section][key]
                if isinstance(value, int):
                    values.append(f"{value:>15,}")
                else:
                    values.append(f"{value:>15}")
            except (KeyError, TypeError):
                values.append(f"{'N/A':>15}")

        print(f"{display_name:<35} {' '.join(values)}")

    # Summary
    print(f"\n{'-' * 100}")
    print("RANKING")
    print(f"{'-' * 100}")

    # Get data for ranking
    fertility_scores = []
    oov_scores = []

    for i, (path, report) in enumerate(reports_data):
        fertility = report["fertility"]["mean_tokens_per_word"]
        oov = report["oov"]["oov_percentage"]
        fertility_scores.append((labels[i], fertility))
        oov_scores.append((labels[i], oov))

    # Sort by metric (lower is better)
    fertility_scores.sort(key=lambda x: x[1])
    oov_scores.sort(key=lambda x: x[1])

    print("\nBest Fertility (tokens/word) - LOWER IS BETTER:")
    for rank, (label, score) in enumerate(fertility_scores, 1):
        status = "✓" if score < 2.0 else "~" if score < 2.5 else "✗"
        print(f"  {rank}. {label:<20} {score:.3f} {status}")

    print("\nBest OOV Coverage - LOWER IS BETTER:")
    for rank, (label, score) in enumerate(oov_scores, 1):
        status = "✓" if score < 0.5 else "~" if score < 1.0 else "✗"
        print(f"  {rank}. {label:<20} {score:.3f}% {status}")

    # Overall winner
    print(f"\n{'-' * 100}")
    overall_winner = None
    best_combined_score = float('inf')

    for label, fertility in fertility_scores:
        oov = next(o for l, o in oov_scores if l == label)
        # Normalize: fertility goal is 2.0, OOV goal is 1.0
        combined = (fertility / 2.0) + (oov / 1.0)
        if combined < best_combined_score:
            best_combined_score = combined
            overall_winner = label

    if overall_winner:
        print(f"Overall Winner: {overall_winner}")
        print(f"Best balance of fertility and OOV coverage")

    print("=" * 100 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    if len(args.reports) < 2:
        print("Error: Please provide at least 2 report files to compare")
        sys.exit(1)

    # Load all reports
    reports_data = []
    for report_path in args.reports:
        report = load_report(report_path)
        reports_data.append((report_path, report))

    # Print comparison
    print_comparison(reports_data)


if __name__ == "__main__":
    main()

