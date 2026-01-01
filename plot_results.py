# plot_results.py
#
# Prettier charts (matplotlib only):
# - horizontal bars
# - hue gradient via colormap
# - value labels
# - subtle grid
# - PNG + SVG outputs
#
# Usage:
#   python plot_results.py --input comparison_results.json --outdir figures --sort rank

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot tokenizer comparison charts from JSON report.")
    p.add_argument("--input", "-i", type=str, default="comparison_results.json", help="Path to JSON report")
    p.add_argument("--outdir", "-o", type=str, default="figures", help="Output directory for images")
    p.add_argument("--dpi", type=int, default=220, help="PNG DPI")
    p.add_argument("--sort", choices=["rank", "fertility", "name"], default="rank", help="Sort order for plots")
    p.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name (e.g., viridis, plasma, turbo)")
    return p.parse_args()


def load_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def to_percent(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(x) * 100.0


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def collect_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    ranking = report.get("ranking", [])
    if ranking:
        for item in ranking:
            name = item["model"]
            m = report.get(name, {})
            rows.append(
                {
                    "model": name,
                    "rank": item.get("rank"),
                    "fert_mean": safe_get(m, "fertility", "mean_tokens_per_word", default=None),
                    "fert_p95": safe_get(m, "fertility", "p95_tokens_per_word", default=None),
                    "unk_pct": to_percent(safe_get(m, "coverage", "unk_fraction", default=0.0)),
                    "bytefb_pct": to_percent(safe_get(m, "coverage", "byte_fallback_fraction", default=None)),
                    "rtf_pct": to_percent(safe_get(m, "round_trip", "exact_match_rate", default=None)),
                    "rtf_tested": safe_get(m, "round_trip", "tested", default=None),
                    "rtf_decode_failures": safe_get(m, "round_trip", "decode_failures", default=None),
                }
            )
        return rows

    for name, m in report.items():
        if name == "ranking" or not isinstance(m, dict):
            continue
        rows.append(
            {
                "model": name,
                "rank": None,
                "fert_mean": safe_get(m, "fertility", "mean_tokens_per_word", default=None),
                "fert_p95": safe_get(m, "fertility", "p95_tokens_per_word", default=None),
                "unk_pct": to_percent(safe_get(m, "coverage", "unk_fraction", default=0.0)),
                "bytefb_pct": to_percent(safe_get(m, "coverage", "byte_fallback_fraction", default=None)),
                "rtf_pct": to_percent(safe_get(m, "round_trip", "exact_match_rate", default=None)),
                "rtf_tested": safe_get(m, "round_trip", "tested", default=None),
                "rtf_decode_failures": safe_get(m, "round_trip", "decode_failures", default=None),
            }
        )
    return rows


def sort_rows(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "rank":
        return sorted(rows, key=lambda r: (r["rank"] is None, r["rank"] if r["rank"] is not None else 10**9))
    if mode == "fertility":
        return sorted(rows, key=lambda r: (r["fert_mean"] is None, r["fert_mean"] if r["fert_mean"] is not None else 10**9))
    return sorted(rows, key=lambda r: r["model"])


def save_csv(rows: List[Dict[str, Any]], outpath: Path) -> None:
    fields = [
        "rank",
        "model",
        "fert_mean",
        "fert_p95",
        "unk_pct",
        "bytefb_pct",
        "rtf_pct",
        "rtf_tested",
        "rtf_decode_failures",
    ]
    with outpath.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _colors_for_n(n: int, cmap_name: str) -> List:
    cmap = mpl.colormaps.get_cmap(cmap_name)
    if n <= 1:
        return [cmap(0.6)]
    # spread across the colormap for a nice "hue" gradient
    return [cmap(i / (n - 1)) for i in range(n)]


def hbar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    xlabel: str,
    outpath_base: Path,
    dpi: int,
    cmap_name: str,
    xlim: Optional[Tuple[float, float]] = None,
    value_fmt: str = "{:.3f}",
) -> None:
    n = len(labels)
    colors = _colors_for_n(n, cmap_name)

    # Bigger height if many labels
    height = max(4.2, 0.55 * n)
    plt.figure(figsize=(11.5, height))

    # Reverse so the best (rank=1) appears at top if already rank-sorted
    labels_r = list(reversed(labels))
    values_r = list(reversed(values))
    colors_r = list(reversed(colors))

    y = list(range(n))
    bars = plt.barh(y, values_r, color=colors_r)

    plt.yticks(y, labels_r)
    plt.xlabel(xlabel)
    plt.title(title)

    # subtle gridlines
    plt.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)

    if xlim is not None:
        plt.xlim(*xlim)

    # value labels at end of each bar
    for b, v in zip(bars, values_r):
        plt.text(
            b.get_width() + (0.01 * max(values_r) if max(values_r) > 0 else 0.01),
            b.get_y() + b.get_height() / 2,
            value_fmt.format(v),
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    # Save PNG + SVG
    plt.savefig(outpath_base.with_suffix(".png"), dpi=dpi)
    plt.savefig(outpath_base.with_suffix(".svg"))
    plt.close()


def hbar_chart_optional(
    labels: List[str],
    values: List[Optional[float]],
    title: str,
    xlabel: str,
    outpath_base: Path,
    dpi: int,
    cmap_name: str,
    value_fmt: str = "{:.4f}",
) -> None:
    # Plot N/A as 0, but annotate text as "n/a"
    n = len(labels)
    colors = _colors_for_n(n, cmap_name)

    height = max(4.2, 0.55 * n)
    plt.figure(figsize=(11.5, height))

    labels_r = list(reversed(labels))
    values_r = list(reversed(values))
    colors_r = list(reversed(colors))

    vals_plot = [0.0 if v is None else float(v) for v in values_r]
    y = list(range(n))
    bars = plt.barh(y, vals_plot, color=colors_r)

    plt.yticks(y, labels_r)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)

    maxv = max(vals_plot) if vals_plot else 0.0
    pad = 0.01 * maxv if maxv > 0 else 0.01

    for b, v in zip(bars, values_r):
        label = "n/a" if v is None else value_fmt.format(float(v))
        plt.text(
            b.get_width() + pad,
            b.get_y() + b.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(outpath_base.with_suffix(".png"), dpi=dpi)
    plt.savefig(outpath_base.with_suffix(".svg"))
    plt.close()


def main() -> None:
    args = parse_args()
    inpath = Path(args.input)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    if not inpath.exists():
        raise SystemExit(f"Input JSON not found: {inpath}")

    report = load_report(inpath)
    rows = sort_rows(collect_rows(report), args.sort)

    save_csv(rows, outdir / "summary_metrics.csv")

    labels = [r["model"] for r in rows]

    fert_mean = [float(r["fert_mean"]) if r["fert_mean"] is not None else 0.0 for r in rows]
    fert_p95 = [float(r["fert_p95"]) if r["fert_p95"] is not None else 0.0 for r in rows]
    rtf_pct = [float(r["rtf_pct"]) if r["rtf_pct"] is not None else 0.0 for r in rows]
    unk_pct = [float(r["unk_pct"]) if r["unk_pct"] is not None else 0.0 for r in rows]
    bytefb_pct = [r["bytefb_pct"] for r in rows]

    # Better typography defaults
    mpl.rcParams["axes.titlepad"] = 12

    hbar_chart(
        labels,
        fert_mean,
        "Середня token fertility (токенів на слово, слова = розбиття за пробілами)",
        "токенів / слово",
        outdir / "fertility_mean",
        dpi=args.dpi,
        cmap_name=args.cmap,
        value_fmt="{:.3f}",
    )

    hbar_chart(
        labels,
        fert_p95,
        "Token fertility, 95-й перцентиль (токенів на слово)",
        "токенів / слово (p95)",
        outdir / "fertility_p95",
        dpi=args.dpi,
        cmap_name=args.cmap,
        value_fmt="{:.2f}",
    )

    hbar_chart(
        labels,
        rtf_pct,
        "Round-trip fidelity: preprocess(decode(encode(x))) == preprocess(x)",
        "RTF (%)",
        outdir / "round_trip_fidelity",
        dpi=args.dpi,
        cmap_name=args.cmap,
        xlim=(0, 100),
        value_fmt="{:.2f}",
    )

    hbar_chart(
        labels,
        unk_pct,
        "Частка UNK серед усіх токенів",
        "UNK (%)",
        outdir / "unk_percent",
        dpi=args.dpi,
        cmap_name=args.cmap,
        value_fmt="{:.4f}",
    )

    hbar_chart_optional(
        labels,
        bytefb_pct,
        "Частка byte fallback серед усіх токенів",
        "Byte fallback (%)",
        outdir / "byte_fallback_percent",
        dpi=args.dpi,
        cmap_name=args.cmap,
        value_fmt="{:.4f}",
    )

    print(f"Saved figures to: {outdir.resolve()}")
    print("Created (PNG + SVG):")
    print("  - fertility_mean.(png/svg)")
    print("  - fertility_p95.(png/svg)")
    print("  - round_trip_fidelity.(png/svg)")
    print("  - unk_percent.(png/svg)")
    print("  - byte_fallback_percent.(png/svg)")
    print("Plus:")
    print("  - summary_metrics.csv")


if __name__ == "__main__":
    main()
