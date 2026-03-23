"""
Build a hybrid BPE tokenizer by transplanting donor tokens into a base tokenizer.

This is intended for "Ayayay-style" tokenizer surgery:
- keep the base tokenizer's special tokens and general wrapper behavior
- replace a configurable set of base token IDs with donor-side tokens
- preserve as much of the base vocabulary as possible

Important:
- This only works safely when the base and donor tokenizers are from the same
  tokenizer family (same pre-tokenizer / normalizer / BPE settings).
- If they are not structurally compatible, the script fails unless --force is used.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: Missing required package. Please install dependencies:")
    print("  pip install transformers")
    sys.exit(1)


_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_TOKENIZER_COPY_FILES = [
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]


@dataclass
class TokenizerArtifacts:
    source: str
    artifact_dir: Path
    tokenizer_json_path: Path
    data: Dict


class HybridTokenizerError(RuntimeError):
    """Raised when the tokenizer surgery cannot be performed safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a hybrid BPE tokenizer by transplanting donor tokens into a base tokenizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-tokenizer",
        required=True,
        help="Base tokenizer path or Hugging Face model/tokenizer ID.",
    )
    parser.add_argument(
        "--donor-tokenizer",
        required=True,
        help="Donor tokenizer path or Hugging Face model/tokenizer ID.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the hybrid tokenizer will be written.",
    )
    parser.add_argument(
        "--replace-count",
        type=int,
        help="Replace this many highest non-special base token IDs.",
    )
    parser.add_argument(
        "--replace-tail-start-id",
        type=int,
        help="Replace non-special base token IDs greater than or equal to this value.",
    )
    parser.add_argument(
        "--replace-existing-cyrillic",
        action="store_true",
        help="Also free non-special base IDs whose tokens already contain Cyrillic.",
    )
    parser.add_argument(
        "--cyrillic-only",
        action="store_true",
        help="Only consider donor top-level candidates containing Cyrillic.",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=2,
        help="Minimum character length for donor top-level candidate tokens.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow surgery even when compatibility checks fail. Use only if you know the tokenizer families match in practice.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading Hugging Face tokenizers.",
    )
    args = parser.parse_args()

    if args.replace_count is None and args.replace_tail_start_id is None:
        parser.error("Provide either --replace-count or --replace-tail-start-id.")

    if args.replace_count is not None and args.replace_count <= 0:
        parser.error("--replace-count must be > 0.")

    if args.replace_tail_start_id is not None and args.replace_tail_start_id < 0:
        parser.error("--replace-tail-start-id must be >= 0.")

    return args


def load_tokenizer_artifacts(spec: str, temp_root: Path, trust_remote_code: bool = False) -> TokenizerArtifacts:
    local_path = Path(spec)

    if local_path.exists():
        if local_path.is_dir():
            artifact_dir = local_path
            tokenizer_json_path = artifact_dir / "tokenizer.json"
        else:
            tokenizer_json_path = local_path
            artifact_dir = local_path.parent

        if not tokenizer_json_path.exists():
            raise HybridTokenizerError(f"tokenizer.json not found for local tokenizer: {spec}")
    else:
        artifact_dir = temp_root / re.sub(r"[^A-Za-z0-9._-]+", "_", spec)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(
            spec,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        tokenizer.save_pretrained(artifact_dir)
        tokenizer_json_path = artifact_dir / "tokenizer.json"
        if not tokenizer_json_path.exists():
            raise HybridTokenizerError(
                f"Downloaded tokenizer '{spec}' does not provide tokenizer.json. "
                "Fast-tokenizer artifacts are required."
            )

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return TokenizerArtifacts(
        source=spec,
        artifact_dir=artifact_dir,
        tokenizer_json_path=tokenizer_json_path,
        data=data,
    )


def normalize_merges(merges: Sequence) -> List[Tuple[str, str]]:
    normalized: List[Tuple[str, str]] = []
    for merge in merges:
        if isinstance(merge, list) and len(merge) == 2:
            normalized.append((merge[0], merge[1]))
        elif isinstance(merge, str):
            parts = merge.split()
            if len(parts) != 2:
                raise HybridTokenizerError(f"Unsupported merge entry: {merge!r}")
            normalized.append((parts[0], parts[1]))
        else:
            raise HybridTokenizerError(f"Unsupported merge entry type: {merge!r}")
    return normalized


def tokenize_component_signature(data: Dict) -> Dict:
    model = data.get("model", {})
    continuing_subword_prefix = model.get("continuing_subword_prefix")
    end_of_word_suffix = model.get("end_of_word_suffix")

    # Aya-family tokenizers can serialize these as null in the base tokenizer and
    # as empty strings after train_new_from_iterator(); treat them as equivalent.
    if continuing_subword_prefix == "":
        continuing_subword_prefix = None
    if end_of_word_suffix == "":
        end_of_word_suffix = None

    return {
        "model_type": model.get("type"),
        "byte_fallback": model.get("byte_fallback"),
        "continuing_subword_prefix": continuing_subword_prefix,
        "end_of_word_suffix": end_of_word_suffix,
        "unk_token": model.get("unk_token"),
        "normalizer": data.get("normalizer"),
        "pre_tokenizer": data.get("pre_tokenizer"),
        "post_processor": data.get("post_processor"),
        "decoder": data.get("decoder"),
    }


def compatibility_report(base_data: Dict, donor_data: Dict) -> List[str]:
    base_sig = tokenize_component_signature(base_data)
    donor_sig = tokenize_component_signature(donor_data)

    mismatches: List[str] = []
    for key in base_sig:
        if base_sig[key] != donor_sig[key]:
            mismatches.append(key)
    return mismatches


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    inverse: Dict[int, str] = {}
    for token, token_id in vocab.items():
        inverse[token_id] = token
    return inverse


def get_special_token_ids(data: Dict, vocab: Dict[str, int]) -> Set[int]:
    special_ids: Set[int] = set()

    for token_info in data.get("added_tokens", []):
        if token_info.get("special") is True:
            token_id = token_info.get("id")
            if isinstance(token_id, int):
                special_ids.add(token_id)
            token_str = token_info.get("content")
            if token_str in vocab:
                special_ids.add(vocab[token_str])

    return special_ids


def contains_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text))


def choose_replace_ids(
    base_vocab_by_id: Dict[int, str],
    special_ids: Set[int],
    replace_count: Optional[int],
    replace_tail_start_id: Optional[int],
    replace_existing_cyrillic: bool,
) -> List[int]:
    eligible_tail_ids: List[int] = []
    if replace_tail_start_id is not None:
        eligible_tail_ids = [
            token_id
            for token_id in sorted(base_vocab_by_id)
            if token_id >= replace_tail_start_id and token_id not in special_ids
        ]
    elif replace_count is not None:
        non_special_ids = [
            token_id
            for token_id in sorted(base_vocab_by_id)
            if token_id not in special_ids
        ]
        eligible_tail_ids = non_special_ids[-replace_count:]

    if replace_existing_cyrillic:
        cyrillic_ids = [
            token_id
            for token_id, token in base_vocab_by_id.items()
            if token_id not in special_ids and contains_cyrillic(token)
        ]
        eligible_tail_ids = sorted(set(eligible_tail_ids) | set(cyrillic_ids))

    return sorted(eligible_tail_ids)


def build_merge_dependency_map(merges: Sequence[Tuple[str, str]], vocab: Dict[str, int]) -> Dict[str, Tuple[str, str, int]]:
    dependency_map: Dict[str, Tuple[str, str, int]] = {}
    for idx, (left, right) in enumerate(merges):
        merged = left + right
        if merged in vocab and merged not in dependency_map:
            dependency_map[merged] = (left, right, idx)
    return dependency_map


def compute_token_closure(
    token: str,
    dependency_map: Dict[str, Tuple[str, str, int]],
    preserved_tokens: Set[str],
    cache: Dict[str, Tuple[Set[str], Set[int]]],
) -> Tuple[Set[str], Set[int]]:
    if token in preserved_tokens:
        return set(), set()
    if token in cache:
        cached_tokens, cached_merges = cache[token]
        return set(cached_tokens), set(cached_merges)

    if token not in dependency_map:
        result = ({token}, set())
        cache[token] = (set(result[0]), set(result[1]))
        return result

    left, right, merge_idx = dependency_map[token]
    tokens_left, merges_left = compute_token_closure(left, dependency_map, preserved_tokens, cache)
    tokens_right, merges_right = compute_token_closure(right, dependency_map, preserved_tokens, cache)

    required_tokens = tokens_left | tokens_right | {token}
    required_merges = merges_left | merges_right | {merge_idx}
    cache[token] = (set(required_tokens), set(required_merges))
    return required_tokens, required_merges


def select_donor_tokens(
    donor_vocab: Dict[str, int],
    donor_special_ids: Set[int],
    dependency_map: Dict[str, Tuple[str, str, int]],
    preserved_base_tokens: Set[str],
    base_tokens_all: Set[str],
    replace_slot_count: int,
    cyrillic_only: bool,
    min_token_length: int,
) -> Tuple[Set[str], Set[int], List[str], List[str]]:
    closure_cache: Dict[str, Tuple[Set[str], Set[int]]] = {}
    selected_tokens: Set[str] = set()
    selected_merge_ids: Set[int] = set()
    selected_top_level_tokens: List[str] = []
    skipped_due_to_capacity: List[str] = []

    donor_candidates = sorted(donor_vocab.items(), key=lambda item: item[1])
    for token, token_id in donor_candidates:
        if token_id in donor_special_ids:
            continue
        if token in base_tokens_all:
            continue
        if len(token) < min_token_length:
            continue
        if cyrillic_only and not contains_cyrillic(token):
            continue

        closure_tokens, closure_merges = compute_token_closure(
            token=token,
            dependency_map=dependency_map,
            preserved_tokens=preserved_base_tokens,
            cache=closure_cache,
        )

        new_tokens = closure_tokens - selected_tokens
        if not new_tokens:
            continue

        if len(selected_tokens) + len(new_tokens) > replace_slot_count:
            skipped_due_to_capacity.append(token)
            continue

        selected_tokens |= closure_tokens
        selected_merge_ids |= closure_merges
        selected_top_level_tokens.append(token)

        if len(selected_tokens) == replace_slot_count:
            break

    return selected_tokens, selected_merge_ids, selected_top_level_tokens, skipped_due_to_capacity


def filter_base_merges(
    base_merges: Sequence[Tuple[str, str]],
    allowed_tokens: Set[str],
) -> List[List[str]]:
    filtered: List[List[str]] = []
    for left, right in base_merges:
        merged = left + right
        if left in allowed_tokens and right in allowed_tokens and merged in allowed_tokens:
            filtered.append([left, right])
    return filtered


def donor_merges_from_ids(
    donor_merges: Sequence[Tuple[str, str]],
    selected_merge_ids: Set[int],
    allowed_tokens: Set[str],
) -> List[List[str]]:
    donor_section: List[List[str]] = []
    for idx, (left, right) in enumerate(donor_merges):
        if idx not in selected_merge_ids:
            continue
        merged = left + right
        if left in allowed_tokens and right in allowed_tokens and merged in allowed_tokens:
            donor_section.append([left, right])
    return donor_section


def copy_sidecar_files(base_artifacts: TokenizerArtifacts, output_dir: Path) -> None:
    for name in _TOKENIZER_COPY_FILES:
        src = base_artifacts.artifact_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hybrid_tokenizer_") as temp_dir_name:
        temp_root = Path(temp_dir_name)

        base_artifacts = load_tokenizer_artifacts(
            args.base_tokenizer,
            temp_root / "base",
            trust_remote_code=args.trust_remote_code,
        )
        donor_artifacts = load_tokenizer_artifacts(
            args.donor_tokenizer,
            temp_root / "donor",
            trust_remote_code=args.trust_remote_code,
        )

        mismatches = compatibility_report(base_artifacts.data, donor_artifacts.data)
        if mismatches and not args.force:
            raise HybridTokenizerError(
                "Base and donor tokenizer families do not match. "
                f"Mismatched components: {', '.join(mismatches)}. "
                "Train the donor tokenizer in the base tokenizer's format first, or rerun with --force."
            )

        base_model = base_artifacts.data.get("model", {})
        donor_model = donor_artifacts.data.get("model", {})
        if base_model.get("type") != "BPE" or donor_model.get("type") != "BPE":
            raise HybridTokenizerError("This script currently supports BPE tokenizers only.")

        base_vocab: Dict[str, int] = base_model["vocab"]
        donor_vocab: Dict[str, int] = donor_model["vocab"]
        base_vocab_by_id = invert_vocab(base_vocab)
        donor_vocab_by_id = invert_vocab(donor_vocab)

        base_special_ids = get_special_token_ids(base_artifacts.data, base_vocab)
        donor_special_ids = get_special_token_ids(donor_artifacts.data, donor_vocab)

        replace_ids = choose_replace_ids(
            base_vocab_by_id=base_vocab_by_id,
            special_ids=base_special_ids,
            replace_count=args.replace_count,
            replace_tail_start_id=args.replace_tail_start_id,
            replace_existing_cyrillic=args.replace_existing_cyrillic,
        )
        if not replace_ids:
            raise HybridTokenizerError("No replaceable base token IDs were selected.")

        preserved_base_tokens = {
            token
            for token_id, token in base_vocab_by_id.items()
            if token_id not in replace_ids
        }
        all_base_tokens = set(base_vocab.keys())

        base_merges = normalize_merges(base_model.get("merges", []))
        donor_merges = normalize_merges(donor_model.get("merges", []))
        donor_dependency_map = build_merge_dependency_map(donor_merges, donor_vocab)

        selected_tokens, selected_merge_ids, selected_top_level_tokens, skipped_due_to_capacity = select_donor_tokens(
            donor_vocab=donor_vocab,
            donor_special_ids=donor_special_ids,
            dependency_map=donor_dependency_map,
            preserved_base_tokens=preserved_base_tokens,
            base_tokens_all=all_base_tokens,
            replace_slot_count=len(replace_ids),
            cyrillic_only=args.cyrillic_only,
            min_token_length=args.min_token_length,
        )
        if not selected_tokens:
            raise HybridTokenizerError(
                "No donor tokens could be selected. "
                "Either there were no compatible candidates or the replace budget was too small."
            )

        selected_tokens_sorted = sorted(selected_tokens, key=lambda token: donor_vocab[token])
        used_replace_ids = replace_ids[: len(selected_tokens_sorted)]
        filler_replace_ids = replace_ids[len(selected_tokens_sorted) :]

        final_vocab_by_id: Dict[int, str] = {
            token_id: token
            for token_id, token in base_vocab_by_id.items()
            if token_id not in used_replace_ids
        }

        donor_assignment = dict(zip(used_replace_ids, selected_tokens_sorted))
        for token_id, token in donor_assignment.items():
            final_vocab_by_id[token_id] = token

        final_vocab_set = set(final_vocab_by_id.values())
        final_vocab: Dict[str, int] = {
            token: token_id
            for token_id, token in sorted(final_vocab_by_id.items())
        }

        base_merge_section = filter_base_merges(base_merges, final_vocab_set)
        donor_merge_section = donor_merges_from_ids(donor_merges, selected_merge_ids, final_vocab_set)

        hybrid_data = json.loads(json.dumps(base_artifacts.data))
        hybrid_data["model"]["vocab"] = final_vocab
        hybrid_data["model"]["merges"] = base_merge_section + donor_merge_section

        tokenizer_out_path = output_dir / "tokenizer.json"
        with open(tokenizer_out_path, "w", encoding="utf-8") as f:
            json.dump(hybrid_data, f, ensure_ascii=False, indent=2)

        copy_sidecar_files(base_artifacts, output_dir)

        merge_info = {
            "base_tokenizer": args.base_tokenizer,
            "donor_tokenizer": args.donor_tokenizer,
            "compatibility_mismatches": mismatches,
            "used_force": bool(args.force),
            "replace_count_requested": args.replace_count,
            "replace_tail_start_id": args.replace_tail_start_id,
            "replace_existing_cyrillic": bool(args.replace_existing_cyrillic),
            "cyrillic_only": bool(args.cyrillic_only),
            "min_token_length": args.min_token_length,
            "replace_slots_total": len(replace_ids),
            "replace_slots_used": len(used_replace_ids),
            "replace_slots_unused": len(filler_replace_ids),
            "selected_top_level_tokens_count": len(selected_top_level_tokens),
            "selected_top_level_tokens_sample": selected_top_level_tokens[:100],
            "selected_donor_tokens_count": len(selected_tokens_sorted),
            "selected_donor_tokens_sample": selected_tokens_sorted[:250],
            "replaced_base_ids": used_replace_ids,
            "replaced_base_tokens": [
                {
                    "id": token_id,
                    "old_token": base_vocab_by_id[token_id],
                    "new_token": donor_assignment[token_id],
                }
                for token_id in used_replace_ids
            ],
            "kept_replaceable_base_ids": [
                {
                    "id": token_id,
                    "token": base_vocab_by_id[token_id],
                }
                for token_id in filler_replace_ids
            ],
            "donor_merge_count_added": len(donor_merge_section),
            "base_merge_count_kept": len(base_merge_section),
            "skipped_due_to_capacity_sample": skipped_due_to_capacity[:100],
            "final_vocab_size": len(final_vocab),
            "base_vocab_size": len(base_vocab),
            "donor_vocab_size": len(donor_vocab),
        }
        with open(output_dir / "merge_info.json", "w", encoding="utf-8") as f:
            json.dump(merge_info, f, ensure_ascii=False, indent=2)

        print(f"Wrote hybrid tokenizer to: {tokenizer_out_path}")
        print(f"Wrote merge report to: {output_dir / 'merge_info.json'}")
        print(
            "Replaced "
            f"{len(used_replace_ids)} base IDs with {len(selected_tokens_sorted)} donor-derived tokens "
            f"(top-level candidates selected: {len(selected_top_level_tokens)})."
        )

        if mismatches:
            print("Compatibility mismatches were ignored because --force was used:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HybridTokenizerError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
