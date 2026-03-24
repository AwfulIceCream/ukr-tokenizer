# Ukrainian BPE Tokenizer — Evaluation Toolkit

A small toolkit to benchmark your **Ukrainian BPE tokenizer** against popular **Hugging Face** tokenizers.

It measures:
- **Token fertility** (tokens per word; mean + percentiles)
- **UNK/OOV rate**
- **Byte fallback rate** (when detectable)
- **Round-trip fidelity** (`decode(encode(x))` exact match under the same normalization)

Supports:
- **Kobza corpus** (`Goader/kobza`, streaming)
- **Your own text file** (UTF-8, one sample per line)

## Project contents
- `evaluate_tokenizer.py` — evaluate one tokenizer (local or HF) and save a JSON report
- `compare_tokenizers.py` — compare multiple tokenizers, print a ranked table, save a JSON report
- `compare_reports.py` — compare multiple saved JSON reports
- `prepare_kobza_corpus.py` — download Kobza locally, preprocess it once, and export a fast local text file for repeated runs
- `train_aya_donor.py` — train an Aya-compatible donor tokenizer with the same tokenizer family as the base Aya tokenizer
- `build_hybrid_tokenizer.py` — transplant donor BPE tokens into a base tokenizer when both tokenizers share the same tokenizer-family settings
- `tokenizer_utils.py` — shared utilities (normalization, corpus loading, tokenization)
- `ukr_bpe_tokenizer/` — local tokenizer files (`tokenizer.json`, merges/vocab, etc.)

## How to run

### 1) Setup
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
````

### 2) Evaluate one tokenizer

Local tokenizer (default: `ukr_bpe_tokenizer/tokenizer.json`):

```bash
python evaluate_tokenizer.py --samples 1000
python evaluate_tokenizer.py --samples 10000 --output-report eval_local.json
```

Hugging Face tokenizer:

```bash
python evaluate_tokenizer.py --model-id meta-llama/Llama-3.1-8B-Instruct --samples 10000 --output-report llama.json
```

Your own corpus file:

```bash
python evaluate_tokenizer.py --corpus file --input-file data/news_uk.txt --samples 5000 --output-report eval_file.json
```

### 3) Compare multiple tokenizers

```bash
python compare_tokenizers.py --samples 10000 --output-report comparison_results.json
```

Compare on your own file:

```bash
python compare_tokenizers.py --samples 10000 --input-file data/news_uk.txt --output-report comparison_results.json
```

### 4) Compare saved reports

```bash
python compare_reports.py eval_local.json eval_candidate.json more.json
```

### 5) Build a hybrid tokenizer

This is the workflow to make an existing tokenizer more Ukrainian-friendly without
throwing away its special tokens and wrapper configuration.

Example:

```bash
python build_hybrid_tokenizer.py ^
  --base-tokenizer some-org/some-base-tokenizer ^
  --donor-tokenizer ukr_bpe_128k_full ^
  --output-dir hybrid_tokenizer ^
  --replace-count 20000 ^
  --cyrillic-only
```

For large byte-level tokenizers such as Aya, use a more controlled dry run first:

```bash
python build_hybrid_tokenizer.py ^
  --base-tokenizer CohereLabs/aya-expanse-8b ^
  --donor-tokenizer aya_uk_donor ^
  --output-dir aya_uk_hybrid_plan ^
  --replace-tail-start-id 220000 ^
  --replace-existing-cyrillic ^
  --max-top-level-tokens 20000 ^
  --max-closure-size 16 ^
  --donor-id-max 180000 ^
  --dry-run
```

The script writes:
- `hybrid_tokenizer/tokenizer.json`
- `hybrid_tokenizer/merge_info.json`

Important limitation:
- This only works safely when the donor tokenizer was trained with the same
  tokenizer-family settings as the base tokenizer: pre-tokenizer, normalizer,
  decoder, post-processor, and BPE options.
- Your current local tokenizer is `Metaspace + BPE`. That can be transplanted
  directly only into another `Metaspace + BPE` tokenizer with matching settings.
- If you want an Aya-style result for a byte-level tokenizer, first train a
  Ukrainian donor tokenizer in the Aya tokenizer format, then run this script.
- For byte-level tokenizers, `--cyrillic-only` is usually the wrong filter,
  because Ukrainian bytes are often stored as byte-level symbols rather than
  literal Cyrillic characters in `vocab`.
- Use `--max-top-level-tokens`, `--max-closure-size`, `--donor-id-min`,
  `--donor-id-max`, and `--dry-run` to make large vocab surgery more controlled.

### 6) Train an Aya-compatible donor tokenizer

Train a donor tokenizer from the Aya tokenizer family on Ukrainian text:

For the fastest repeated workflow, first export Kobza to a local file:

```bash
python prepare_kobza_corpus.py ^
  --samples 100000 ^
  --output-file data/kobza_uk.txt
```

Then train from that local file instead of remote streaming:

```bash
python train_aya_donor.py ^
  --base-tokenizer CohereLabs/aya-expanse-8b ^
  --corpus file ^
  --input-file data/kobza_uk.txt ^
  --samples 100000 ^
  --output aya_uk_donor ^
  --trust-remote-code
```

Then use that donor in the hybrid build step:

```bash
python build_hybrid_tokenizer.py ^
  --base-tokenizer CohereLabs/aya-expanse-8b ^
  --donor-tokenizer aya_uk_donor ^
  --output-dir aya_uk_hybrid ^
  --replace-tail-start-id 150000 ^
  --replace-existing-cyrillic ^
  --cyrillic-only
```

## Notes

* First run will stream Kobza and download HF tokenizers.
* Some HF models may be gated. If needed:

  * `hf auth login`, or set `HUGGINGFACE_HUB_TOKEN`.

```
::contentReference[oaicite:0]{index=0}
```
