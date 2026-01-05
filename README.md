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

## Notes

* First run will stream Kobza and download HF tokenizers.
* Some HF models may be gated. If needed:

  * `hf auth login`, or set `HUGGINGFACE_HUB_TOKEN`.

```
::contentReference[oaicite:0]{index=0}
```
