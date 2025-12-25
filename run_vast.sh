#!/bin/bash
# Runner script for vast.ai with sensible defaults for cloud training
# Usage: ./run_vast.sh [samples] [vocab_size]

set -e

SAMPLES=${1:-500000}
VOCAB_SIZE=${2:-32000}
OUTPUT_DIR="ukr_bpe_tokenizer"

echo "=== Starting BPE Tokenizer Training on vast.ai ==="
echo "Samples: $SAMPLES"
echo "Vocab size: $VOCAB_SIZE"
echo "Output: $OUTPUT_DIR"
echo ""

# Run training
python create_bpe_pairs.py \
    --samples "$SAMPLES" \
    --vocab-size "$VOCAB_SIZE" \
    --output "$OUTPUT_DIR"

# Create archive for easy download
echo ""
echo "=== Creating archive for download ==="
tar -czvf ukr_bpe_tokenizer.tar.gz "$OUTPUT_DIR"

echo ""
echo "=== Training Complete ==="
echo "Download your tokenizer:"
echo "  scp -P <port> root@<host>:$(pwd)/ukr_bpe_tokenizer.tar.gz ."
echo ""
ls -lh ukr_bpe_tokenizer.tar.gz

