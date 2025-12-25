# Setup script for vast.ai
# Usage: curl -sSL https://raw.githubusercontent.com/<user>/ukr-tokenizer/main/setup_vast.sh | bash
# Or: ./setup_vast.sh [git-repo-url]

set -e

# Default repo - change this to your repo
DEFAULT_REPO="https://github.com/<your-username>/ukr-tokenizer.git"
REPO_URL="${1:-$DEFAULT_REPO}"
WORK_DIR="ukr-tokenizer"

echo "=== Setting up BPE Tokenizer Training Environment ==="

# Update pip
pip install --upgrade pip

# Install dependencies
pip install datasets tokenizers tqdm

# Clone repo if URL provided and doesn't contain placeholder
if [[ "$REPO_URL" != *"<"* ]]; then
    echo ""
    echo "=== Cloning repository ==="
    if [ -d "$WORK_DIR" ]; then
        echo "Directory $WORK_DIR exists, pulling latest..."
        cd "$WORK_DIR"
        git pull
    else
        git clone "$REPO_URL" "$WORK_DIR"
        cd "$WORK_DIR"
    fi
    chmod +x *.sh 2>/dev/null || true
fi

# Verify installation
python -c "from datasets import load_dataset; from tokenizers import Tokenizer; print('✓ All dependencies installed successfully')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Run training with:"
echo "  cd $WORK_DIR"
echo "  python create_bpe_pairs.py --samples 100000 --vocab-size 32000"
echo ""
echo "Or use the runner script:"
echo "  ./run_vast.sh 500000 32000"
echo ""

