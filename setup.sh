#!/usr/bin/env bash
# setup.sh — one-shot environment setup for TurboQuantKV benchmark
# Usage: chmod +x setup.sh && ./setup.sh
set -euo pipefail

ENV_NAME="vllm-cpu"

echo ""
echo "=========================================="
echo "  TurboQuantKV — Environment Setup"
echo "=========================================="
echo ""

# ── 1. Check AVX2 ────────────────────────────────────────────────────
echo "[1/6] Checking AVX2 support..."
if ! grep -q avx2 /proc/cpuinfo; then
    echo "ERROR: AVX2 not found. Use Docker instead:"
    echo "  docker build -t turboquant-bench . && docker run --rm turboquant-bench"
    exit 1
fi
echo "  AVX2 OK"

# ── 2. Install system deps ───────────────────────────────────────────
echo ""
echo "[2/6] Installing system dependencies..."
sudo apt-get update -q
sudo apt-get install -y -q \
    gcc-12 g++-12 libnuma-dev libtcmalloc-minimal4 curl jq git wget

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-12
echo "  gcc $(gcc --version | head -1)"

# ── 3. Install conda if missing ──────────────────────────────────────
echo ""
echo "[3/6] Checking conda..."
if ! command -v conda &>/dev/null; then
    echo "  conda not found — installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    "$HOME/miniconda3/bin/conda" init bash
    source "$HOME/.bashrc" 2>/dev/null || true
    export PATH="$HOME/miniconda3/bin:$PATH"
    echo "  Miniconda installed"
else
    echo "  conda $(conda --version) already installed"
fi

# ── 4. Create conda env ──────────────────────────────────────────────
echo ""
echo "[4/6] Creating conda environment '$ENV_NAME' (Python 3.12)..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  Environment '$ENV_NAME' already exists — skipping create"
else
    conda create -n "$ENV_NAME" python=3.12 -y -q
    echo "  Environment created"
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "  Active: $(which python)"

# ── 5. Install Python packages ───────────────────────────────────────
echo ""
echo "[5/6] Installing Python packages..."
pip install --upgrade pip -q

echo "  Installing vllm-cpu from PyPI (auto-detects AVX2/AVX512, ~1.5 GB)..."
pip install vllm-cpu -q \
    --extra-index-url https://download.pytorch.org/whl/cpu

echo "  Installing other dependencies..."
pip install pyyaml psutil requests matplotlib numpy huggingface-hub transformers -q

# Verify vLLM
VLLM_VER=$(python3 -c "import vllm; print(vllm.__version__)")
echo "  vLLM $VLLM_VER OK"

# ── 6. Download model ────────────────────────────────────────────────
echo ""
echo "[6/6] Downloading TinyLlama-1.1B-Chat-v1.0 (~2.2 GB)..."
python3 - <<'EOF'
from huggingface_hub import snapshot_download
path = snapshot_download(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)
print(f"  Cached at: {path}")
EOF

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "  Next steps:"
echo "    conda activate $ENV_NAME"
echo "    VLLM_CPU_KVCACHE_SPACE=4 python3 main.py --dry-run"
echo "    VLLM_CPU_KVCACHE_SPACE=4 python3 main.py"
echo ""
