# Dockerfile — TurboQuantKV Cache Benchmark
# Usage:
#   docker build -t turboquant-bench .
#   docker run --rm -v $(pwd)/results:/app/results turboquant-bench

FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-12 \
    g++-12 \
    libnuma-dev \
    libtcmalloc-minimal4 \
    wget \
    git \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 \
       --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install vLLM CPU wheel + dependencies ────────────────────────────
# Done before copying source so this layer is cached on rebuilds
RUN pip install --upgrade pip --quiet && \
    pip install vllm-cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --quiet && \
    pip install pyyaml psutil requests matplotlib numpy huggingface-hub transformers \
        --quiet

# ── Pre-download the model into the image ────────────────────────────
# Remove this block if you'd rather mount a cache volume instead.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0', \
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*'] \
)"

# ── Copy project files ────────────────────────────────────────────────
COPY main.py config.yaml prompts.json ./
COPY generate_plots.py ./
# Add any other scripts here

# ── Runtime environment ───────────────────────────────────────────────
ENV VLLM_CPU_KVCACHE_SPACE=4
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=4

# Results directory (mount a volume here to get files out)
RUN mkdir -p /app/results

# ── Default command ───────────────────────────────────────────────────
CMD ["python3", "main.py", "--output", "results/kv_cache_results.json"]
