# Dockerfile — TurboQuantKV Cache Benchmark (Conda version)

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    gcc \
    g++ \
    libnuma-dev \
    libtcmalloc-minimal4 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Miniconda ─────────────────────────────────────────────────
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# ── Create conda environment ──────────────────────────────────────────
RUN conda create -y -n vllm-cpu python=3.12 && \
    conda clean -afy

# ── Install Python dependencies inside env ────────────────────────────
RUN conda run -n vllm-cpu pip install --upgrade pip && \
    conda run -n vllm-cpu pip install \
        vllm-cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    conda run -n vllm-cpu pip install \
        pyyaml psutil matplotlib numpy huggingface-hub transformers

# ── Copy project files ────────────────────────────────────────────────
COPY scripts/ scripts/
COPY params/ params/

# ── Runtime env vars ──────────────────────────────────────────────────
ENV VLLM_CPU_KVCACHE_SPACE=4
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=4

# ── Create results dir ────────────────────────────────────────────────
RUN mkdir -p /app/results

# ── Default command (runs inside conda env) ───────────────────────────
CMD ["conda", "run", "--no-capture-output", "-n", "vllm-cpu", \
     "python", "scripts/main.py", \
     "--config", "scriptsconfig.yaml", \
     "--prompts", "scripts/prompts.json", \
     "--output", "results/results.json"]