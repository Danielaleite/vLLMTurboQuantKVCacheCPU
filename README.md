# TurboQuantKV Cache Evaluation

A simple benchmarking framework comparing Baseline and TurboQuant-simulated KVcache strategies on CPU-only vLLM inference using TinyLlama-1.1B.

> **What this does:** Runs two vLLM configurations, measures latency (E2E p50/p95/p99), throughput (tok/s), RAM usage, and output quality (Token-F1).
---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Environment Setup](#3-environment-setup)
   - 3.1 [Fix WSL2 Memory Limits](#31-fix-wsl2-memory-limits-do-this-first)
   - 3.2 [Install Miniconda](#32-install-miniconda)
   - 3.3 [Create Conda Environment](#33-create-conda-environment)
   - 3.4 [Install System Dependencies](#34-install-system-dependencies)
   - 3.5 [Install vLLM CPU Wheel](#35-install-vllm-cpu-wheel)
   - 3.6 [Install Python Dependencies](#36-install-python-dependencies)
   - 3.7 [Pre-download the Model](#37-pre-download-the-model)
   - 3.8 [Verify Installation](#38-verify-installation)
4. [Benchmark](#4-running-the-benchmark)
6. [Generating some plots](#6-generating-plots-and-tables)
7. [Understanding the Outputs](#7-understanding-the-output)
8. [Configuration Reference](#8-configuration-reference)
9. [Prompts Reference](#9-prompts-reference)
10. [How Metrics Are Calculated](#10-how-metrics-are-calculated)
11. [TurboQuant Simulation — Assumptions](#11-turboquant-simulation--assumptions)

---

## 1. Overview

[TurboQuant](https://github.com/0xSero/turboquant) is a GPU-native KV cache quantization technique in vLLM that compresses Key-Value tensors to INT8/FP8, reducing memory by 4–8x. This repo simulates its effects using CPU-available knobs and documents all assumptions made.

| Configuration | dtype | max_model_len | Memory budget | What it simulates |
|---|---|---|---|---|
| **Baseline** | float32 | 2048 | 90% | Standard vLLM, no optimization |
| **TurboQuant** | float16 | 1024 | 50% | FP16 ≈ 2x KV memory reduction; truncated context ≈ eviction pressure |

**Key result on CPU:** TurboQuant is ~20% slower and ~17% lower throughput, but uses ~14% less RAM. This is expected — FP16 has no hardware acceleration on x86 CPUs. On GPU (where INT8 Tensor Cores exist), results would invert (based on the main paper).

---

## 2. Repository Structure

```
vLLMTurboQuantKVCacheCPU
│
├── main.py          # Main benchmark runner (self-contained)
├── generate_plots.py           
├── config.yaml    # all parameters here
├── prompts.json         # All benchmark prompts - one can edit it here 
│
├── benchmark/
│   ├── metrics_eval.py             # HTTP streaming runner (measures TTFT + ITL)      
│
├── scripts/
│   ├── setup.sh              # Environment setup
│
├── results/
│   ├── figures/              # Auto-generated PNG plots│
├── docs/
│   └── assumptions.md        # Full TurboQuant simulation rationale
└── .gitignore
```

---

## 3. Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | WSL2 Ubuntu 22.04 | WSL2 Ubuntu 22.04 |
| RAM | 8 GB | 16 GB |
| CPU | Any x86_64 with AVX2 | 4+ cores |
| Disk | 5 GB free | 10 GB free |
| Python | 3.12 | 3.12 |
| conda | Any | Miniconda |
| GPU | Not required | Not required |

> **Check AVX2 support** (required by the vLLM CPU wheel):
> ```bash
> grep -m1 avx2 /proc/cpuinfo && echo "AVX2 supported" || echo "AVX2 NOT found"
> ```
> Any CPU made after 2013 will have it. If not, use the Docker path in step 4.

---

## 4. Environment Setup

### 4.1 Fix WSL2 Memory Limits (do this first)

I used WSL2, which caps RAM at 50% by default. The vLLM model load will silently get killed without this fix.

Open **Windows PowerShell** (not WSL):

```powershell
notepad "$env:USERPROFILE\.wslconfig"
```

Paste the following (adjust `memory` — if you have 16 GB, use 12):

```ini
[wsl2]
memory=12GB
swap=8GB
processors=4
```

Save, restart WSL2:

```powershell
wsl --shutdown
```
---

### 4.2 Install Miniconda

Skip this step if you already have conda or miniconda installed (`conda --version` works).

```bash
# Download installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Install silently
bash ~/miniconda.sh -b -p ~/miniconda3

# Add to shell
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Verify
conda --version
# Expected output: conda 24.x.x or similar
```

---

### 4.3 Create Conda Environment

```bash
# Create a clean Python 3.12 environment
conda create -n vllm-cpu python=3.12 -y

# Activate it
conda activate vllm-cpu

# Confirm you are in the right environment
which python
# Should show: ~/miniconda3/envs/vllm-cpu/bin/python

python --version
# Should show: Python 3.12.x
```

> From here on, **always activate this environment** before running anything:
> ```bash
> conda activate vllm-cpu
> ```

---

### 4.4 Install System Dependencies

```bash
sudo apt-get update -y
sudo apt-get install -y \
    gcc-12 \
    g++-12 \
    libnuma-dev \
    libtcmalloc-minimal4 \
    curl \
    jq \
    git

# Set gcc-12 as the default compiler (vLLM CPU requires >= 12.3)
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Verify
gcc --version
# Should show: gcc (Ubuntu 12.x.x...)
```

---

### 4.5 Install vLLM CPU Wheel

This uses the official pre-built CPU wheel — **no compilation required**. This is the step that most tutorials get wrong by trying to install the GPU wheel.

```bash
conda activate vllm-cpu

# Upgrade pip first
pip install --upgrade pip

# Install the vLLM CPU wheel directly
# This downloads ~1.5 GB and installs PyTorch CPU + vLLM in one command
pip install \
    https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cpu-cp38-abi3-manylinux_2_35_x86_64.whl \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

**Verify the installation:**

```bash
python3 -c "import vllm; print(vllm.__version__)"
# Must show: 0.8.5+cpu  (the +cpu suffix confirms it's the CPU wheel)
```

If you see a version **without** `+cpu`, you accidentally installed the GPU wheel. Fix:

```bash
pip uninstall vllm -y
# Then re-run the pip install command above
```

---

### 4.6 Install Python Dependencies

```bash
conda activate vllm-cpu
cd ~/genaiops-kvcache   # or wherever you cloned this repo

pip install \
    pyyaml \
    psutil \
    requests \
    matplotlib \
    numpy \
    huggingface-hub \
    transformers
```

---

### 4.7 Pre-download the Model

```bash
conda activate vllm-cpu

python3 -c "
from huggingface_hub import snapshot_download
print('Downloading TinyLlama-1.1B-Chat-v1.0 (~2.2 GB)...')
path = snapshot_download(
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*'],
)
print(f'Done. Cached at: {path}')
"
```

---

### 4.8 Verify Installation

Run a quick end-to-end smoke test. This loads the model and generates one response (~2 minutes on CPU):

```bash
conda activate vllm-cpu

VLLM_CPU_KVCACHE_SPACE=4 python3 -c "
from vllm import LLM, SamplingParams

print('Loading model...')
llm = LLM(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    dtype='float32',
    max_model_len=512,
    enforce_eager=True,
)
outputs = llm.generate(['What is 2 + 2?'], SamplingParams(max_tokens=20, temperature=0.0))
print('Output:', outputs[0].outputs[0].text)
print('Installation verified.')
"
```

Expected output:
```
Loading model...
Output:  4
Installation verified.
```

---

## 5. Running the Benchmark

### 5.1 Dry Run (recommended first step)

Prints exactly what will run — configs, prompts, output path — without loading the model:

```bash
conda activate vllm-cpu
cd ~/vLLMTurboQuantKVCacheCPU

python3 main.py --dry-run
```

Sample output:
```
=== DRY RUN ===

Experiment : kv_cache_eval_v1
Model      : TinyLlama/TinyLlama-1.1B-Chat-v1.0
Prompts    : 11 loaded from prompts.json
Output     : results.json

Configs to run:
  [baseline]    dtype=float32, max_model_len=2048
  [turboquant]  dtype=float16, max_model_len=1024

Prompts:
  [short_1] (short)  What is the capital of France?
  [long_1]  (long)   You are a senior software architect...
  ...
```

---

### 5.2 Full Benchmark

Runs both configs sequentially, loads the model twice, generates all 11 prompts per config. Takes **20–40 minutes** on a modern laptop CPU.

```bash
conda activate vllm-cpu
cd ~/vLLMTurboQuantKVCacheCPU
VLLM_CPU_KVCACHE_SPACE=4 python3 kv_cache_test.py
```

Progress is printed in real time:

```
14:02:01 [INFO] Loading config: experiment_config.yaml
14:02:01 [INFO] Loaded 11 prompts from prompts.json
14:02:01 [INFO] CONFIG: BASELINE
14:02:01 [INFO]   dtype: float32
14:02:01 [INFO]   max_model_len: 2048
14:04:30 [INFO]   [short_1] Simple factual question...
14:04:32 [INFO]     → 2140ms | 48 tok | 22.4 tok/s
...
```

When finished, results are saved to `results.json`.

---

### 5.3 Filtered Runs

```bash
# Short prompts only (~5 min — good for quick iteration)
VLLM_CPU_KVCACHE_SPACE=4 python3 test.py --prompt-types short

# Long prompts only
VLLM_CPU_KVCACHE_SPACE=4 python3 test.py --prompt-types long

# Multi-turn only
VLLM_CPU_KVCACHE_SPACE=4 python3 test.py --prompt-types multiturn

# Short + long, skip multi-turn
VLLM_CPU_KVCACHE_SPACE=4 python3 test.py --prompt-types short long

# Run baseline config only (skip turboquant)
VLLM_CPU_KVCACHE_SPACE=4 python3 test.py --only baseline

# Run turboquant only (if baseline already done)
VLLM_CPU_KVCACHE_SPACE=4 python3 test.py --only turboquant

# Custom output file (useful for multiple experiment runs)
VLLM_CPU_KVCACHE_SPACE=4 python3 test.py --output results/run_002.json

# Use a different prompts file entirely
VLLM_CPU_KVCACHE_SPACE=4 python3 test.py --prompts my_custom_prompts.json
```

---

## 6. Generating Plots and Tables

After `results.json` exists, run:

```bash
conda activate vllm-cpu
python3 plot_results.py

```bash
python3 plot_results.py --results results/run_002.json --output-dir results/run_002
```

---

## 7. Understanding the Output

### kv_cache_results.json

```json
{
  "experiment": { "name": "...", "model": "...", "run_at": "..." },
  "configs_run": ["baseline", "turboquant"],
  "results": {
    "baseline": {
      "e2e_p50_ms": 18240,
      "e2e_p95_ms": 38410,
      "e2e_p99_ms": 61730,
      "avg_tps": 4.1,
      "peak_ram_mb": 6840,
      "per_prompt": [ { "id": "short_1", "e2e_ms": 8200, "tokens": 48, ... } ]
    },
    "turboquant": { ... }
  },
  "quality": {
    "per_prompt": { "short_1": 0.923, "long_1": 0.712, ... },
    "avg_token_f1": 0.833
  }
}
```

### Metrics explained

| Metric | How measured | What it tells you |
|---|---|---|
| **E2E latency** | `time.perf_counter()` wall clock, request-to-response | Total user-perceived latency |
| **p50/p95/p99** | Percentiles across all prompts | p99 reveals worst-case tail latency |
| **tok/s** | `tokens_generated / elapsed_seconds` | Generation speed |
| **Peak RAM** | `psutil.virtual_memory().used` sampled during run | Memory pressure |
| **Token-F1** | Word-level F1 between baseline and TurboQuant outputs | Output similarity / quality preservation |

> **TTFT and ITL note:** The offline `LLM.generate()` API returns all tokens at once and cannot measure Time-to-First-Token or Inter-Token Latency directly. For those metrics, use `benchmark/runner.py` which calls the vLLM HTTP streaming API (`/v1/completions` with `stream=True`).

---

## 8. Configuration Reference

All experiment parameters live in `experiment_config.yaml`. Edit this file to change the model, configs, or run settings — no Python changes needed.

```yaml
experiment:
  model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  prompts_file: "prompts.json"
  output_file: "kv_cache_results.json"
  max_tokens: 200
  temperature: 0.0        # greedy — deterministic for reproducibility
  warmup_requests: 1

configs:
  baseline:
    dtype: "float32"
    max_model_len: 2048
    gpu_memory_utilization: 0.90
    swap_space: 4
    enforce_eager: true
    max_num_seqs: 4

  turboquant:
    dtype: "float16"      # ~2x KV memory reduction vs float32
    max_model_len: 1024   # simulates KV cache eviction pressure
    gpu_memory_utilization: 0.50
    swap_space: 1
    enforce_eager: true
    max_num_seqs: 4

quality:
  high_threshold: 0.80    # Token-F1 >= 0.80 → HIGH
  med_threshold: 0.60     # Token-F1 >= 0.60 → MEDIUM, else LOW

env:
  VLLM_CPU_KVCACHE_SPACE: "4"       # GB of RAM for KV cache blocks
  TOKENIZERS_PARALLELISM: "false"
  OMP_NUM_THREADS: "4"
```

---

## 9. Prompts Reference

All prompts live in `prompts.json`. Each entry has this format:

```json
{
  "my_prompt_id": {
    "type": "short",
    "note": "Human-readable description",
    "text": "The actual prompt text sent to the model"
  }
}
```

Valid types: `short`, `long`, `multiturn`.

**To add your own prompts**, edit `prompts.json` — no code changes needed. The runner reads it at runtime.

| Prompt ID | Type | Purpose |
|---|---|---|
| short_1–5 | short | Low KV footprint, tests TTFT and allocation overhead |
| long_1–3 | long | Architecture review, code review, incident post-mortem — fills KV cache |
| multiturn_1–3 | multiturn | Kubernetes tutorial, debugging session, travel planning — accumulating context |

---

## 10. Troubleshooting

**`Killed` silently during model load**
WSL2 ran out of memory. Go back to step 4.1 and increase the memory limit in `.wslconfig`, then run `wsl --shutdown` and restart.

**`TypeError: EngineArgs.__init__() got an unexpected keyword argument 'device'`**
Your vLLM version removed the `device` argument. Remove it from any call — the CPU wheel detects CPU automatically. The code in this repo does not use it.

**`No module named 'vllm._C'` or `undefined symbol`**
You installed the GPU wheel, not the CPU one. Run:
```bash
pip uninstall vllm -y
pip install https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cpu-cp38-abi3-manylinux_2_35_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cpu
```

**`AVX2 not supported`**
Your CPU is very old (pre-2013). Use the Docker path:
```bash
docker pull vllm/vllm-openai-cpu:latest-x86_64
docker run --rm -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 --shm-size=4g \
    vllm/vllm-openai-cpu:latest-x86_64 \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --enforce-eager
```

**Model download hangs or times out**
Set a longer timeout or download manually:
```bash
HF_HUB_DOWNLOAD_TIMEOUT=300 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
"
```

**`conda: command not found` after install**
```bash
source ~/.bashrc
# or restart your terminal
```

**Plots have no display / `cannot connect to X server`**
This is expected in WSL2 — the script uses `matplotlib.use("Agg")` (non-interactive backend) so plots are saved to files, never displayed. Open the PNG files in Windows Explorer: navigate to `\\wsl$\Ubuntu\home\YOUR_USER\genaiops-kvcache\results\figures\`.

---

## 11. How Metrics Are Calculated

### Latency (E2E)

```python
t0 = time.perf_counter()          # high-resolution wall clock
output = llm.generate([text], sampling_params)
t1 = time.perf_counter()

e2e_ms = (t1 - t0) * 1000        # milliseconds
```

Covers: tokenization + KV cache allocation + all prefill + all decode steps + detokenization.

### Throughput

```python
tokens_per_second = tokens_generated / (t1 - t0)
```

### RAM

```python
import psutil
peak_ram_mb = psutil.virtual_memory().used / 1024 / 1024
# Sampled after each request; peak across the run is reported
```

### Token-F1 (Quality)

Measures word-level overlap between baseline and TurboQuant outputs for the same prompt:

```python
tok_a = set(re.findall(r"\b\w+\b", baseline_output.lower()))
tok_b = set(re.findall(r"\b\w+\b", turboquant_output.lower()))
common = tok_a & tok_b
precision = len(common) / len(tok_b)
recall    = len(common) / len(tok_a)
f1 = 2 * precision * recall / (precision + recall)
```

Score of 1.0 = identical word sets. Score < 0.60 = significant divergence.

---

## 12. TurboQuant Simulation — Assumptions

Real TurboQuant is GPU-only (`--kv-cache-dtype fp8` in vLLM, requires Ampere+ GPU). Here its observable effects on CPU are simulated:

| TurboQuant Effect | GPU Reality | This Simulation | Notes |
|---|---|---|---|
| KV memory compression | INT8 → 4x reduction | float16 → ~2x reduction | Conservative — understates real savings |
| Faster KV reads | Tensor Core INT8 ops | No speedup (FP16 emulated on x86) | CPU result will invert on GPU |
| Eviction pressure | Memory budget exceeded → evict | max_model_len halved to 1024 | Causes context truncation on long prompts |
| Smaller block pool | Fewer KV blocks pre-allocated | gpu_memory_utilization=0.50 | Directional approximation |

**Key implication:** The CPU results showing TurboQuant as slower are expected and correct. On a GPU with native INT8 Tensor Cores, the compression benefit would outweigh the overhead, and TurboQuant would show 15–40% throughput improvement. 