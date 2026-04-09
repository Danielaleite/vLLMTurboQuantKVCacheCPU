# TurboQuantKV Cache Evaluation

A simple benchmarking framework comparing Baseline and TurboQuant-simulated KVcache strategies on CPU-only vLLM inference using TinyLlama-1.1B.

> **What this does:** Runs two vLLM configurations, measures latency (E2E p50/p95/p99), throughput (tok/s), RAM usage, and output quality (Token-F1).
---

## Overview

[TurboQuant](https://github.com/0xSero/turboquant) is a GPU-native KV cache quantization technique in vLLM that compresses Key-Value tensors to INT8/FP8, reducing memory by 4–8x. This repo simulates its effects using CPU-available knobs and documents all assumptions made.

| Configuration | dtype | max_model_len | Memory budget | What it simulates |
|---|---|---|---|---|
| **Baseline** | float32 | 2048 | 90% | Standard vLLM, no optimization |
| **TurboQuant** | float16 | 1024 | 50% | FP16 ≈ 2x KV memory reduction; truncated context ≈ eviction pressure |

**Key result on CPU:** TurboQuant uses ~6% less RAM, but has higher latency compared to the baseline. This is expected — FP16 has no hardware acceleration on x86 CPUs.

## Repository Structure

```
vLLMTurboQuantKVCacheCPU
│
├── main.py          # Main script
├── generate_plots.py  'uses the results files to generate nice plots         
├── config.yaml    # all parameters here
├── prompts.json         # all benchmark prompts - one can edit it here
├── setup.sh              # Environment setup 
├── results.json        # results json saved
│
├── benchmark/
│   ├── metrics_eval.py             
│
├── results/
│   ├── figures/              # Auto-generated PNG plots│
├── docs/
│   └── assumptions.md        # Full TurboQuant simulation rationale
└── .gitignore
```

---

## Prerequisites

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

---

## Installation

Pick one of two paths:

### Option A — Local (WSL2 / Linux)

**Requirements:** WSL2 Ubuntu 22.04, 8 GB+ RAM, x86_64 CPU with AVX2 (any CPU after ~2013), Python 3.12, conda.

**1. If using WSL2, increase the memory limit first** (open PowerShell, not WSL):

```powershell
notepad "$env:USERPROFILE\.wslconfig"
```

Paste this (adjust `memory` to ~75% of your total RAM):

```ini
[wsl2]
memory=12GB
swap=8GB
processors=4
```

Then restart WSL2:

```powershell
wsl --shutdown
```

**2. Run the setup script:**

```bash
chmod +x setup.sh && ./setup.sh
```

This installs conda, creates the environment, installs the vLLM CPU wheel, and downloads the model (~2.2 GB). Takes 5–10 minutes.

**3. Run the benchmark:**

```bash
conda activate vllm-cpu
VLLM_CPU_KVCACHE_SPACE=4 python3 main.py
```

Results saved to `kv_cache_results.json`.

---

### Option B — Docker (easiest, no setup required)

---

## Common Commands

```bash
# Preview what will run without loading the model
python3 main.py --dry-run

# Run only short prompts (~5 min)
VLLM_CPU_KVCACHE_SPACE=4 python3 main.py --prompt-types short

# Run only one config
VLLM_CPU_KVCACHE_SPACE=4 python3 main.py --only baseline
VLLM_CPU_KVCACHE_SPACE=4 python3 main.py --only turboquant

# Custom output file
VLLM_CPU_KVCACHE_SPACE=4 python3 main.py --output results/run_01.json

# Generate plots (after results.json exists)
python3 generate_plots.py
```

---

## Configuration

All parameters are in `config.yaml` — no code changes needed. Key settings:

| Setting | Default | What it does |
|---|---|---|
| `model` | TinyLlama-1.1B-Chat-v1.0 | Model to benchmark |
| `max_tokens` | 200 | Max tokens to generate per prompt |
| `temperature` | 0.0 | 0 = deterministic/reproducible |
| `warmup_requests` | 1 | Warmup passes before measuring |

The two configs being compared:

| | Baseline | TurboQuant |
|---|---|---|
| dtype | float32 | float16 |
| max_model_len | 2048 | 1024 |
| Memory budget | 90% | 50% |

To add or edit prompts, edit `prompts.json`. Each entry:

```json
{
  "my_prompt": {
    "type": "short",
    "note": "Description",
    "text": "The actual prompt sent to the model"
  }
}
```

Valid types: `short`, `long`, `multiturn`.

---

## Understanding Results

The benchmark prints a comparison table and one of three verdicts:

- **ADOPT** — quality preserved, meaningful RAM/speed gains
- **CONDITIONAL** — acceptable quality, validate on your own prompts
- **REJECT** — quality degraded too much

Results are also saved as JSON with per-prompt breakdowns.

**Note on CPU results:** TurboQuant will appear ~20% *slower* on CPU — this is expected. FP16 has no hardware acceleration on x86. On a GPU with INT8 Tensor Cores, the results invert and TurboQuant shows 15–40% throughput improvement.

---

## Troubleshooting

**Process killed silently during model load** → WSL2 ran out of memory. Increase `memory=` in `.wslconfig` and run `wsl --shutdown`.

**`No module named 'vllm._C'`** → vLLM not installed correctly. Run:
```bash
pip uninstall vllm vllm-cpu -y
pip install vllm-cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

**`AVX2 not supported`** → Use the Docker path (Option A above).

**Plots don't display in WSL2** → Expected. Find the PNG files in Windows Explorer at `\\wsl$\Ubuntu\home\YOUR_USER\...\results\figures\`.

**`conda: command not found`** → Run `source ~/.bashrc` or restart your terminal.
