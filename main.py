"""
main.py
─────────────────────────────────────────────────────────────────────
DevOps-ready KV cache benchmark: Baseline vs TurboQuant (simulated).

All parameters live in config.yaml.
All prompts live in prompts.json.
No hardcoded values in this file.

Usage:
    # Run with defaults (reads config.yaml + prompts.json)
    python3 main.py

    # Use a different config or prompts file
    python3 main.py --config config.yaml --prompts prompts.json

    # Run only one config
    python3 main.py --only baseline
    python3 main.py --only turboquant

    # Filter prompts by type
    python3 main.py --prompt-types short long
    python3 main.py --prompt-types multiturn

    # Change output file
    python3 main.py --output results/run_001.json

    # Dry run — print config and prompts without running inference
    python3 main.py --dry-run
─────────────────────────────────────────────────────────────────────
"""

import argparse
import gc
import json
import logging
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import yaml

# ─────────────────────────────────────────────────────────────────────
# LOGGING — structured, timestamped
# ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("kv_cache_bench")


# ─────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PromptEntry:
    id: str
    type: str       # short | long | multiturn
    text: str
    note: str = ""


@dataclass
class RequestResult:
    prompt_id: str
    prompt_type: str
    e2e_ms: float
    tokens_generated: int
    tokens_per_second: float
    output: str
    truncated: bool = False


@dataclass
class ConfigResult:
    config_name: str
    config_params: dict
    requests: List[RequestResult] = field(default_factory=list)
    peak_ram_mb: float = 0.0
    started_at: str = ""
    finished_at: str = ""

    def latencies(self) -> List[float]:
        return [r.e2e_ms for r in self.requests]

    def tps_list(self) -> List[float]:
        return [r.tokens_per_second for r in self.requests]

    @staticmethod
    def percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        k = (len(s) - 1) * pct / 100
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (k - lo) * (s[hi] - s[lo])


# ─────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    log.info(f"Loaded config: {config_path}")
    return cfg


def load_prompts(prompts_path: str, type_filter: Optional[List[str]] = None) -> List[PromptEntry]:
    path = Path(prompts_path)
    if not path.exists():
        log.error(f"Prompts file not found: {prompts_path}")
        sys.exit(1)
    with open(path) as f:
        raw = json.load(f)

    prompts = []
    for key, val in raw.items():
        if key.startswith("_"):
            continue  # skip _comment / _format metadata fields
        entry = PromptEntry(
            id=key,
            type=val.get("type", "short"),
            text=val["text"],
            note=val.get("note", ""),
        )
        if type_filter is None or entry.type in type_filter:
            prompts.append(entry)

    log.info(
        f"Loaded {len(prompts)} prompts from {prompts_path}"
        + (f" (types: {type_filter})" if type_filter else "")
    )
    return prompts


def apply_env(env_vars: dict):
    """Set environment variables from config before vLLM imports."""
    for key, val in env_vars.items():
        os.environ.setdefault(key, str(val))
        log.info(f"  env: {key}={os.environ[key]}")


# ─────────────────────────────────────────────────────────────────────
# QUALITY METRIC
# ─────────────────────────────────────────────────────────────────────

def token_f1(text_a: str, text_b: str) -> float:
    """Token-level F1 score. No external dependencies required."""
    tok_a = set(re.findall(r"\b\w+\b", text_a.lower()))
    tok_b = set(re.findall(r"\b\w+\b", text_b.lower()))
    if not tok_a or not tok_b:
        return 0.0
    common = tok_a & tok_b
    precision = len(common) / len(tok_b)
    recall = len(common) / len(tok_a)
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────────────────

def run_config(
    config_name: str,
    cfg_params: dict,
    prompts: List[PromptEntry],
    max_tokens: int,
    temperature: float,
    warmup_count: int,
) -> ConfigResult:
    # Late import so env vars are set first
    from vllm import LLM, SamplingParams

    log.info("")
    log.info("=" * 58)
    log.info(f"  CONFIG: {config_name.upper()}")
    log.info(f"  {cfg_params.get('description', '')}")
    log.info("=" * 58)
    for k, v in cfg_params.items():
        if k not in ("description", "model"):
            log.info(f"  {k}: {v}")

    # Only pass recognised vLLM LLM() kwargs
    vllm_keys = {
        "dtype", "max_model_len", "gpu_memory_utilization",
        "swap_space", "enforce_eager", "max_num_seqs",
    }
    vllm_params = {k: v for k, v in cfg_params.items() if k in vllm_keys}
    model = cfg_params.get("model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    result = ConfigResult(
        config_name=config_name,
        config_params=vllm_params,
        started_at=datetime.now().isoformat(),
    )

    ram_before = psutil.virtual_memory().used / 1024 / 1024
    llm = LLM(model=model, **vllm_params)

    for i in range(warmup_count):
        log.info(f"  [warmup {i+1}/{warmup_count}]")
        llm.generate(["Warmup."], SamplingParams(max_tokens=10, temperature=0.0))
    log.info("  Warmup done.\n")

    sampling = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    peak_ram = ram_before

    for prompt in prompts:
        log.info(f"  [{prompt.id}] ({prompt.type}) {prompt.note}")

        # Trim prompt to fit context window budget
        max_chars = (cfg_params.get("max_model_len", 2048) - max_tokens - 10) * 4
        text = prompt.text
        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True
            log.warning(f"    Truncated to fit max_model_len={cfg_params.get('max_model_len')}")

        t0 = time.perf_counter()
        output = llm.generate([text], sampling)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        out_text = output[0].outputs[0].text
        n_tokens = len(output[0].outputs[0].token_ids)
        tps = n_tokens / elapsed if elapsed > 0 else 0.0

        current_ram = psutil.virtual_memory().used / 1024 / 1024
        if current_ram > peak_ram:
            peak_ram = current_ram

        result.requests.append(RequestResult(
            prompt_id=prompt.id,
            prompt_type=prompt.type,
            e2e_ms=elapsed * 1000,
            tokens_generated=n_tokens,
            tokens_per_second=tps,
            output=out_text,
            truncated=truncated,
        ))
        log.info(
            f"    → {elapsed*1000:.0f}ms | {n_tokens} tok | {tps:.1f} tok/s"
            + (" [TRUNCATED]" if truncated else "")
        )

    result.peak_ram_mb = peak_ram
    result.finished_at = datetime.now().isoformat()

    del llm
    gc.collect()
    return result


# ─────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────

def delta_str(b: float, t: float) -> str:
    if b == 0:
        return "  N/A"
    pct = (t - b) / b * 100
    return f"{'+' if pct > 0 else ''}{pct:.1f}%"


def print_report(
    baseline: ConfigResult,
    turboquant: ConfigResult,
    quality_scores: Dict[str, float],
    quality_cfg: dict,
):
    hi = quality_cfg.get("high_threshold", 0.80)
    me = quality_cfg.get("med_threshold", 0.60)

    print("\n" + "=" * 60)
    print("  RESULTS: Baseline vs TurboQuant")
    print("=" * 60)

    # Latency table
    print(f"\n  {'LATENCY (ms)':<22} {'Baseline':>10} {'TurboQuant':>12} {'Δ':>8}")
    print(f"  {'─'*54}")
    for label, pct in [("E2E p50", 50), ("E2E p95", 95), ("E2E p99", 99)]:
        b = ConfigResult.percentile(baseline.latencies(), pct)
        t = ConfigResult.percentile(turboquant.latencies(), pct)
        print(f"  {label:<22} {b:>10.0f} {t:>12.0f} {delta_str(b, t):>8}")

    # Throughput table
    b_tps = statistics.mean(baseline.tps_list()) if baseline.tps_list() else 0
    t_tps = statistics.mean(turboquant.tps_list()) if turboquant.tps_list() else 0
    b_total = sum(r.tokens_generated for r in baseline.requests)
    t_total = sum(r.tokens_generated for r in turboquant.requests)
    print(f"\n  {'THROUGHPUT':<22} {'Baseline':>10} {'TurboQuant':>12} {'Δ':>8}")
    print(f"  {'─'*54}")
    print(f"  {'Avg tok/s':<22} {b_tps:>10.1f} {t_tps:>12.1f} {delta_str(b_tps, t_tps):>8}")
    print(f"  {'Total tokens':<22} {b_total:>10} {t_total:>12}")

    # Memory table
    print(f"\n  {'MEMORY':<22} {'Baseline':>10} {'TurboQuant':>12} {'Δ':>8}")
    print(f"  {'─'*54}")
    print(f"  {'Peak RAM (MB)':<22} {baseline.peak_ram_mb:>10.0f} {turboquant.peak_ram_mb:>12.0f} {delta_str(baseline.peak_ram_mb, turboquant.peak_ram_mb):>8}")

    # Quality table
    print(f"\n  {'QUALITY (Token-F1)':<22} {'Score':>10} {'Rating':>12}")
    print(f"  {'─'*46}")
    for pid, score in quality_scores.items():
        label = "HIGH" if score >= hi else "MED" if score >= me else "LOW"
        print(f"  {pid:<22} {score:>10.3f} {label:>12}")
    avg_q = statistics.mean(quality_scores.values()) if quality_scores else 0
    print(f"  {'─'*46}")
    print(f"  {'AVERAGE':<22} {avg_q:>10.3f}")

    # Verdict
    tps_gain = (t_tps - b_tps) / b_tps * 100 if b_tps > 0 else 0
    ram_saving = (baseline.peak_ram_mb - turboquant.peak_ram_mb) / baseline.peak_ram_mb * 100 \
                 if baseline.peak_ram_mb > 0 else 0

    if avg_q >= hi and (tps_gain > 0 or ram_saving > 5):
        verdict = "ADOPT"
        reason = f"Quality preserved (F1={avg_q:.2f}), {ram_saving:.1f}% RAM savings."
    elif avg_q >= me:
        verdict = "CONDITIONAL"
        reason = f"Acceptable quality (F1={avg_q:.2f}). Validate on your domain prompts."
    else:
        verdict = "REJECT (CPU)"
        reason = (
            f"Quality degraded (F1={avg_q:.2f}). "
            "FP16 has no HW acceleration on x86 CPU. "
            "Context truncation hurts long/multiturn prompts."
        )

    print(f"\n  VERDICT : {verdict}")
    print(f"  REASON  : {reason}")
    print(f"  NOTE    : CPU simulation only. GPU TurboQuant (INT8) achieves 4x KV")
    print(f"            compression — results would differ significantly on GPU.")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────
# CLI + MAIN
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="KV Cache Benchmark — reads all params from YAML + JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="config.yaml",
                   help="YAML experiment config (default: config.yaml)")
    p.add_argument("--prompts", default=None,
                   help="JSON prompts file (overrides config prompts_file)")
    p.add_argument("--output", default=None,
                   help="JSON results output path (overrides config output_file)")
    p.add_argument("--only", default=None, choices=["baseline", "turboquant"],
                   help="Run only one config")
    p.add_argument("--prompt-types", nargs="+", default=None,
                   choices=["short", "long", "multiturn"],
                   help="Filter prompts by type")
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved config and prompts, skip inference")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    configs = cfg["configs"]
    quality_cfg = cfg.get("quality", {})

    log.info("Setting environment variables...")
    apply_env(cfg.get("env", {}))

    prompts_path = args.prompts or exp.get("prompts_file", "prompts.json")
    output_path = args.output or exp.get("output_file", "kv_cache_results.json")
    prompts = load_prompts(prompts_path, type_filter=args.prompt_types)

    if not prompts:
        log.error("No prompts loaded. Check prompts file or --prompt-types filter.")
        sys.exit(1)

    # ── Dry run ───────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'='*60}")
        print("  DRY RUN — no inference will run")
        print(f"{'='*60}")
        print(f"  Experiment : {exp['name']}")
        print(f"  Model      : {exp['model']}")
        print(f"  Prompts    : {len(prompts)} from {prompts_path}")
        print(f"  Output     : {output_path}")
        print(f"\n  Configs:")
        for name, params in configs.items():
            skip = args.only and name != args.only
            status = f"SKIPPED (--only {args.only})" if skip else \
                     f"dtype={params['dtype']}, max_model_len={params['max_model_len']}"
            print(f"    [{name}]  {status}")
        print(f"\n  Prompts ({len(prompts)}):")
        for p in prompts:
            preview = p.text[:72] + ("..." if len(p.text) > 72 else "")
            print(f"    [{p.id}] ({p.type})  {preview}")
        return

    # ── Banner ────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 58)
    log.info(f"  {exp['name']}")
    log.info(f"  Model: {exp['model']}")
    log.info(f"  Prompts: {len(prompts)}")
    log.info("=" * 58)

    # Inject model into each config profile
    for name in configs:
        configs[name]["model"] = exp["model"]

    configs_to_run = {
        k: v for k, v in configs.items()
        if args.only is None or k == args.only
    }

    # ── Run each config ───────────────────────────────────────────────
    results: Dict[str, ConfigResult] = {}
    for config_name, cfg_params in configs_to_run.items():
        results[config_name] = run_config(
            config_name=config_name,
            cfg_params=cfg_params,
            prompts=prompts,
            max_tokens=exp.get("max_tokens", 200),
            temperature=exp.get("temperature", 0.0),
            warmup_count=exp.get("warmup_requests", 1),
        )

    # ── Quality + report ──────────────────────────────────────────────
    quality_scores: Dict[str, float] = {}
    if "baseline" in results and "turboquant" in results:
        bl_outputs = {r.prompt_id: r.output for r in results["baseline"].requests}
        for req in results["turboquant"].requests:
            if req.prompt_id in bl_outputs:
                quality_scores[req.prompt_id] = token_f1(bl_outputs[req.prompt_id], req.output)
        print_report(results["baseline"], results["turboquant"], quality_scores, quality_cfg)
    else:
        name = list(results.keys())[0]
        r = results[name]
        log.info(f"\nSingle-config run: {name}")
        log.info(f"  Requests  : {len(r.requests)}")
        if r.tps_list():
            log.info(f"  Avg tok/s : {statistics.mean(r.tps_list()):.1f}")
        log.info(f"  Peak RAM  : {r.peak_ram_mb:.0f} MB")

    # ── Save JSON results ─────────────────────────────────────────────
    avg_q = statistics.mean(quality_scores.values()) if quality_scores else None
    output_data = {
        "experiment": {
            "name": exp["name"],
            "model": exp["model"],
            "prompts_file": prompts_path,
            "run_at": datetime.now().isoformat(),
            "prompt_types": list({p.type for p in prompts}),
            "total_prompts": len(prompts),
        },
        "configs_run": list(results.keys()),
        "results": {
            name: {
                "config": r.config_params,
                "started_at": r.started_at,
                "finished_at": r.finished_at,
                "e2e_p50_ms": ConfigResult.percentile(r.latencies(), 50),
                "e2e_p95_ms": ConfigResult.percentile(r.latencies(), 95),
                "e2e_p99_ms": ConfigResult.percentile(r.latencies(), 99),
                "avg_tps": statistics.mean(r.tps_list()) if r.tps_list() else 0,
                "total_tokens": sum(req.tokens_generated for req in r.requests),
                "peak_ram_mb": r.peak_ram_mb,
                "per_prompt": [
                    {
                        "id": req.prompt_id,
                        "type": req.prompt_type,
                        "e2e_ms": round(req.e2e_ms, 1),
                        "tokens": req.tokens_generated,
                        "tps": round(req.tokens_per_second, 2),
                        "truncated": req.truncated,
                        "output_preview": req.output[:200],
                    }
                    for req in r.requests
                ],
            }
            for name, r in results.items()
        },
        "quality": {
            "per_prompt": quality_scores,
            "avg_token_f1": avg_q,
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved → {output_path}")


if __name__ == "__main__":
    main()