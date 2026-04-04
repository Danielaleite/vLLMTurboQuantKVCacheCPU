"""
benchmark/workloads.py
──────────────────────
Benchmark workload definitions for TurboQuant KV cache evaluation.

Covers three categories that stress KV cache differently:
  1. SHORT prompts  → minimal KV cache usage, baseline latency
  2. LONG prompts   → heavy KV cache allocation, memory pressure
  3. MULTI-TURN     → cache reuse patterns, eviction behavior
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Prompt:
    id: str
    category: str          # short | long | multiturn
    system: Optional[str]
    messages: List[dict]   # OpenAI-style messages array
    expected_min_tokens: int = 50
    expected_max_tokens: int = 512
    tags: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
#  SHORT PROMPTS  (< 50 input tokens)
#  Stresses: TTFT, minimal KV allocation
# ─────────────────────────────────────────────

SHORT_PROMPTS = [
    Prompt(
        id="short_factual_1",
        category="short",
        system="You are a helpful assistant. Be concise.",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        expected_min_tokens=5,
        expected_max_tokens=30,
        tags=["factual", "deterministic"],
    ),
    Prompt(
        id="short_factual_2",
        category="short",
        system="You are a helpful assistant. Be concise.",
        messages=[{"role": "user", "content": "What is 17 multiplied by 13?"}],
        expected_min_tokens=5,
        expected_max_tokens=20,
        tags=["math", "deterministic"],
    ),
    Prompt(
        id="short_creative_1",
        category="short",
        system=None,
        messages=[{"role": "user", "content": "Write a haiku about the moon."}],
        expected_min_tokens=10,
        expected_max_tokens=50,
        tags=["creative"],
    ),
    Prompt(
        id="short_reasoning_1",
        category="short",
        system="Answer briefly.",
        messages=[{"role": "user", "content": "If all cats are animals and Whiskers is a cat, what is Whiskers?"}],
        expected_min_tokens=5,
        expected_max_tokens=40,
        tags=["reasoning", "logic"],
    ),
    Prompt(
        id="short_code_1",
        category="short",
        system="You are a coding assistant.",
        messages=[{"role": "user", "content": "Write a Python one-liner to reverse a string."}],
        expected_min_tokens=10,
        expected_max_tokens=60,
        tags=["code"],
    ),
]

# ─────────────────────────────────────────────
#  LONG PROMPTS  (200–800 input tokens)
#  Stresses: KV cache memory, cache block allocation
# ─────────────────────────────────────────────

LONG_CONTEXT_PASSAGE = """
The history of computing spans several decades and multiple technological revolutions.
In the 1940s, the first electronic computers filled entire rooms and were programmed
using punch cards. ENIAC, completed in 1945, was one of the earliest general-purpose
electronic computers and weighed 30 tons. The invention of the transistor in 1947
by Bell Labs scientists Shockley, Bardeen, and Brattain marked a turning point that
would eventually miniaturize computing. Through the 1950s and 1960s, mainframes
dominated corporate and government computing, with IBM becoming the dominant player.
The microprocessor revolution of the 1970s, sparked by Intel's 4004 chip in 1971,
brought computing power to individuals. Apple and Microsoft emerged in the late 1970s
and early 1980s, ushering in the personal computer era. The Internet's commercialization
in the 1990s transformed computing from a productivity tool into a global communication
platform. The 2000s brought mobile computing and smartphones, while the 2010s saw the
rise of cloud computing, deep learning, and AI — completing a journey from room-sized
machines to AI assistants running on devices that fit in a pocket.
"""

LONG_TECHNICAL_PASSAGE = """
Transformer architecture, introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017),
fundamentally changed natural language processing. Unlike recurrent neural networks that process
sequences step by step, transformers use self-attention mechanisms to process all tokens in parallel.
The key innovation is the attention mechanism: for each token, the model computes query (Q), key (K),
and value (V) vectors. Attention scores are computed as softmax(QK^T / sqrt(d_k)) * V, allowing
each token to attend to all other tokens simultaneously. Multi-head attention runs this process
in parallel across multiple "heads," capturing different types of relationships. The KV cache in
autoregressive generation stores the key and value tensors from previous tokens to avoid recomputation
during each generation step. This cache grows linearly with sequence length and is a primary
bottleneck for memory and latency in production LLM deployments. Quantizing the KV cache from
FP16/FP32 to INT8 or INT4 reduces memory by 2-4x at the cost of some precision loss in attention
computation, which may affect output coherence for long sequences.
"""

LONG_PROMPTS = [
    Prompt(
        id="long_summarize_1",
        category="long",
        system="You are a helpful assistant. Provide thorough responses.",
        messages=[{
            "role": "user",
            "content": f"Please summarize the following passage and identify the three most important milestones:\n\n{LONG_CONTEXT_PASSAGE}"
        }],
        expected_min_tokens=100,
        expected_max_tokens=400,
        tags=["summarization", "long-context"],
    ),
    Prompt(
        id="long_technical_1",
        category="long",
        system="You are a machine learning engineer.",
        messages=[{
            "role": "user",
            "content": f"Based on this technical description, explain in simple terms what the KV cache does and why quantizing it is useful:\n\n{LONG_TECHNICAL_PASSAGE}"
        }],
        expected_min_tokens=100,
        expected_max_tokens=400,
        tags=["technical", "explanation", "long-context"],
    ),
    Prompt(
        id="long_analysis_1",
        category="long",
        system="You are an analyst. Be thorough.",
        messages=[{
            "role": "user",
            "content": (
                "I am designing a distributed system with the following requirements: "
                "high availability (99.99% uptime), global distribution across 5 regions, "
                "sub-100ms read latency, eventual consistency for writes, support for "
                "100,000 concurrent users, GDPR compliance for EU users, automatic failover, "
                "horizontal scalability, cost optimization for read-heavy workloads (95% reads, 5% writes), "
                "and integration with existing PostgreSQL databases. "
                "Please analyze these requirements and suggest an architecture, "
                "discussing trade-offs for each major decision."
            )
        }],
        expected_min_tokens=200,
        expected_max_tokens=512,
        tags=["analysis", "architecture", "long-output"],
    ),
]

# ─────────────────────────────────────────────
#  MULTI-TURN CONVERSATIONS
#  Stresses: KV cache reuse, eviction under growing context
# ─────────────────────────────────────────────

MULTITURN_PROMPTS = [
    Prompt(
        id="multiturn_coding_1",
        category="multiturn",
        system="You are a Python programming assistant.",
        messages=[
            {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."},
            {"role": "assistant", "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
            {"role": "user", "content": "Now make it iterative instead of recursive."},
            {"role": "assistant", "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b\n    return b"},
            {"role": "user", "content": "Add memoization to the recursive version and show me both approaches with timing."},
        ],
        expected_min_tokens=100,
        expected_max_tokens=400,
        tags=["code", "multiturn", "context-reuse"],
    ),
    Prompt(
        id="multiturn_factual_1",
        category="multiturn",
        system="You are a knowledgeable assistant.",
        messages=[
            {"role": "user", "content": "Tell me about the Roman Empire."},
            {"role": "assistant", "content": "The Roman Empire was one of the largest empires in history, spanning from 27 BC to 476 AD in the West. At its height it controlled most of Europe, North Africa, and parts of the Middle East."},
            {"role": "user", "content": "What caused its decline?"},
            {"role": "assistant", "content": "The decline had multiple causes: military overextension, economic troubles, political instability (the Crisis of the Third Century), pressures from Germanic tribes, and the division into Eastern and Western empires."},
            {"role": "user", "content": "How did its fall impact medieval Europe?"},
        ],
        expected_min_tokens=100,
        expected_max_tokens=350,
        tags=["factual", "multiturn", "history"],
    ),
    Prompt(
        id="multiturn_creative_1",
        category="multiturn",
        system="You are a creative writing assistant.",
        messages=[
            {"role": "user", "content": "Let's write a short sci-fi story together. Start with a character waking up on a space station."},
            {"role": "assistant", "content": "Commander Elena Vasquez opened her eyes to the soft hum of life support systems. The station's viewport framed a perfect crescent Earth below — but something was wrong. The crew quarters were empty."},
            {"role": "user", "content": "She checks the ship's log and discovers the crew was evacuated 72 hours ago. Continue."},
            {"role": "assistant", "content": "The log's timestamp made her stomach drop. Seventy-two hours. She'd been unconscious for three days. The last entry read: 'Anomalous signal detected from Sector 7. All non-essential personnel evacuated. Vasquez in medical bay — status unknown. God help us.'"},
            {"role": "user", "content": "She investigates Sector 7. What does she find?"},
        ],
        expected_min_tokens=100,
        expected_max_tokens=400,
        tags=["creative", "multiturn", "narrative"],
    ),
]

# ─────────────────────────────────────────────
#  Aggregated workload
# ─────────────────────────────────────────────

ALL_PROMPTS: List[Prompt] = SHORT_PROMPTS + LONG_PROMPTS + MULTITURN_PROMPTS

WORKLOAD_CATEGORIES = {
    "short": SHORT_PROMPTS,
    "long": LONG_PROMPTS,
    "multiturn": MULTITURN_PROMPTS,
    "all": ALL_PROMPTS,
}


def get_workload(category: str = "all") -> List[Prompt]:
    if category not in WORKLOAD_CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Choose from {list(WORKLOAD_CATEGORIES.keys())}")
    return WORKLOAD_CATEGORIES[category]