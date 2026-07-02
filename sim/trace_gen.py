"""Synthetic MoE expert-routing trace generator.

A trace is a list of tokens; each token is a list of (layer, expert_id) pairs,
top_k per MoE layer, in execution order.

The generator models the two properties of real MoE routing that matter for a
cache (both reported across the MoE-offloading literature: MoE-Infinity,
HOBBIT, PreScope):

  1. Popularity skew: expert selection frequency per layer follows a heavy-tail
     (Zipf-like) distribution — some experts are globally "hot".
  2. Temporal locality: at batch size 1, consecutive decode tokens reuse the
     recent working set of experts far more than chance.

Knobs (calibrate against REAL traces collected per docs/TRACE_COLLECTION.md):
  zipf_s    — popularity skew exponent (0 = uniform, 1.0 = classic Zipf)
  locality  — probability a selection is drawn from the recent window
  window    — how many past tokens form the "recent" working set per layer

These defaults are deliberately conservative assumptions, not measurements.
Replace them with fitted values once real traces exist; the cache simulator
consumes real traces in the same format (see load_trace_jsonl).
"""

import json
import random
from pathlib import Path

from presets import ModelPreset


def _zipf_cum_weights(n: int, s: float, rng: random.Random) -> tuple[list[int], list[float]]:
    """Build a shuffled expert population with Zipf(s) cumulative weights."""
    population = list(range(n))
    rng.shuffle(population)  # which experts are hot differs per layer
    weights = [1.0 / (rank + 1) ** s for rank in range(n)]
    cum, total = [], 0.0
    for w in weights:
        total += w
        cum.append(total)
    return population, cum


def generate_trace(
    model: ModelPreset,
    n_tokens: int = 2000,
    zipf_s: float = 1.0,
    locality: float = 0.5,
    window: int = 32,
    seed: int = 42,
) -> list[list[tuple[int, int]]]:
    rng = random.Random(seed)
    layer_dists = [
        _zipf_cum_weights(model.experts_per_layer, zipf_s, rng)
        for _ in range(model.moe_layers)
    ]
    # per-layer flat list of expert ids used in the last `window` tokens
    recent: list[list[int]] = [[] for _ in range(model.moe_layers)]

    trace = []
    for _ in range(n_tokens):
        token = []
        for layer in range(model.moe_layers):
            population, cum = layer_dists[layer]
            chosen: set[int] = set()
            while len(chosen) < model.top_k:
                if recent[layer] and rng.random() < locality:
                    expert = rng.choice(recent[layer])
                else:
                    expert = rng.choices(population, cum_weights=cum, k=1)[0]
                chosen.add(expert)
            picks = list(chosen)
            token.extend((layer, e) for e in picks)
            recent[layer].extend(picks)
            max_len = window * model.top_k
            if len(recent[layer]) > max_len:
                recent[layer] = recent[layer][-max_len:]
        trace.append(token)
    return trace


def load_trace_jsonl(path: Path) -> list[list[tuple[int, int]]]:
    """Load a real routing trace: one JSON array per line (one line per token),
    each entry a [layer, expert_id] pair. This is the format the llama.cpp
    trace collector (docs/TRACE_COLLECTION.md) emits."""
    trace = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                trace.append([tuple(pair) for pair in json.loads(line)])
    return trace
