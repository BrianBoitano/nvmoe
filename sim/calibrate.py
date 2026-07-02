#!/usr/bin/env python3
"""Calibrate the synthetic trace generator against a REAL routing trace.

Grid-searches (zipf_s, locality) so the synthetic trace matches the real
trace on the two statistics that drive cache behavior, then reports the
metric that actually matters: LRU hit rate at several cache sizes, real vs
calibrated-synthetic. If the synthetic generator can't reproduce the real
hit-rate curve even after calibration, the simulator's tok/s ceilings must
be taken from real traces only — that result is worth publishing either way.

Usage:
    python3 sim/calibrate.py --model qwen3-30b-a3b --trace traces/qwen3-all.tokens.jsonl
"""

import argparse
from collections import Counter
from pathlib import Path

from cache_sim import simulate_lru
from presets import MODELS
from trace_gen import generate_trace, load_trace_jsonl

GRID_ZIPF = [0.0, 0.3, 0.6, 0.9, 1.2]
GRID_LOCALITY = [0.0, 0.2, 0.4, 0.6, 0.8]
CACHE_FRACTIONS = [0.05, 0.10, 0.25, 0.50]  # of total expert count


def trace_stats(trace: list[list[tuple[int, int]]]) -> tuple[float, float]:
    """Return (top10_traffic_share, mean_token_overlap)."""
    counts = Counter(key for token in trace for key in token)
    ranked = [c for _, c in counts.most_common()]
    total = sum(ranked)
    top10 = sum(ranked[: max(len(ranked) // 10, 1)]) / total if total else 0.0

    overlaps = []
    for a, b in zip(trace, trace[1:]):
        sa = set(a)
        overlaps.append(len(sa & set(b)) / len(sa))
    overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    return top10, overlap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--trace", required=True, type=Path,
                        help="post-processed real trace (trace_post.py output)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = MODELS[args.model]
    real = load_trace_jsonl(args.trace)
    real_top10, real_overlap = trace_stats(real)
    print(f"real trace: {len(real)} tokens, top10-share={real_top10:.3f}, "
          f"overlap={real_overlap:.3f}")

    # fit generator params to the real trace's summary statistics
    best, best_err = None, float("inf")
    for zipf_s in GRID_ZIPF:
        for locality in GRID_LOCALITY:
            synth = generate_trace(model, n_tokens=min(len(real), 1500),
                                   zipf_s=zipf_s, locality=locality, seed=args.seed)
            top10, overlap = trace_stats(synth)
            err = (top10 - real_top10) ** 2 + (overlap - real_overlap) ** 2
            if err < best_err:
                best, best_err = (zipf_s, locality, top10, overlap), err

    zipf_s, locality, top10, overlap = best
    print(f"best fit  : zipf_s={zipf_s}, locality={locality} "
          f"(synthetic top10={top10:.3f}, overlap={overlap:.3f})")

    # the test that matters: does the fit reproduce the hit-rate curve?
    synth = generate_trace(model, n_tokens=min(len(real), 1500),
                           zipf_s=zipf_s, locality=locality, seed=args.seed)
    print(f"\n{'cache':>12} {'real LRU':>9} {'synth LRU':>10} {'delta':>7}")
    for frac in CACHE_FRACTIONS:
        capacity = int(model.total_experts * frac)
        r = simulate_lru(real, capacity, warmup_tokens=min(100, len(real) // 5))
        s = simulate_lru(synth, capacity, warmup_tokens=min(100, len(synth) // 5))
        print(f"{frac:>11.0%} {r.hit_rate:>9.1%} {s.hit_rate:>10.1%} "
              f"{s.hit_rate - r.hit_rate:>+7.1%}")


if __name__ == "__main__":
    main()
