#!/usr/bin/env python3
"""nvmoe cache simulator — estimate the I/O-bound decode ceiling for MoE
models whose routed experts live on NVMe and page into a 16GB-VRAM cache.

Examples:
    python3 run_sim.py --all
    python3 run_sim.py --model deepseek-r1-671b --tokens 3000
    python3 run_sim.py --model qwen3-next-80b --trace traces/real.jsonl

The tok/s figures are CEILINGS: they assume expert fetches perfectly overlap
compute and that NVMe sustains its sequential-read bandwidth on ~1-100MB
expert-sized reads (io_uring + O_DIRECT territory). Real throughput lands
below this; the point of the simulator is to rank strategies and cache
budgets before writing any runtime code.
"""

import argparse
from pathlib import Path

from cache_sim import hottest_experts, simulate_lru
from presets import HARDWARE, MODELS, HardwarePreset, ModelPreset
from trace_gen import generate_trace, load_trace_jsonl


def cache_budget_gb(model: ModelPreset, hw: HardwarePreset) -> float:
    """VRAM left for the expert cache after the resident floor and KV cache."""
    return max(hw.vram_gb - model.always_on_gb - model.kv_gb, 0.0)


def ceiling_toks(model: ModelPreset, hw: HardwarePreset, hit_rate: float) -> tuple[float, float]:
    """Return (bytes_per_token_gb, tokens_per_second_ceiling)."""
    miss_bytes = model.active_experts_per_token * model.expert_bytes * (1.0 - hit_rate)
    gb = miss_bytes / 1e9
    return gb, (hw.nvme_gbps / gb if gb > 0 else float("inf"))


def report(model_key: str, args: argparse.Namespace) -> None:
    model = MODELS[model_key]
    hw = HARDWARE[args.hardware]

    budget = cache_budget_gb(model, hw)
    capacity = int(budget * 1e9 / model.expert_bytes)

    if args.trace:
        trace = load_trace_jsonl(Path(args.trace))
        trace_kind = f"real ({args.trace})"
    else:
        trace = generate_trace(
            model, n_tokens=args.tokens, zipf_s=args.zipf_s,
            locality=args.locality, seed=args.seed,
        )
        trace_kind = f"synthetic (zipf_s={args.zipf_s}, locality={args.locality})"

    lru = simulate_lru(trace, capacity)
    pins = hottest_experts(trace, capacity // 2)
    pinned = simulate_lru(trace, capacity, pinned=pins)

    print(f"\n=== {model.name} ===")
    print(f"hardware            : {hw.name}")
    print(f"trace               : {trace_kind}, {len(trace)} tokens")
    print(f"routed experts      : {model.total_experts} x {model.expert_bytes / 1e6:.1f}MB "
          f"= {model.total_expert_gb:.0f}GB on NVMe")
    print(f"VRAM floor          : {model.always_on_gb:.1f}GB dense/attn/shared "
          f"+ {model.kv_gb:.1f}GB KV")
    print(f"expert cache budget : {budget:.1f}GB -> {capacity} experts "
          f"({100 * capacity / model.total_experts:.1f}% of all experts)")
    print(f"full prefill sweep  : {model.total_expert_gb / hw.nvme_gbps:.0f}s per pass "
          f"over all experts (long-prompt worst case)")

    header = f"{'policy':<10} {'hit rate':>9} {'GB/token':>9} {'ceiling tok/s':>14}"
    print(header)
    print("-" * len(header))
    for result in (simulate_lru(trace, 0), lru, pinned):
        gb, tps = ceiling_toks(model, hw, result.hit_rate)
        label = "no cache" if result.capacity_experts == 0 else result.policy
        print(f"{label:<10} {result.hit_rate:>8.1%} {gb:>9.2f} {tps:>14.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), help="model preset")
    parser.add_argument("--all", action="store_true", help="run every preset")
    parser.add_argument("--hardware", default="5070ti-990pro", choices=sorted(HARDWARE))
    parser.add_argument("--trace", help="real routing trace (JSONL) instead of synthetic")
    parser.add_argument("--tokens", type=int, default=2000)
    parser.add_argument("--zipf-s", type=float, default=1.0)
    parser.add_argument("--locality", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    keys = sorted(MODELS) if args.all or not args.model else [args.model]
    for key in keys:
        report(key, args)


if __name__ == "__main__":
    main()
