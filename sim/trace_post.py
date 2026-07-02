#!/usr/bin/env python3
"""Convert raw llama-nvmoe-trace output into the simulator's per-token format.

Raw input (one JSON object per callback, i.e. per MoE layer per decode step):
    {"l": <layer>, "t": <n_tokens>, "e": [[expert_ids...], ...one per token]}

Output (simulator format, one JSON array per TOKEN):
    [[layer, expert], [layer, expert], ...]   in layer order

A new step is detected when the layer index does not increase (the graph
executes layers in order within one llama_decode call). Prefill steps have
t > 1; decode steps have t == 1. Use --decode-only to keep only decode tokens
(cache hit rates should be measured on decode; prefill sweeps experts by
design and is analyzed separately).

Usage:
    python3 trace_post.py raw.jsonl -o trace.jsonl [--decode-only]
    python3 trace_post.py raw.jsonl --stats        # workload statistics only
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_steps(raw_path: Path) -> list[list[dict]]:
    """Group raw records into decode/prefill steps by layer-order reset."""
    steps, current, prev_layer = [], [], None
    with raw_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if prev_layer is not None and rec["l"] <= prev_layer:
                steps.append(current)
                current = []
            current.append(rec)
            prev_layer = rec["l"]
    if current:
        steps.append(current)
    return steps


def steps_to_tokens(steps: list[list[dict]], decode_only: bool) -> list[list[tuple[int, int]]]:
    """Rebuild per-token expert lists from a step's layer records.

    During prefill llama.cpp computes the final layer only for output rows
    (usually just the last token), so that layer's record has a smaller t than
    the rest of the step. Records with t < step max align to the TRAILING
    positions of the step.
    """
    tokens = []
    for step in steps:
        n_tokens = max(rec["t"] for rec in step)
        if decode_only and n_tokens > 1:
            continue
        for tok_idx in range(n_tokens):
            token = []
            for rec in step:
                rec_idx = tok_idx - (n_tokens - rec["t"])
                if rec_idx >= 0:
                    token.extend((rec["l"], expert) for expert in rec["e"][rec_idx])
            tokens.append(token)
    return tokens


def print_stats(steps: list[list[dict]]) -> None:
    decode_tokens = steps_to_tokens(steps, decode_only=True)
    all_tokens = steps_to_tokens(steps, decode_only=False)
    layers = sorted({rec["l"] for step in steps for rec in step})
    top_k = len(steps[0][0]["e"][0]) if steps else 0

    print(f"steps               : {len(steps)} "
          f"({sum(1 for s in steps if s[0]['t'] > 1)} prefill, "
          f"{sum(1 for s in steps if s[0]['t'] == 1)} decode)")
    print(f"tokens              : {len(all_tokens)} total, {len(decode_tokens)} decode")
    print(f"moe layers          : {len(layers)} (ids {layers[0]}..{layers[-1]}), top_k={top_k}")

    # popularity skew: what share of accesses hit the top 10% of experts?
    counts = Counter(key for token in decode_tokens for key in token)
    if counts:
        ranked = [c for _, c in counts.most_common()]
        total = sum(ranked)
        top10 = sum(ranked[: max(len(ranked) // 10, 1)])
        print(f"unique (layer,expert) touched: {len(counts)}")
        print(f"top-10% experts carry : {100 * top10 / total:.1f}% of decode routing traffic")

    # temporal locality: overlap of consecutive decode tokens' expert sets
    overlaps = []
    for a, b in zip(decode_tokens, decode_tokens[1:]):
        sa, sb = set(a), set(b)
        overlaps.append(len(sa & sb) / len(sa))
    if overlaps:
        print(f"token-to-token expert overlap: {100 * sum(overlaps) / len(overlaps):.1f}% mean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("-o", "--out", type=Path)
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    steps = parse_steps(args.raw)
    if args.stats:
        print_stats(steps)
        return

    tokens = steps_to_tokens(steps, args.decode_only)
    out = args.out or args.raw.with_suffix(".tokens.jsonl")
    with out.open("w") as fh:
        for token in tokens:
            fh.write(json.dumps([list(pair) for pair in token]) + "\n")
    print(f"wrote {len(tokens)} tokens -> {out}")


if __name__ == "__main__":
    main()
