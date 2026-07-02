#!/usr/bin/env python3
"""Would cross-layer prefetch prediction help? Ask the traces. (stdlib only)

This is the analysis that shaped the runtime's lookahead prefetcher
(runtime/patches/0004). It evaluates, against a real routing trace
(traces/*.tokens.jsonl), the two prefetch predictors that DON'T need model
internals — and shows why neither is good enough, which is why the runtime
predicts from hidden states instead (see docs/INTEGRATION.md):

  [1] same-ids heuristic: do layer L's selected expert ids reappear at
      layer L+1 on the same token? (If ids correlated positionally across
      layers, prefetch would be a lookup.)
  [2] offline conditional table P(expert at L+1 | expert at L), evaluated
      as held-out recall@M vs a static per-layer popularity baseline.
  [3] the number that actually matters: replay an LRU cache at a given
      budget with table-driven lookahead-1 prefetch, and measure the hit
      rate it buys vs the extra I/O it costs.

Usage:
    python3 tools/analyze_lookahead.py traces/qwen3-all.tokens.jsonl

Findings on the shipped Qwen3-30B-A3B trace (1396 decode tokens):
  [1] overlap 0.455/8 vs 0.500 chance -> ids carry no positional signal.
  [2] recall@8 51% (table) vs 18% (popularity) -> correlation is real...
  [3] ...but where the cache already works the misses are the unpredictable
      tail: at the ~8GB budget (96.9% base hit) even the in-sample table
      buys +0.6pp hit rate for 14% extra I/O. An aggregate predictor mostly
      re-predicts what the cache already holds.
The runtime's hidden-state lookahead predicts 86% of the actually-routed
ids at top-8 (measured live; llama-nvmoe-logits prints it) because it is
input-dependent — it sees the token, not just the statistics.
"""
import json
import sys
from collections import Counter, OrderedDict, defaultdict

if len(sys.argv) != 2:
    sys.exit(__doc__)

tokens = []  # per token: {layer: [expert ids]}
with open(sys.argv[1]) as f:
    for line in f:
        per_layer = defaultdict(list)
        for layer, expert in json.loads(line):
            per_layer[layer].append(expert)
        tokens.append(dict(per_layer))

n_layers = max(max(t) for t in tokens) + 1
top_k = max(len(v) for t in tokens for v in t.values())
n_expert = max(e for t in tokens for v in t.values() for e in v) + 1
total_experts = n_layers * n_expert
print(f"trace: {len(tokens)} tokens, {n_layers} layers, top-{top_k}, "
      f"{n_expert} experts/layer ({total_experts} total)")

# ---------- [1] same-ids heuristic ----------
inter = cnt = 0
for t in tokens:
    for L in range(n_layers - 1):
        a, b = set(t.get(L, ())), set(t.get(L + 1, ()))
        if a and b:
            inter += len(a & b)
            cnt += 1
chance = top_k * top_k / n_expert
print(f"\n[1] same-token adjacent-layer id overlap: {inter/cnt:.3f}/{top_k} "
      f"expected by chance: {chance:.3f}")

# ---------- [2] conditional table, held-out recall ----------
split = int(len(tokens) * 0.7)


def build_tables(train):
    cond = [defaultdict(Counter) for _ in range(n_layers - 1)]
    freq = [Counter() for _ in range(n_layers)]
    for t in train:
        for L in range(n_layers):
            for e in t.get(L, ()):
                freq[L][e] += 1
        for L in range(n_layers - 1):
            nxt = t.get(L + 1, ())
            for e in t.get(L, ()):
                for e2 in nxt:
                    cond[L][e][e2] += 1
    return cond, freq


def predict(cond_L, cur_ids, M):
    score = Counter()
    for e in cur_ids:
        c = cond_L[e]
        n = sum(c.values()) or 1
        for e2, k in c.items():
            score[e2] += k / n
    return [e for e, _ in score.most_common(M)]


cond, freq = build_tables(tokens[:split])
print(f"\n[2] held-out recall of layer L+1's ids ({len(tokens)-split} test tokens):")
print(f"    {'M':>4} {'cond-table':>11} {'popularity':>11}")
for M in (8, 16, 32):
    hit_c = hit_f = tot = 0
    for t in tokens[split:]:
        for L in range(n_layers - 1):
            actual = set(t.get(L + 1, ()))
            if not actual:
                continue
            tot += len(actual)
            hit_c += len(actual & set(predict(cond[L], t.get(L, ()), M)))
            hit_f += len(actual & {e for e, _ in freq[L + 1].most_common(M)})
    print(f"    {M:>4} {hit_c/tot:>10.1%} {hit_f/tot:>10.1%}")

# ---------- [3] cache-level value ----------
cond_full, _ = build_tables(tokens)  # in-sample: upper bound for the table


def replay(budget_frac, M):
    """LRU over (layer, expert); at layer L optionally prefetch the top-M
    table predictions for L+1 that aren't cached. Returns (hit rate,
    fetches per token, prefetch precision)."""
    slots = int(total_experts * budget_frac)
    cache = OrderedDict()  # key -> True while speculative and not yet used
    hits = refs = fetches = spec = spec_used = 0

    for t in tokens:
        for L in range(n_layers):
            for e in t.get(L, ()):
                refs += 1
                key = (L, e)
                if key in cache:
                    hits += 1
                    if cache[key]:
                        cache[key] = False  # speculative entry's first real use
                        spec_used += 1
                    cache.move_to_end(key)
                else:
                    fetches += 1
                    cache[key] = False
                    if len(cache) > slots:
                        cache.popitem(last=False)
            if M > 0 and L < n_layers - 1:
                for e2 in predict(cond_full[L], t.get(L, ()), M):
                    key = (L + 1, e2)
                    if key not in cache:
                        fetches += 1
                        spec += 1
                        cache[key] = True  # speculative, not yet used
                        if len(cache) > slots:
                            cache.popitem(last=False)
    prec = spec_used / spec if spec else 0.0
    return hits / refs, fetches / len(tokens), prec


print("\n[3] LRU replay with table-driven lookahead-1 prefetch")
print("    (in-sample table = the predictor's best case):")
print(f"    {'budget':>7} {'M':>3} {'hit rate':>9} {'fetch/tok':>10} {'extra I/O':>10} {'precision':>10}")
for frac, label in ((0.24, "~4GB"), (0.48, "~8GB")):
    base_hit, base_ft, _ = replay(frac, 0)
    print(f"    {label:>7} {0:>3} {base_hit:>8.1%} {base_ft:>10.1f} {'—':>10} {'—':>10}")
    for M in (4, 8):
        hit, ft, prec = replay(frac, M)
        print(f"    {label:>7} {M:>3} {hit:>8.1%} {ft:>10.1f} {ft/base_ft-1:>9.1%} {prec:>9.1%}")
