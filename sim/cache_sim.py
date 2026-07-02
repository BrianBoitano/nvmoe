"""VRAM expert-cache simulation over a routing trace.

Policies:
  lru        — plain LRU over (layer, expert) keys.
  lru+pin    — half the budget permanently pins the globally hottest experts
               (measured over the trace itself, so this is an ORACLE upper
               bound for offline hot-expert profiling a la PowerInfer),
               remaining budget runs LRU.

Output is a hit rate; run_sim.py converts that into bytes-per-token and an
I/O-bound decode ceiling for a given NVMe bandwidth.
"""

from collections import Counter, OrderedDict
from dataclasses import dataclass


@dataclass
class CacheResult:
    policy: str
    capacity_experts: int
    accesses: int
    hits: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0


def simulate_lru(
    trace: list[list[tuple[int, int]]],
    capacity_experts: int,
    pinned: frozenset[tuple[int, int]] = frozenset(),
    warmup_tokens: int = 200,
) -> CacheResult:
    """Run the trace through a pinned-set + LRU cache.

    Hits/accesses are only counted after `warmup_tokens` so a cold cache does
    not understate steady-state behavior (local chat sessions are long-lived).
    """
    lru_capacity = max(capacity_experts - len(pinned), 0)
    lru: OrderedDict[tuple[int, int], None] = OrderedDict()
    accesses = hits = 0

    for i, token in enumerate(trace):
        counting = i >= warmup_tokens
        for key in token:
            if counting:
                accesses += 1
            if key in pinned:
                if counting:
                    hits += 1
                continue
            if key in lru:
                lru.move_to_end(key)
                if counting:
                    hits += 1
            elif lru_capacity > 0:
                lru[key] = None
                if len(lru) > lru_capacity:
                    lru.popitem(last=False)

    policy = "lru+pin" if pinned else "lru"
    return CacheResult(policy, capacity_experts, accesses, hits)


def hottest_experts(trace: list[list[tuple[int, int]]], n: int) -> frozenset[tuple[int, int]]:
    counts = Counter(key for token in trace for key in token)
    return frozenset(key for key, _ in counts.most_common(n))
