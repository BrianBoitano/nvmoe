# nvmoe design

## Problem statement

Run flagship-scale open-weight MoE models (80B-671B+) on a machine whose only
abundant resources are a 16GB GPU and a fast NVMe SSD. System RAM is treated as
scarce (target: hard cap of ~4GB), because on real machines it belongs to other
workloads (Docker stacks, games, VMs).

## Memory tiers and their physics

| Tier | Size (target box) | Bandwidth | Role in nvmoe |
|---|---|---|---|
| VRAM (RTX 5070 Ti) | 16GB | ~896GB/s | dense weights, KV cache, expert cache |
| Pinned host staging | ≤4GB | PCIe 4.0 x16, ~26GB/s effective | DMA landing zone for NVMe reads |
| NVMe (990 PRO, PCIe 4.0 x4) | 2TB | ~7GB/s seq read | expert store (packed extents) |

Decode is bandwidth-bound: each token must materialize `moe_layers x top_k`
routed experts. Every expert that misses the VRAM cache costs an NVMe read.
The entire system design reduces to maximizing cache hit rate and hiding the
miss latency behind compute.

Two hard truths accepted up front:

1. Literally zero host RAM is impossible. The GPU cannot DMA from a consumer
   NVMe directly (GeForce lacks true GPUDirect Storage P2P; cuFile falls back
   to a bounce buffer anyway). So the honest guarantee is a bounded, small,
   pinned staging area — not zero.
2. Prefill sweeps experts. With hundreds of prompt tokens per ubatch, nearly
   every expert in every layer activates at least once, so prefill degenerates
   to streaming the whole expert store once per ubatch (~18s for R1 at
   1.58-bit on 7GB/s NVMe). Mitigations: large ubatch (amortize one sweep over
   many tokens — sequential streaming, NVMe's best case), prompt caching, and
   accepting that nvmoe is a decode-optimized system.

## Architecture

```
GGUF file(s)                     ┌──────────────────────────────┐
     │  offline repack           │            VRAM              │
     ▼                           │  dense/attn/shared (resident)│
expert extents on NVMe           │  KV cache                    │
(one aligned extent per expert,  │  expert cache (LRU + pinned  │
 grouped by layer, O_DIRECT-     │   hot set, quant-tiered)     │
 friendly alignment)             └──────▲───────────────────────┘
     │                                  │ cudaMemcpyAsync (H2D)
     ▼                                  │
io_uring reader (QD 2-8)  ──► pinned staging ring (≤4GB) ──────┘
     ▲
     │ fetch queue
router-guided prefetcher (layer L router output + cross-layer
correlation predicts experts for layers L+1..L+k; issues reads
k layers ahead so I/O overlaps attention/dense compute)
```

### Components

1. **Offline repacker.** Parse the GGUF, split routed-expert tensors into
   per-expert extents, write them padded to 4KB alignment in layer order.
   Dense/attention/shared-expert tensors stay in a normal GGUF loaded to VRAM.
   Also emits a manifest (offsets, sizes, quant type per expert) so mixed-bit
   experts (HOBBIT-style: hot experts at higher precision) are possible later.

2. **VRAM expert cache.** Fixed-size slab pool of expert-sized slots.
   Eviction: LRU with a pinned tier for globally hot experts (from offline
   profiling or online frequency counting). Simulator (Phase 0) says pinning
   adds ~3-8pp hit rate over plain LRU; revisit with real traces.

3. **Prefetcher.** The routing decision for layer L is known after layer L's
   router runs — too late to hide a 1-10ms read. **Built (Phase 2.3c-ii,
   fourth patch):** lookahead-1 by applying layer L+1's router (with the
   RMS-norm weight ratio folded in offline) to layer L's FFN input —
   measured to contain ~86% of the actually-routed ids at top-8. Two
   candidate predictors were evaluated against the Phase 1 traces first and
   rejected (`tools/analyze_lookahead.py`): same-id positional overlap is
   exactly chance, and offline correlation tables can't predict the misses
   that matter (the tail). Wider and deeper speculation both measured
   slower — wasted extents cost more than hidden misses on an I/O-bound
   pipeline — and lookahead auto-disables at near-resident budgets. See
   docs/INTEGRATION.md for the mechanism.

4. **io_uring reader.** O_DIRECT, registered buffers, queue depth tuned to
   saturate the SSD at expert-sized reads (1.8-13MB in target models — large
   enough to get near-sequential throughput). Page cache is bypassed
   deliberately: it would silently consume the RAM we promised not to take.
   Measured (paging/, Phase 2.2a): at 2.5-3.4MB extents the sweet spot is
   shallow — ~7GB/s at QD2, and throughput *drops* past QD4 while latency
   grows linearly. Multi-MB reads are already parallel inside the device;
   the prefetcher should keep 2-4 in flight, not 16-32 as first guessed.

5. **Runtime integration.** Fork/patch llama.cpp: introduce an `exps=NVMe`
   placement in the `--override-tensor` machinery. The MoE FFN path
   (`build_moe_ffn`) gains a cache-lookup + fetch-wait step per layer. Start
   with the simplest correct thing (synchronous fetch on miss), then add
   prefetch overlap, then mixed precision.

### RAM budget enforcement

Everything host-side lives in one allocation domain: staging ring + manifest +
predictor state. This is a headline feature, not an implementation detail —
"runs with Docker eating the rest of your RAM" is the differentiator vs every
exps=CPU setup. **Proven (Phase 2.4):** the full Qwen3-30B decode runs inside
a Docker `--memory=4g --memory-swap=4g` cgroup at full speed with a measured
peak of 756MB host memory (see runtime/README.md for the command).

## What nvmoe is NOT

- Not a new quantization method — it consumes Unsloth/llama.cpp quants as-is.
- Not a training/distillation system — layer on top of existing checkpoints.
- Not a datacenter batch-inference engine — batch size 1-4, local, single GPU.

## Model fit guide (from Phase 0 simulation)

- **Best fit:** ultra-sparse, fine-grained experts (Qwen3-Next-80B-A3B:
  1.8MB experts, 29% cacheable in 12.5GB, 87% hit rate, ~66 tok/s ceiling).
- **Good fit:** GPT-OSS-120B (MXFP4, 20% cacheable, ~11 tok/s ceiling).
- **Stunt fit:** DeepSeek-R1 671B at dynamic 1.58-bit (4% cacheable,
  ~2.4 tok/s ceiling) — proves capacity, not a daily driver.
- **Poor fit:** coarse-expert MoE (Mixtral: 99MB experts — each miss is a
  14ms read; cache granularity too coarse) and all dense models (405B dense
  would stream every weight every token: ~0.04 tok/s. Architecture, not
  engineering, rules dense out).

## Risks

- llama.cpp upstream could ship direct-IO expert paging first (there are
  open discussions about mmap alternatives). nvmoe therefore ships as a
  standalone repo + patch set against llama.cpp, not as a PR-first project.
  Note: llama.cpp's contribution policy (AGENTS.md/CONTRIBUTING.md) does not
  accept predominantly AI-generated PRs — private forks are explicitly
  exempt, which is what nvmoe is. Any future upstream submission must be
  human-authored and human-owned.
- Consumer NVMe sustained-read behavior under mixed-size random reads may
  land under 7GB/s; measure early (Phase 1 includes an fio profile of the
  990 PRO at expert-sized reads).
- Synthetic-trace hit rates may be optimistic; real traces are milestone 1
  for exactly this reason.
