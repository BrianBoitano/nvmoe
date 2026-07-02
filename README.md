# nvmoe

**Run flagship-scale MoE models on 16GB of VRAM by paging experts from NVMe — with a hard RAM budget.**

Every existing hybrid-inference system (KTransformers, Fiddler, HOBBIT, MoE-Infinity, llama.cpp's `--override-tensor "exps=CPU"`) assumes the model's expert weights live in system RAM. That assumption breaks on real machines: a home server running a Docker stack, a workstation with 32GB, a gaming PC where RAM belongs to the game. nvmoe treats **NVMe as the expert store and VRAM as the expert cache**, so a 671B-parameter model needs a 16GB GPU, a fast SSD, and almost nothing else.

## Why this works

Modern flagship open-weight models (DeepSeek-R1 671B, Qwen3-Next-80B, GPT-OSS-120B, Kimi K2) are Mixture-of-Experts: only a few percent of parameters activate per token. The dense parts (attention, shared experts, KV cache) fit comfortably in 16GB. The routed experts — 88%+ of the weights — are accessed sparsely, with heavy popularity skew and strong token-to-token reuse. That is a caching workload, and VRAM is the cache.

The literature already proved the pieces (all software-only, no retraining):

- Expert-aware caching and prefetching: MoE-Infinity (3.1-16.7x latency gains), HOBBIT (up to 9.9x), PreScope
- CPU/GPU hybrid scheduling: KTransformers (SOSP '25), Fiddler (ICLR '25)
- Structure-aware quantization: Unsloth dynamic quants (R1: 720GB → 131GB at ~1.58-bit experts, minimal quality loss at 3-4 bit)
- Flash-tier weight streaming: Apple's "LLM in a Flash", PowerInfer-2 (phones only, never shipped for desktop GPUs)

Nobody has composed them into **NVMe → VRAM expert paging with a bounded RAM footprint on consumer hardware**. That is nvmoe.

## Design guarantees

1. **VRAM is the primary memory.** Dense weights + KV cache + expert cache live on the GPU.
2. **NVMe is the weight store.** Routed experts are read with io_uring + O_DIRECT (bypassing the OS page cache) straight into pinned staging buffers.
3. **Hard RAM budget, default 4GB.** The only host RAM used is the staging ring buffer and bookkeeping. Enforceable with cgroups. Your Docker stack keeps its memory.

## Simulated ceilings (RTX 5070 Ti 16GB + Samsung 990 PRO, PCIe 4.0 x4)

Output of `python3 sim/run_sim.py --all` — I/O-bound decode ceilings assuming fetches overlap compute:

| Model | Experts on NVMe | Cache budget | Hit rate (LRU+pin) | Ceiling |
|---|---|---|---|---|
| Qwen3-Next-80B-A3B (4.5-bit) | 43GB | 12.5GB (29% of experts) | 87.5% | **66 tok/s** |
| GPT-OSS-120B (MXFP4) | 61GB | 12.0GB (20%) | 67.8% | **11.4 tok/s** |
| Mixtral-8x7B (4.5-bit) | 25GB | 13.5GB (53%) | 68.7% | 3.5 tok/s |
| DeepSeek-R1 671B (dyn 1.58-bit) | 129GB | 5.5GB (4.3%) | 27.3% | **2.4 tok/s** |

Read that table as the thesis: ultra-sparse MoE (Qwen3-Next-class) is genuinely *usable* from NVMe on a 16GB card, GPT-OSS-120B is workable, and full R1 671B runs as a proof that the ceiling on "what can this GPU hold" is gone. Coarse-expert models like Mixtral cache poorly (99MB experts) — sparsity granularity matters more than parameter count.

Caveats, stated plainly: these are ceilings from a *synthetic* trace (Zipf popularity + temporal locality, conservative defaults) — see the measured Phase 1 results below, which say the synthetic assumptions were conservative. Prefill is a separate problem: a long prompt sweeps nearly all experts (~18s per pass for R1). Real decode throughput will land below ceiling due to imperfect overlap.

## Measured results (Phase 1)

**NVMe delivers.** `tools/nvme_probe.py` on a Samsung 990 PRO (PCIe 4.0 x4), O_DIRECT random reads at expert-sized blocks: 4.4 GB/s at 2MB/QD1 rising to ~6-7 GB/s at 9MB+, and ~8 GB/s at 2MB/QD4. Expert-granular random access costs almost nothing vs sequential — the 7 GB/s planning number is real.

**Real routing traces beat the synthetic assumptions.** Using `collector/nvmoe-trace.cpp` (a llama.cpp eval-callback tool that logs every `ffn_moe_topk` selection), we traced Qwen3-30B-A3B (48 MoE layers, 128 experts, top-8) across four workloads — chat, code, story, summarization; 1,396 decode tokens:

| VRAM cache holds | Real LRU hit rate | Calibrated synthetic |
|---|---|---|
| 5% of experts | 0% (thrash: below per-token working set) | 0% |
| 10% | **48.2%** | 29.6% |
| 25% | **81.4%** | 82.8% |
| 50% | **97.7%** | 97.3% |

Top-10% of experts carry 34.7% of routing traffic; consecutive tokens reuse 43.4% of their expert set. On the 16GB target box, Qwen3-30B-A3B's cache holds 83% of its experts → **99.6% hit rate**: a model that doesn't fit VRAM becomes effectively VRAM-resident. That is the second product story: not just "run 671B at all," but "run the model one size up from your card at near-full speed."

**The thrash cliff is a planner constraint.** A cache smaller than one token's active set (`moe_layers x top_k / total_experts` — 6.3% for Qwen3-30B, only 3.1% for DeepSeek-R1's fine-grained 256-expert design) hits exactly 0%. Fine-grained MoE architectures have *lower* cliffs, which is why R1's tiny 4.3%-of-experts cache still helps. Any auto-planner must check this floor before promising anything.

## Roadmap

- [x] **Phase 0 — cache simulator** (`sim/`): model presets, synthetic traces, LRU / pinned-hot-expert policies, tok/s ceilings
- [x] **Phase 1 — real traces + hardware probes**: `collector/` eval-callback tracer, four-workload suite, generator calibration (`sim/calibrate.py`), NVMe probe (`tools/nvme_probe.py`) — measured results above. Still open: traces from a DeepSeek-family proxy (V2-Lite) for R1-style 256-expert routing
- [ ] **Phase 2 — runtime**: fork llama.cpp; add an `exps=NVMe` tensor-placement path — packed expert extents on disk, io_uring reader, pinned staging ring, VRAM LRU cache, router-guided prefetch (cross-layer expert prediction)
- [ ] **Phase 3 — planner**: probe hardware (VRAM, NVMe bandwidth, CPU flags), read GGUF metadata, and emit the optimal placement plan per model automatically
- [ ] **Stunt flag**: DeepSeek-R1 671B, 16GB VRAM, ≤4GB RAM, on video

## Quick start (simulator)

```bash
python3 sim/run_sim.py --all                      # every preset
python3 sim/run_sim.py --model deepseek-r1-671b   # one model
python3 sim/run_sim.py --model qwen3-next-80b --locality 0.7 --zipf-s 1.2
python3 sim/run_sim.py --model mixtral-8x7b --trace traces/real.jsonl
```

No dependencies — Python 3.10+ stdlib only.

## Status

Early. Phase 0 works today; the runtime does not exist yet. If you know the llama.cpp internals around `build_moe_ffn` or have io_uring/GPUDirect experience, issues and PRs are very welcome.

## License

MIT
