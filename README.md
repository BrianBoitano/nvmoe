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

Caveats, stated plainly: these are ceilings from a *synthetic* trace (Zipf popularity + temporal locality, conservative defaults) — real routing traces are milestone 1. Prefill is a separate problem: a long prompt sweeps nearly all experts (~18s per pass for R1). Real decode throughput will land below ceiling due to imperfect overlap and sub-peak NVMe reads at expert-sized granularity.

## Roadmap

- [x] **Phase 0 — cache simulator** (`sim/`): model presets, synthetic traces, LRU / pinned-hot-expert policies, tok/s ceilings
- [ ] **Phase 1 — real traces**: instrument llama.cpp's eval callback to log expert routing (docs/TRACE_COLLECTION.md), calibrate the generator, re-rank policies with measured hit rates
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
