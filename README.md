# nvmoe

**Run flagship-scale MoE models on a 16GB GPU by paging experts from your NVMe SSD — with a hard RAM budget.**

Modern flagship open-weight models (DeepSeek-R1 671B, Qwen3-Next-80B, GPT-OSS-120B) are Mixture-of-Experts: only a few percent of their parameters activate per token. nvmoe's bet is that this makes VRAM a *cache*, not a *container* — the model's bulk lives on a fast SSD, and the GPU holds only what's hot. No retraining, no new quantization format, no 128GB workstation. A 16GB card, a PCIe 4.0 NVMe, and about 4GB of RAM.

## Who is this for

- **You have a 8-24GB GPU and want to run models that don't fit.** The measured result below shows a model slightly too big for VRAM running at a 99.6% cache hit rate — effectively VRAM speed.
- **You want the big ones.** DeepSeek-R1 671B (dynamic 1.58-bit, 131GB on disk) pencils out to ~2 tok/s on a 16GB card + consumer SSD. Slow, but it *runs*, on hardware that costs less than the RAM other approaches require.
- **Your RAM is spoken for.** Every existing offload system (KTransformers, Fiddler, llama.cpp's `exps=CPU`) parks expert weights in system RAM. If your machine runs a Docker stack, a game, or VMs, that RAM isn't free. nvmoe bypasses the page cache entirely (O_DIRECT) and caps host memory use.

## The idea in plain English

When a MoE model generates a token, each layer picks a handful of "experts" (say 8 of 128) and ignores the rest. Which experts get picked is heavily skewed (some are popular) and sticky (consecutive tokens reuse them). Skewed + sticky = cacheable. So:

1. Dense weights, KV cache, and an **expert cache** live in VRAM.
2. All routed experts live on **NVMe**, read on demand with io_uring + O_DIRECT.
3. A small **pinned staging buffer** (the only host RAM used, ~4GB cap) lands the reads.
4. A **prefetcher** predicts upcoming experts and hides SSD latency behind compute.

Decode speed then depends on one number: how many bytes of experts miss the cache per token, divided into your SSD's bandwidth. Everything in this repo exists to measure, predict, and eventually minimize that number.

## Quickstart — 10 minutes, no GPU needed

Everything below is Python 3.10+ stdlib. Nothing to install.

```bash
git clone https://github.com/BrianBoitano/nvmoe && cd nvmoe

# 1. Simulate the built-in model presets on the reference box (16GB + 7GB/s NVMe)
python3 sim/run_sim.py --all

# 2. Simulate YOUR hardware
python3 sim/run_sim.py --model gpt-oss-120b --vram-gb 24 --nvme-gbps 12

# 3. Measure YOUR SSD at expert-sized reads (Linux; point it at any large file on the drive)
python3 tools/nvme_probe.py /path/to/any/big/file
```

Step 3's output is the honest version of your SSD's spec sheet: O_DIRECT random reads at 2-99MB blocks, the exact I/O pattern the runtime will use. Feed the result back into step 2 via `--nvme-gbps`.

### Optional: trace a real model (~30 min, still no GPU)

The simulator ships with real traces (`traces/*.tokens.jsonl`), but collecting your own is the fun part:

```bash
git clone https://github.com/ggml-org/llama.cpp
./collector/install.sh ./llama.cpp     # adds + builds the llama-nvmoe-trace tool

# any MoE GGUF works; small ones trace fine on CPU
BIN=./llama.cpp/build/bin/llama-nvmoe-trace \
MODEL=/path/to/Qwen3-30B-A3B-Q4_K_M.gguf \
bash tools/collect_qwen_traces.sh

# then simulate with YOUR trace on YOUR hardware
python3 sim/run_sim.py --model qwen3-30b-a3b --trace traces/qwen3-all.tokens.jsonl --vram-gb 16
python3 sim/calibrate.py --model qwen3-30b-a3b --trace traces/qwen3-all.tokens.jsonl
```

### Optional: repack a model into an expert pack (Phase 2 has begun)

The runtime doesn't page experts out of a GGUF — it pages them out of a
**pack**: every `(layer, expert)` as its own 4KB-aligned extent (gate+up+down
together, so a cache miss is exactly one aligned read). The repacker builds
one from any MoE GGUF, offline, losslessly:

```bash
python3 tools/repack_gguf.py models/olmoe-q4_0.gguf
#   -> models/olmoe-q4_0.nvmoe/{resident.gguf, experts.pack, manifest.json}

# prove the repack is byte-identical to the source (every extent, every tensor)
python3 tools/verify_pack.py models/olmoe-q4_0.nvmoe models/olmoe-q4_0.gguf
```

Measured on the reference models: OLMoE-1B-7B (Q4_0) → 1,024 extents of
3.4MB, 92.3% of the file paged; Qwen3-30B-A3B (Q4_K_M) → 6,144 extents of
2.5–2.9MB (mixed Q4_K/Q6_K layers), 94.6% paged. Both verify byte-identical,
0.000% alignment padding, and the repack runs in seconds to tens of seconds.
Format spec: [docs/PACK_FORMAT.md](docs/PACK_FORMAT.md).

## Measured results (reproducible with the commands above)

**NVMe delivers.** Samsung 990 PRO (PCIe 4.0 x4), O_DIRECT random reads at expert-sized blocks: 4.4 GB/s at 2MB/QD1 rising to ~6-7 GB/s at 9MB+, ~8 GB/s at 2MB/QD4. Expert-granular random access costs almost nothing vs sequential.

**A cache miss costs ~0.7ms — disk all the way into VRAM.** io_uring + O_DIRECT random extent fetches from real repacked expert packs (`paging/nvmoe-iobench`), measured both into host RAM and end-to-end into a VRAM slab (`--gpu`: pinned staging + `cudaMemcpyHtoDAsync`, byte-verified): p50 0.7ms / p99 ~1ms per expert at QD1, peaking at **~7 GB/s and ~2,300 experts/s at QD2**, with the GPU hop costing only 2-5% (PCIe hides behind the NVMe reads — the SSD stays the bottleneck, as designed). ~220MB of pinned host RAM sustained the peak: the ≤4GB budget has 10x headroom. One surprise: throughput *falls* past QD4 (multi-MB reads are already parallel inside the SSD), so the prefetcher design keeps 2-4 reads in flight, not the 16-32 first guessed. Details + full tables: [paging/README.md](paging/README.md).

**Real routing is very cacheable.** Traced Qwen3-30B-A3B (48 MoE layers, 128 experts/layer, top-8) across chat, code, story, and summarization — 1,396 decode tokens:

| VRAM cache holds | Real LRU hit rate | Calibrated synthetic |
|---|---|---|
| 5% of experts | 0% (thrash: below per-token working set) | 0% |
| 10% | **48.2%** | 29.6% |
| 25% | **81.4%** | 82.8% |
| 50% | **97.7%** | 97.3% |

Top-10% of experts carry 34.7% of routing traffic; consecutive tokens reuse 43.4% of their expert set. On a 16GB card, Qwen3-30B-A3B's cache holds 83% of its experts → **99.6% hit rate**: a model that doesn't fit in VRAM becomes effectively VRAM-resident.

**The thrash cliff.** A cache smaller than one token's active set (`moe_layers x top_k / total_experts` — 6.3% for Qwen3-30B, only 3.1% for R1's fine-grained 256-expert design) hits exactly 0%. Fine-grained MoE has lower cliffs; any placement planner must check this floor first.

## Simulated decode ceilings (16GB GPU + 7GB/s NVMe)

| Model | Experts on NVMe | Cache budget | Ceiling |
|---|---|---|---|
| Qwen3-Next-80B-A3B (4.5-bit) | 43GB | 12.5GB (29% of experts) | **~66 tok/s** |
| GPT-OSS-120B (MXFP4) | 61GB | 12.0GB (20%) | **~11 tok/s** |
| Mixtral-8x7B (4.5-bit) | 25GB | 13.5GB (53%) | 3.5 tok/s |
| DeepSeek-R1 671B (dyn 1.58-bit) | 129GB | 5.5GB (4.3%) | **~2.4 tok/s** |

These are I/O-bound ceilings (fetches perfectly overlapped with compute); real throughput lands below them. Ultra-sparse fine-grained MoE (Qwen3-Next-class) is genuinely usable from NVMe; R1 671B runs as proof that VRAM capacity is no longer the wall; coarse-expert models (Mixtral's 99MB experts) cache poorly — granularity matters more than parameter count. Dense models (Llama 405B) are ruled out by physics: every weight streams every token.

Prefill is the known weak spot: a long prompt touches nearly every expert (~18s per sweep for R1 at 1.58-bit on 7GB/s). nvmoe is a decode-optimized design.

## Roadmap

- [x] **Phase 0 — cache simulator** (`sim/`): presets, synthetic traces, LRU/pinned policies, tok/s ceilings
- [x] **Phase 1 — real traces + hardware probes** (`collector/`, `tools/`): eval-callback tracer, four-workload suite, generator calibration, NVMe probe. Open item: DeepSeek-V2-Lite traces for R1-style 256-expert routing
- [ ] **Phase 2 — runtime**: llama.cpp fork with an `exps=NVMe` placement path
  - [x] **2.1 offline repacker** (`tools/repack_gguf.py`): any MoE GGUF → resident GGUF + per-expert 4KB-aligned extents + manifest ([format spec](docs/PACK_FORMAT.md)); proven byte-lossless on OLMoE-1B-7B and Qwen3-30B-A3B via `tools/verify_pack.py`
  - [x] **2.2a io_uring extent reader** (`paging/`): raw-syscall io_uring + O_DIRECT benchmark on real packs — 0.7ms per expert miss, ~7GB/s @ QD2
  - [x] **2.2b NVMe→VRAM end-to-end** (`--gpu`): pinned staging + async H2D into a VRAM slab, byte-verified — the GPU hop costs 2-5%; ~6.6GB/s / ~2,300 experts/s to VRAM on ~220MB of host RAM. No CUDA toolkit needed (driver API via dlopen)
  - [ ] 2.3 llama.cpp integration: synchronous fetch-on-miss first (identical logits vs stock as the gate), then router-guided prefetch
  - [ ] 2.4 measured tok/s vs stock full-VRAM on Qwen3-30B-A3B, then GPT-OSS-120B
- [ ] **Phase 3 — planner**: probe hardware, read GGUF metadata, emit the optimal quant + placement plan per model automatically
- [ ] **Stunt flag**: DeepSeek-R1 671B, 16GB VRAM, ≤4GB RAM, on video

## Repo map

```
sim/            cache simulator: presets.py, trace_gen.py, cache_sim.py,
                run_sim.py (CLI), trace_post.py, calibrate.py
paging/         Phase 2 paging library: nvmoe_iobench.c (io_uring extent-fetch
                + NVMe→VRAM benchmark; raw syscalls + dlopen'd CUDA driver
                API, no deps) + measured results
collector/      llama.cpp trace tool (nvmoe-trace.cpp) + install.sh
tools/          repack_gguf.py + verify_pack.py (GGUF → expert pack, Phase 2),
                gguf_lite.py (stdlib GGUF reader/writer),
                nvme_probe.py (SSD bench), collect_qwen_traces.sh
tests/          test_repack.py — full repack round-trip on a synthetic tiny
                MoE GGUF, runs in milliseconds, no model download
prompts/        the four standard trace workloads (ChatML format)
traces/         real routing traces (*.tokens.jsonl committed as samples)
docs/           DESIGN.md (architecture), PACK_FORMAT.md (expert pack spec),
                TRACE_COLLECTION.md (how tracing works)
```

## FAQ

**Why not just offload to RAM like everyone else?** If you have 64-128GB of free RAM, do that — KTransformers and llama.cpp `-ot "exps=CPU"` are excellent. nvmoe is for machines where RAM is occupied or small, and for models bigger than any consumer RAM (Kimi-K2-class at 600GB+ quantized).

**Does this work for dense models?** No, and it can't — a dense model reads every weight for every token, so you'd get SSD-bandwidth-divided-by-model-size (~0.04 tok/s for 405B). MoE sparsity is load-bearing.

**Will 1.58-bit quality be terrible?** 1.58-bit is the stunt tier and shows real degradation. 3-4 bit dynamic quants hold up well in public benchmarks (3-bit DeepSeek V3.1 scored 75.6% vs 76.1% unquantized on Aider Polyglot). The runtime is quant-agnostic; it pages whatever GGUF you give it.

**Can I run this today?** The simulator, SSD probe, trace collector, and the expert-pack repacker: yes, today — that's Phases 0-1 and the first piece of Phase 2. The paging runtime itself is under active development. Star/watch the repo if you want the moment it lands.

**Windows/macOS?** The simulator runs anywhere Python does. The NVMe probe and the planned runtime are Linux-first (io_uring, O_DIRECT). macOS unified memory largely doesn't need this; Windows support would need a different I/O backend.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — adding a model preset, tracing a new model, or extending cache policies are all small, well-bounded first contributions. If you know llama.cpp internals around `build_moe_ffn`, or have io_uring/CUDA experience, Phase 2 needs you.

## License

MIT
