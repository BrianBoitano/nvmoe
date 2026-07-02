# nvmoe

**Run flagship-scale MoE models on a 16GB GPU by paging experts from your NVMe SSD — with a hard RAM budget.**

Modern flagship open-weight models (DeepSeek-R1 671B, Qwen3-Next-80B, GPT-OSS-120B) are Mixture-of-Experts: only a few percent of their parameters activate per token. nvmoe's bet is that this makes VRAM a *cache*, not a *container* — the model's bulk lives on a fast SSD, and the GPU holds only what's hot. No retraining, no new quantization format, no 128GB workstation. A 16GB card, a PCIe 4.0 NVMe, and about 4GB of RAM.

## Who is this for

- **You have a 8-24GB GPU and want to run models that don't fit.** Measured, not simulated: Qwen3-30B-A3B (18.6GB, doesn't fit a 16GB card) decodes at **166 tok/s** from its NVMe pack — 2.1x the best llama.cpp offload split — using 756MB of host RAM. **GPT-OSS-120B (63GB) decodes at 24.5 tok/s** on the same card, ~3x the best stock configuration this box can attempt, inside a 4GB memory cgroup ([tables](runtime/README.md)).
- **You want a flagship you'll actually use.** Qwen3-Next-80B-A3B (48.5GB) decodes at **44.8 tok/s** from its pack on the 16GB reference card — measured, warm, with ~1GB of host RAM. The 80-120B ultra-sparse tier is this design's sweet spot: big enough to be a real flagship, sparse enough to cache. (The 671B-class runs too — DeepSeek-R1 pencils out to ~2 tok/s — but flat DeepSeek-family routing makes that a capacity proof, not a daily driver.)
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

# 4. Plan a specific model on your hardware (reads the GGUF's MoE geometry)
python3 tools/plan.py /path/to/model.gguf --vram-gb 16 --nvme-gbps 7
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

### Optional: run a model from its pack (Phase 2.3)

The `runtime/` patch series teaches llama.cpp to load a pack directly — the
routed experts are fetched from `experts.pack` on demand instead of loaded:

```bash
./runtime/apply.sh && cd llama.cpp-nvmoe
cmake -B build && cmake --build build -j --target llama-nvmoe-logits llama-bench

# any tool, no new flags -- the pack is detected from resident.gguf's metadata
./build/bin/llama-bench -m ../models/olmoe-q4_0.nvmoe/resident.gguf -p 16 -n 32
NVMOE_CACHE_MB=512 ./build/bin/llama-bench -m ...    # cap the expert cache
```

The correctness bar is the strongest one available: **bit-identical logits
vs stock llama.cpp**, verified over dozens of greedy-decode steps on both
reference packs, including under heavy cache eviction. Reproduce it with the
commands in [runtime/README.md](runtime/README.md); how the integration
works is in [docs/INTEGRATION.md](docs/INTEGRATION.md).

### The planner: will *your* model run, and how fast? (Phase 3)

`tools/plan.py` turns everything this repo measured into an answer for a
model it has never seen. Give it a GGUF (it reads the MoE geometry directly)
and your hardware numbers; it checks whether the model just fits (then use
stock llama.cpp — paging costs ~13% when you don't need it), checks the
thrash-cliff floor, picks the routing-family hit curve by architecture,
and prints the expected decode range plus every command to get there:

```bash
python3 tools/plan.py models/qwen3-30b-a3b-q4_k_m.gguf                # a real file
python3 tools/plan.py --preset deepseek-r1-671b                       # not downloaded yet
python3 tools/plan.py model.gguf --vram-gb 24 --nvme-gbps 12          # your box
python3 tools/plan.py --postdict                                      # its receipts
```

`--postdict` is the honesty check: the planner re-predicts the eleven
configurations measured in [runtime/README.md](runtime/README.md) and prints
predicted vs measured side by side. Its hit-rate model lands within a few
points of live runtime counters on real prompts; measured `llama-bench`
decode lands 0.7-2.5x of the predicted tok/s, and the tail is systematic —
`-p 0` benchmark generation routes far more repetitively than real
workloads, most of all on flat-routing DeepSeek-family models and at small
caches on fine-grained-expert models. The planner predicts real use, not
benchmark flattery, and reports that spread as its error bars.

## Measured results (reproducible with the commands above)

**The runtime works, end to end.** Qwen3-30B-A3B Q4_K_M (18.6GB GGUF, does not fit in 16GB VRAM) decodes at **166.1 ± 2.2 tok/s** from its pack with a 12GB VRAM expert cache and synchronous fetch-on-miss — vs 79.2 for stock llama.cpp's best partial offload and 45.2 for the experts-in-RAM recipe (which needs ~17GB of free host RAM; nvmoe used **756MB peak, measured inside a hard 4GB cgroup**). Cache-budget sweep, prefill caveat, and every command: [runtime/README.md](runtime/README.md). Correctness bar: **bit-identical logits vs stock**, CPU and CUDA, including under heavy cache eviction.

**The sweet spot is real: an 80B flagship at reading speed.** Qwen3-Next-80B-A3B (Q4_K_M, 48.5GB, ~3B active) — the model the planner picked as this design's best fit — decodes at **44.8 ± 2.6 tok/s** warm from its pack at an 11.5GB VRAM cache: 2.0x the best stock partial offload and 2.1x the experts-in-RAM recipe (which wants ~44GB of RAM; the pack run peaked at **1.01GB, measured in a hard 4GB cgroup at full speed**). It is the first hybrid-attention architecture through the pack (bit-identical logits, CPU and split-offload CUDA) and the first model where lookahead prefetch pays at every cache budget (+12-17% — its 1.9MB extents are exactly what the prefetcher was built for). Tables and commands: [runtime/README.md](runtime/README.md).

**It scales to models 4x the card.** GPT-OSS-120B (MXFP4, 63GB GGUF, 5.1B active) decodes at **24.5 ± 3.3 tok/s** with an 11GB VRAM expert cache — 2.2x the simulator's conservative ceiling and ~3x the best stock configuration this 64GB-RAM box can attempt (`exps=CPU` manages 8.3 ± 4.0 while monopolizing ~57GB of page cache; nvmoe's whole decode ran inside a **hard 4GB cgroup at full speed**, 3.1GB peak). The pack: 4,608 extents of 12.6MB, 96.1% of the file paged, byte-identical repack in 68s, logits gate passed.

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

**...but routing flatness is an architecture property.** The same four-workload trace on DeepSeek-V2-Lite (26 layers, 64 experts, top-6 — the small proxy for R1-style fine-grained routing) tells a different story: top-10% experts carry only **17.7%** of traffic, consecutive tokens reuse just **24.2%** of their experts, and the LRU curve sits roughly half as high — 23.2% hits at a 10% cache and 40.6% at 25% (vs Qwen3's 48.2% / 81.4%). Calibrated `zipf_s` drops from ~1.0 to 0.3. DeepSeek-family models pay for their fine granularity with much flatter routing: they need proportionally more cache for the same hit rate, and any R1-class plan must budget with these curves, not Qwen3's.

**GPT-OSS-120B is the opposite extreme — and it explains the 2.2x.** Its four-workload trace (collected at full speed on the GPU *through the pack*, since a 63GB model can't be traced on CPU — see [docs/TRACE_COLLECTION.md](docs/TRACE_COLLECTION.md)) measures top-10% experts carrying **56.7%** of traffic and **50.4%** token-to-token overlap — the most cacheable routing of the three families, well beyond Qwen3's. That is why its measured 24.5 tok/s beat the synthetic simulator's ~11 ceiling by 2.2x: top-4-of-128 routing at only 36 layers is heavily concentrated in practice.

**The thrash cliff.** A cache smaller than one token's active set (`moe_layers x top_k / total_experts` — 6.3% for Qwen3-30B, 9.4% for V2-Lite, only 3.1% for R1's fine-grained 256-expert design) hits exactly 0% — the V2-Lite trace confirms it on a second architecture (0.0% at a 5% cache). Fine-grained MoE has lower cliffs; any placement planner must check this floor first.

## Simulated decode ceilings (16GB GPU + 7GB/s NVMe)

| Model | Experts on NVMe | Cache budget | Ceiling |
|---|---|---|---|
| Qwen3-Next-80B-A3B (4.5-bit) | 43GB | 12.5GB (29% of experts) | **~66 tok/s** |
| GPT-OSS-120B (MXFP4) | 61GB | 12.0GB (20%) | ~11 tok/s — **measured: 24.5** (real routing caches far better than the synthetic model) |
| Mixtral-8x7B (4.5-bit) | 25GB | 13.5GB (53%) | 3.5 tok/s |
| DeepSeek-R1 671B (dyn 1.58-bit) | 129GB | 5.5GB (4.3%) | **~2.4 tok/s** |

These are I/O-bound ceilings (fetches perfectly overlapped with compute); real throughput lands below them. Ultra-sparse fine-grained MoE (Qwen3-Next-class) is genuinely usable from NVMe; R1 671B runs as proof that VRAM capacity is no longer the wall; coarse-expert models (Mixtral's 99MB experts) cache poorly — granularity matters more than parameter count. Dense models (Llama 405B) are ruled out by physics: every weight streams every token.

Prefill is the known weak spot: a long prompt touches nearly every expert (~18s per sweep for R1 at 1.58-bit on 7GB/s). nvmoe is a decode-optimized design.

## Roadmap

- [x] **Phase 0 — cache simulator** (`sim/`): presets, synthetic traces, LRU/pinned policies, tok/s ceilings
- [x] **Phase 1 — real traces + hardware probes** (`collector/`, `tools/`): eval-callback tracer, four-workload suite, generator calibration, NVMe probe. DeepSeek-V2-Lite traces measured 2026-07-02: fine-grained routing is ~half as cacheable as Qwen3's (see above) — the R1 planning input
- [ ] **Phase 2 — runtime**: llama.cpp fork with an `exps=NVMe` placement path
  - [x] **2.1 offline repacker** (`tools/repack_gguf.py`): any MoE GGUF → resident GGUF + per-expert 4KB-aligned extents + manifest ([format spec](docs/PACK_FORMAT.md)); proven byte-lossless on OLMoE-1B-7B and Qwen3-30B-A3B via `tools/verify_pack.py`
  - [x] **2.2a io_uring extent reader** (`paging/`): raw-syscall io_uring + O_DIRECT benchmark on real packs — 0.7ms per expert miss, ~7GB/s @ QD2
  - [x] **2.2b NVMe→VRAM end-to-end** (`--gpu`): pinned staging + async H2D into a VRAM slab, byte-verified — the GPU hop costs 2-5%; ~6.6GB/s / ~2,300 experts/s to VRAM on ~220MB of host RAM. No CUDA toolkit needed (driver API via dlopen)
  - [x] **2.3a llama.cpp integration, CPU correctness** (`runtime/`): pack loading + expert-cache pools + sync fetch-on-miss behind an unchanged `ggml_mul_mat_id`; gate = **bit-identical logits vs stock** on OLMoE-1B-7B and Qwen3-30B-A3B, including under heavy cache eviction ([how it hooks in](docs/INTEGRATION.md))
  - [x] **2.3b GPU path**: pools live on the layer's device (VRAM under `-ngl`, split per device on partial offload), fetches DMA through a pinned bounce — pure ggml-backend API, no CUDA code in the runtime; **bit-identical logits on the CUDA backend** (full offload, heavy VRAM eviction, and mixed CPU+GPU pools)
  - [x] **2.3c(i) overlapped miss fetches**: an op's batched misses fetch at QD 2-4 (the measured optimum) — +25% decode at a 4GB cache, logits still bit-identical
  - [x] **2.3c(ii) router-logit lookahead prefetch**: predict layer L+1's experts from layer L's hidden state (one folded matmul per layer — **~86% of routed ids at top-8**, measured) and fetch them during L's compute via a priority fetch pool; +4-7% decode at fetch-bound budgets, auto-off near-resident, still bit-identical. The offline-table alternative was evaluated against real traces and rejected (`tools/analyze_lookahead.py`)
  - [x] **2.4 measured tok/s on Qwen3-30B-A3B** ([tables + commands](runtime/README.md)): **166 tok/s decode from the pack at a 12GB VRAM cache — 2.1x the best stock offload split, 3.7x the exps-in-RAM recipe — with 756MB peak host RAM, proven inside a hard 4GB cgroup.**
  - [x] **2.4b GPT-OSS-120B** ([tables + commands](runtime/README.md)): a 63GB model on the 16GB card — **24.5 tok/s decode at an 11GB cache, ~3x the best stock attempt, inside a 4GB cgroup**; the sweep (10.5/18.7/24.5 at 4/8/11GB) tracks the hit-rate curves, and its 12.6MB extents produced the extent-size speculation gate (wasted prefetch guesses cost more than they hide on fetch-bound decode)
- [x] **Phase 3 — planner** (`tools/plan.py`): reads any MoE GGUF's geometry, takes your VRAM/SSD numbers, and emits the placement plan — cache budget, prefetch setting, expected decode range, and the exact repack/verify/gate/bench commands. Validated by postdiction: `--postdict` reprints its predictions against all eleven measured configurations (0.7-2.5x, with the systematic direction explained there)
- [x] **The usable-flagship tier — Qwen3-Next-80B-A3B** ([tables + commands](runtime/README.md)): the planner's pick, measured same-day — **44.8 tok/s warm decode at an 11.5GB cache, 2.0x the best stock offload, at full speed inside a 4GB cgroup with a 1.01GB peak**; first hybrid-attention architecture through the pack (bit-identical CPU and split-offload GPU), and the first model where lookahead prefetch pays at every budget (+12-17%)
- [ ] **Second sweet-spot model**: GLM-4.5-Air 106B-A12B (`glm4moe`, supported by the pinned base) — a routing family this repo hasn't traced yet

*(A DeepSeek-R1-671B run was on the roadmap as a capacity stunt. The planner's own postdiction killed it: flat DeepSeek-family routing + a cache budget pinned at its 3.1% thrash cliff pencils out to 1-3 tok/s at visible 1.58-bit quality loss — it would prove VRAM isn't the wall, but nobody would use it. The 16GB card's real win is the tier below, where decode is faster than reading speed.)*

## Repo map

```
sim/            cache simulator: presets.py, trace_gen.py, cache_sim.py,
                run_sim.py (CLI), trace_post.py, calibrate.py
paging/         Phase 2 paging library: nvmoe_iobench.c (io_uring extent-fetch
                + NVMe→VRAM benchmark; raw syscalls + dlopen'd CUDA driver
                API, no deps) + measured results
runtime/        the llama.cpp exps=NVMe patch series + apply.sh (Phase 2.3)
                and the identical-logits gate procedure
collector/      llama.cpp trace tool (nvmoe-trace.cpp) + install.sh
tools/          plan.py (Phase 3 planner: GGUF + your hardware → placement
                plan, expected tok/s, and the commands; --postdict validates
                it against every measured configuration),
                repack_gguf.py + verify_pack.py (GGUF → expert pack, Phase 2),
                gguf_lite.py (stdlib GGUF reader/writer),
                nvme_probe.py (SSD bench), collect_qwen_traces.sh
tests/          test_repack.py — full repack round-trip on a synthetic tiny
                MoE GGUF, runs in milliseconds, no model download
prompts/        the four standard trace workloads (ChatML format)
traces/         real routing traces (*.tokens.jsonl committed as samples)
docs/           DESIGN.md (architecture), PACK_FORMAT.md (expert pack spec),
                INTEGRATION.md (llama.cpp hook points),
                TRACE_COLLECTION.md (how tracing works)
```

## FAQ

**Why not just offload to RAM like everyone else?** If you have 64-128GB of free RAM, do that — KTransformers and llama.cpp `-ot "exps=CPU"` are excellent. nvmoe is for machines where RAM is occupied or small, and for models bigger than any consumer RAM (Kimi-K2-class at 600GB+ quantized).

**Does this work for dense models?** No, and it can't — a dense model reads every weight for every token, so you'd get SSD-bandwidth-divided-by-model-size (~0.04 tok/s for 405B). MoE sparsity is load-bearing.

**Will 1.58-bit quality be terrible?** 1.58-bit is the stunt tier and shows real degradation. 3-4 bit dynamic quants hold up well in public benchmarks (3-bit DeepSeek V3.1 scored 75.6% vs 76.1% unquantized on Aider Polyglot). The runtime is quant-agnostic; it pages whatever GGUF you give it.

**Can I run this today?** Yes. The simulator, SSD probe, trace collector, repacker, planner, and the paging runtime itself (as a patch series against a pinned llama.cpp commit — `runtime/apply.sh`) all work today; every number above has a reproduce command. What remains on the roadmap is the R1-671B stunt run.

**Windows/macOS?** The simulator runs anywhere Python does. The NVMe probe and the planned runtime are Linux-first (io_uring, O_DIRECT). macOS unified memory largely doesn't need this; Windows support would need a different I/O backend.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — adding a model preset, tracing a new model, or extending cache policies are all small, well-bounded first contributions. If you know llama.cpp internals around `build_moe_ffn`, or have io_uring/CUDA experience, Phase 2 needs you.

## License

MIT
