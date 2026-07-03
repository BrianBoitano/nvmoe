# runtime/ — the llama.cpp exps=NVMe patch series (Phase 2.3)

The actual runtime: a patch series against a pinned llama.cpp commit that
teaches it to load an nvmoe pack and page routed experts from NVMe. See
[docs/INTEGRATION.md](../docs/INTEGRATION.md) for how the hook points work.

```bash
./runtime/apply.sh              # clone + apply -> ./llama.cpp-nvmoe (branch "nvmoe")
cd llama.cpp-nvmoe
cmake -B build && cmake --build build -j --target llama-nvmoe-logits llama-bench

# CUDA build (needs the toolkit; a nvidia/cuda:12.8+-devel container works):
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda -j --target llama-nvmoe-logits llama-bench
```

With a GPU build and `-ngl`, the expert cache pools live in **VRAM** and
misses stream NVMe → pinned bounce → VRAM; partial offload splits the pools
per device. Same code path either way — the runtime talks only to the
ggml-backend API.

No new flags: point any tool at a pack's `resident.gguf` and the pack is
detected from its `nvmoe.pack.version` KV.

```bash
./build/bin/llama-bench -m ../models/olmoe-q4_0.nvmoe/resident.gguf -p 16 -n 32
NVMOE_CACHE_MB=512 ./build/bin/llama-bench -m ...   # cap the expert cache
```

## Chat with it: llama-server

The pack works as a local OpenAI-compatible chat server — this is the
"actually use the model" path (build the `llama-server` target; the cmake
configure needs `-DLLAMA_BUILD_SERVER=ON` if it wasn't on):

```bash
NVMOE_CACHE_MB=11776 ./build-cuda/bin/llama-server \
    -m models/qwen3-next-80b-a3b-instruct-q4_k_m.nvmoe/resident.gguf \
    -ngl 99 -c 16384 --host 0.0.0.0 --port 8901
curl http://localhost:8901/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hello"}]}'
```

Point Open WebUI (or anything OpenAI-compatible) at `http://<host>:8901/v1`.
Measured on the reference box: the 80B pack streams ~150 tokens in ~5s
through the API, prefill included. Use `NVMOE_CACHE_MB` to leave VRAM for
the KV cache at your `-c`; the server's `--fit` estimate does not yet count
the expert cache (see docs/INTEGRATION.md), so size those two explicitly.
One `llama_context` per pack model still applies — the default single
server process is exactly that.

## Lookahead prefetch (the fourth patch)

At small cache budgets decode is fetch-bound, so the runtime predicts the
*next* layer's experts before its router runs and fetches them during the
current layer's matmuls. The predictor is one extra `mul_mat` per layer:
layer L+1's router with the RMS-norm weight ratio `w_{L+1}/w_L` folded in
at load time, applied to layer L's normed FFN input — exact up to L's
missing FFN residual. Measured on both reference models, that contains
**~86% of the ids actually routed** one layer ahead (top-8), and 83-86% of
the extents it prefetches get used. A persistent fetch pool (`NVMOE_QD`
workers) runs mandatory misses ahead of speculation, keeps one worker free
of speculative jobs, and promotes queued speculation an op starts waiting
on.

Two variants measured slower and are off by default, kept as env knobs:
top-16 prediction (`NVMOE_LOOKAHEAD=16`) and a two-layer horizon
(`NVMOE_LOOKAHEAD_DEPTH=2`) — on an I/O-bound pipeline the wasted extents
cost more than the extra hidden misses. For the same reason speculation is
gated by **extent size**: at GPT-OSS-120B's 12.6MB extents an 82%-accurate
predictor still *lost* 11% throughput (decode there is ~97% SSD-time — no
idle bandwidth to hide waste in), so layers with extents over 6MB don't
speculate by default. Lookahead also auto-disables when the cache holds
over 60% of the paged experts (its per-layer kernel launches, with CUDA
graphs already off, outweigh the rare hidden miss). `NVMOE_LOOKAHEAD=8`
forces lookahead on, `NVMOE_LOOKAHEAD=0` off; `NVMOE_PREFETCH=1` forces
speculation at any extent size (try it on a faster SSD), `NVMOE_PREFETCH=0`
keeps the prediction statistics without speculative I/O. The stats print
at model teardown (visible via `llama-nvmoe-logits`; `llama-bench`
silences model logs).

Rationale with data: the same-ids and offline-correlation predictors were
evaluated against the Phase 1 traces first and rejected —
`python3 tools/analyze_lookahead.py traces/qwen3-all.tokens.jsonl`.

## The correctness gate (run it yourself)

`llama-nvmoe-logits` greedy-decodes a fixed prompt and dumps the **full
logits vector at every step**; `tools/compare_logits.py` demands they match.
On the CPU backend the pack path must be **bit-identical** to stock — same
bytes through the same kernels leave no room for "close enough".

```bash
B=llama.cpp-nvmoe/build/bin
$B/llama-nvmoe-logits -m models/olmoe-q4_0.gguf              -o /tmp/stock.bin -n 32
$B/llama-nvmoe-logits -m models/olmoe-q4_0.nvmoe/resident.gguf -o /tmp/pack.bin  -n 32
python3 tools/compare_logits.py /tmp/stock.bin /tmp/pack.bin
# PASS: 32 steps, 50304 logits/step -- BIT-IDENTICAL
```

Verified 2026-07-02 (this exact procedure; GPU rows on an RTX 5070 Ti,
CUDA 12.8, `-ngl` as shown; "prefetch" = lookahead prefetch active):

| backend | model | steps | cache | result |
|---|---|---|---|---|
| CPU | OLMoE-1B-7B Q4_0 | 32 | all resident | BIT-IDENTICAL |
| CPU | OLMoE-1B-7B Q4_0 | 32 | 512MB (heavy eviction) | BIT-IDENTICAL |
| CPU | OLMoE-1B-7B Q4_0 | 32 | 512MB + prefetch | BIT-IDENTICAL |
| CPU | OLMoE-1B-7B Q4_0 | 48, second prompt | all resident | BIT-IDENTICAL |
| CPU | Qwen3-30B-A3B Q4_K_M | 24 | all resident (2 pool groups, mixed Q4_K/Q6_K) | BIT-IDENTICAL |
| CPU | Qwen3-30B-A3B Q4_K_M | 16 | 4GB + prefetch (eviction, 2 pool groups) | BIT-IDENTICAL |
| CPU | GPT-OSS-120B MXFP4 | 12 | 8GB (63GB model, eviction) | BIT-IDENTICAL |
| CPU | DeepSeek-V2-Lite Q4_K_M | 24 | all resident AND 2GB + prefetch (shared experts, MLA, 3 quant types) | BIT-IDENTICAL |
| CUDA `-ngl 99` | OLMoE-1B-7B Q4_0 | 32 | all resident in VRAM | BIT-IDENTICAL |
| CUDA `-ngl 99` | OLMoE-1B-7B Q4_0 | 32 | 512MB VRAM (heavy eviction) | BIT-IDENTICAL |
| CUDA `-ngl 99` | OLMoE-1B-7B Q4_0 | 32 | 512MB VRAM + prefetch | BIT-IDENTICAL |
| CUDA `-ngl 8` | OLMoE-1B-7B Q4_0 | 32 | pools split CPU + VRAM (+ prefetch) | BIT-IDENTICAL |
| CPU | Qwen3-Next-80B Q4_K_M | 16 | 8GB + prefetch (hybrid attention, shared expert) | BIT-IDENTICAL |
| CUDA `-ngl 14` | Qwen3-Next-80B Q4_K_M | 16 | 8GB, pools split CPU + VRAM | BIT-IDENTICAL |
| CPU | GLM-4.5-Air Q4_K_M | 12 | 8GB (sigmoid+bias gating, shared expert, NextN skip) | BIT-IDENTICAL |

Prefetch cannot change the math — it only warms cache slots; the ids the
matmuls consume always come from the real router — but the eviction /
in-flight / remap interplay is exactly where a bug would hide, so the gate
runs with it on.

The gate tool normalizes two kernel choices so both sides run the same
math (kernel-choice differences, not data differences — see
docs/INTEGRATION.md): CPU weight repacking is off (`use_extra_bufts =
false` — the interleaved kernels sum in a different order than the plain
kernels the pool tensors use), and host-op offload is off (`op_offload =
false` — at partial offload the scheduler may ship CPU-resident ops to
the GPU, and the pack graph's custom-op splits change which ops qualify,
so stock and pack would otherwise run different kernels on different
devices).

## Measured decode speed (Phase 2.4)

Reference box: RTX 5070 Ti 16GB, Samsung 990 PRO (PCIe 4.0 x4), CUDA 12.8.
Model: **Qwen3-30B-A3B Q4_K_M — an 18.6GB GGUF that does not fit in 16GB
VRAM.** All rows are `llama-bench` in the CUDA container; the nvmoe rows
point at the pack's `resident.gguf`, the stock rows at the original GGUF.

| config | host RAM | pp512 | tg128 (decode) |
|---|---|---|---|
| **nvmoe pack, 12GB VRAM expert cache** | **<0.8GB** | 1042 ± 103 | **166.1 ± 2.2** |
| stock, max partial offload (`-ngl 38`) | ~5GB | 1519 ± 29 | 79.2 ± 0.5 |
| stock, experts in RAM (`-ot 'blk\..*\.ffn_.*_exps.*=CPU' -ngl 99`) | ~17GB | 603 ± 11 | 45.2 ± 0.5 |
| stock, full VRAM | — | *does not fit* | *does not fit* |

```bash
# nvmoe row (166 tok/s):
NVMOE_CACHE_MB=12288 ./build-cuda/bin/llama-bench \
    -m <pack>/resident.gguf -ngl 99 -p 512 -n 128 -r 5 -t 8
# stock baselines:
./build-cuda/bin/llama-bench -m <model>.gguf -ngl 38 -p 512 -n 128 -r 3 -t 8
./build-cuda/bin/llama-bench -m <model>.gguf -ngl 99 -p 512 -n 128 -r 3 -t 8 \
    -ot 'blk\..*\.ffn_.*_exps.*=CPU'
```

**Decode from the pack is 2.1x the best stock configuration and 3.7x the
exps-in-RAM recipe, using ~20x less host memory.** This is the sim's
"model one size over your VRAM" prediction made real: at a 12GB cache
(~73% of experts) the measured hit rate over a 133-token run was 94.2%
*including* the cold start, and warm decode holds 166 tok/s with sync
fetch-on-miss — no prefetch involved. Prefill is slower than the best
stock split (1042 vs 1519) because a long prompt sweeps most experts
through the cache; nvmoe is decode-optimized by design.

The stock partial-offload row keeps ~5GB of layers in RAM; the exps=CPU
row keeps all 17GB of experts in RAM. If that RAM isn't free — the whole
premise of nvmoe — those configs don't exist.

**Cache budget → throughput** (same pack, decode-only `-p 0 -n 128 -r 5`,
includes cold-start misses; the floor is `n_expert` slots per pool group):

| NVMOE_CACHE_MB | share of experts | tg128, sync fetch | tg128, + lookahead prefetch |
|---|---|---|---|
| 4096 | ~24% | 21.9 ± 1.5 | **23.4 ± 1.4** |
| 6144 | ~36% | 28.0 ± 2.5 *(older run, patch 3)* | — |
| 8192 | ~48% | 60.2 ± 10.3 | **62.4 ± 10.9** |
| 12288 | ~73% | **169.7 ± 1.4 warm** (`-p 512`; the 166.1 headline config re-measured on the patch-4 binary) | 161.7 ± 3.2 forced on |

Hit rate becomes throughput, exactly as the Phase 1 trace curves
predicted. Even the 8GB row beats the 17GB-of-RAM exps=CPU recipe.
Progression on the 4GB row across the patch series: sequential misses
17.0 → QD-4 batched (patch 3) 21.9 → lookahead prefetch (patch 4) 23.4.
Lookahead is worth ~4-7% where fetches dominate and costs ~4% where they
don't (the 12288 row) — hence the auto-off. The sync columns are the
same binary with `NVMOE_LOOKAHEAD=0`, measured back-to-back; the large
± at 8192 is the cold first rep, common to both columns.

## GPT-OSS-120B — a 63GB model on the 16GB card (Phase 2.4b)

Same box, same procedure, a model **4x the VRAM**: GPT-OSS-120B MXFP4
(63.4GB GGUF; 36 layers × 128 experts, top-4; 5.1B active params). The
three HF splits merge with `llama-gguf-split --merge`; the repacker then
produces 4,608 extents of 12.6MB (96.1% of the file paged, 2.3GB resident)
in 68s, `verify_pack.py` proves every byte, and the logits gate passes:
**bit-identical vs stock over 12 greedy steps** (201k logits/step, CPU,
8GB cache — the all-resident pools wouldn't fit in RAM, which is the
point).

| config | tg64 (decode) |
|---|---|
| **nvmoe pack, 11GB VRAM cache** | **24.5 ± 3.3** |
| nvmoe pack, 8GB | 18.7 ± 1.0 |
| nvmoe pack, 4GB | 10.5 ± 0.5 |
| stock, experts in RAM (`-ot ...exps=CPU -ngl 99`) | 8.3 ± 4.0 — and it monopolizes ~57GB of page cache |
| stock, best partial offload that fits (`-ngl 7`) | 6.6 ± 3.5 |
| stock, full VRAM | *does not fit, at all* |

```bash
./build-cuda/bin/llama-gguf-split --merge gpt-oss-120b-mxfp4-00001-of-00003.gguf gpt-oss-120b-mxfp4.gguf
python3 tools/repack_gguf.py models/gpt-oss-120b-mxfp4.gguf
python3 tools/verify_pack.py models/gpt-oss-120b-mxfp4.nvmoe models/gpt-oss-120b-mxfp4.gguf
NVMOE_CACHE_MB=11264 ./build-cuda/bin/llama-bench \
    -m models/gpt-oss-120b-mxfp4.nvmoe/resident.gguf -ngl 99 -p 0 -n 64 -r 3 -t 8
```

Prefill at the 11GB cache: pp512 = 98.6 ± 0.03 (a long prompt sweeps most
of the 57GB of experts through the cache — decode-optimized by design).
The simulator's ceiling for this model was ~11 tok/s; reality is 2.2x
that, because real routing is far more cacheable than the synthetic
traces assumed (the same direction Phase 1 measured on Qwen3). And the
RAM story holds at 120B scale: the whole decode ran inside a
`--memory=4g` cgroup at full speed (24.5 → 23.3 within noise), peak
3.1GB — most of which is the one-time 2.3GB resident-weight load
streaming through page cache.

## Qwen3-Next-80B-A3B — the usable-flagship tier (Phase 3's pick)

The planner's recommendation made real, same day: Qwen3-Next-80B-A3B
Instruct Q4_K_M (48.5GB GGUF; 48 MoE layers × 512 experts top-10 + a
shared expert; ~3B active) — the ultra-sparse geometry this design was
aimed at from Phase 0. Repack 54s → 24,576 extents of 1.7-1.9MB (96.5%
paged, zero alignment padding), byte-verified. First **hybrid-attention**
architecture (gated delta net + full attention) through the pack:
**bit-identical logits** on CPU (heavy eviction + prefetch) *and* on the
CUDA backend at `-ngl 14` with the pools split across CPU and GPU.

| config | host RAM | pp512 | decode (tg128) |
|---|---|---|---|
| **nvmoe pack, 11.5GB VRAM cache (planner's pick), warm** | **~1GB** | 134.6 ± 2.9 | **44.8 ± 2.6** |
| nvmoe pack, same, inside a `--memory=4g` cgroup | 1.01GB peak, measured | — | 38.9 ± 3.6 *(cold-inclusive)* |
| stock, best partial offload (`-ngl 14`) | ~33GB (page cache) | 272.6 ± 73.5 | 22.6 ± 0.4 |
| stock, experts in RAM (`-ot exps=CPU -ngl 99`) | ~44GB | — | 20.9 ± 5.6 |
| stock, full VRAM | — | *does not fit* | *does not fit* |

**Cache sweep** (decode-only `-p 0 -n 128 -r 3`, includes cold start) —
the first model where the 1.9MB extents sit far under the 6MB speculation
gate, and lookahead prefetch pays at *every* fetch-bound budget:

| NVMOE_CACHE_MB | share of experts | sync fetch | + lookahead prefetch |
|---|---|---|---|
| 4096 | ~9% | 18.0 ± 0.7 | **20.2 ± 0.6** (+12%) |
| 8192 | ~18% | 27.3 ± 1.0 | **31.9 ± 0.9** (+17%) |
| 11776 | ~26% | 35.4 ± 4.0 | **39.9 ± 3.6** (+13%) |

The lookahead predictor's fifth architecture lands in the same band as
the other four: 84.4-84.5% of routed ids predicted one layer ahead
(top-10), with 81% of speculative fetches used.

```bash
python3 tools/plan.py models/qwen3-next-80b-a3b-instruct-q4_k_m.gguf   # prints all of this
python3 tools/repack_gguf.py models/qwen3-next-80b-a3b-instruct-q4_k_m.gguf
python3 tools/verify_pack.py models/qwen3-next-80b-a3b-instruct-q4_k_m.nvmoe \
    models/qwen3-next-80b-a3b-instruct-q4_k_m.gguf
NVMOE_CACHE_MB=11776 ./build-cuda/bin/llama-bench \
    -m models/qwen3-next-80b-a3b-instruct-q4_k_m.nvmoe/resident.gguf \
    -ngl 99 -p 512 -n 128 -r 5 -t 8
```

## GLM-4.5-Air — the honest negative: active params are the wall

The fourth architecture (46 MoE layers × 128 experts top-8, sigmoid gating
with selection bias, shared expert, a NextN/MTP layer whose experts are
packed but never referenced) passes the gate — **bit-identical on CPU,
12 steps** — and then proves the planner right about why it should not be
run this way: A12B means ~4.2GB of expert reads per token (2.2x
GPT-OSS-120B's), and 5GB of resident weights squeeze the cache to ~13% of
experts on a 16GB card.

| config | tg64 |
|---|---|
| nvmoe pack, 7.9GB cache (max) | 2.80 ± 0.08 |
| nvmoe pack, 4GB | 2.29 ± 0.04 |
| stock, best partial offload (`-ngl 8`) | 2.14 ± 0.50 |

The planner predicted 2.4 tok/s at the max cache from the freshly collected
trace before the bench measured 2.80 (1.15x) — its fourth routing family
(`traces/glm-all.tokens.jsonl`: top-10% experts carry 26.7% of traffic,
between Qwen3's 34.7% and V2-Lite's 17.7%). The design lesson, now measured
at both ends: **usable-from-NVMe on a 16GB card means A3B-class active
parameters.** Qwen3-Next-80B (A3B) runs at 44.8; GLM-4.5-Air (A12B) runs
at 2.8. Total size barely matters; active size is destiny.

## DeepSeek-V2-Lite — third architecture, and what paging costs when you don't need it

DeepSeek-V2-Lite Q4_K_M (10.4GB, 26 MoE layers × 64 experts top-6, 2
shared experts, MLA attention, extents in two geometries and three quant
types) validates the DeepSeek graph path end-to-end: repack 9.2s,
byte-verified, **bit-identical logits** all-resident and under 2GB
eviction with prefetch. The lookahead predictor's accuracy on its third
architecture: **86.4% at top-6** (OLMoE 85.8%, Qwen3 85.9%, GPT-OSS
84.9% — the residual-stream approximation is remarkably arch-stable).

This model *fits* in 16GB, which makes it the honest measure of pack
overhead and of fine-grained routing under pressure (tg128, `-p 0 -r 3`
except where noted):

| config | tg128 |
|---|---|
| stock `-ngl 99` (warm) | 269.0 ± 1.2 |
| pack, all resident (warm) | 234.7 ± 5.3 — **paging costs ~13% when you didn't need it; use stock when the model fits** |
| pack, 4GB cache | 32.3 ± 2.5 |
| pack, 2GB cache | 14.6 ± 0.7 |

The small-cache rows are steep because DeepSeek-style routing is flat
(see the trace analysis in the main README): at a 2GB cache the hit rate
is 61.5% where Qwen3's curve would predict ~85%. Prefetch measured a
wash here (32.6 vs 32.3 at 4GB; 14.3 vs 14.6 at 2GB) with its 5-6MB
extents sitting right at the 6MB speculation gate — a live confirmation
that the gate's default is approximately the break-even extent size on
the reference SSD.

## The host-RAM proof (Phase 2 headline)

The entire 30B decode above runs inside a **hard 4GB memory cgroup** —
and doesn't come close to the wall:

```bash
docker run --rm --gpus all --memory=4g --memory-swap=4g ... \
    llama-bench -m <pack>/resident.gguf -ngl 99 -n 128 ...
# tg128 94.6 ± 56.5 (same as unconstrained cold-start)
# cgroup memory.peak: 793,010,176 bytes = 756MB
```

Peak host memory: **756MB** for the whole process — resident weights
stream through to VRAM, experts go NVMe → pinned bounce (a few MB) →
VRAM, and O_DIRECT keeps the page cache out of it. The design budget was
≤4GB; measured reality is 5x under it. Docker can keep the rest of your
RAM.
