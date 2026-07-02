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
CUDA 12.8, `-ngl` as shown):

| backend | model | steps | cache | result |
|---|---|---|---|---|
| CPU | OLMoE-1B-7B Q4_0 | 32 | all resident | BIT-IDENTICAL |
| CPU | OLMoE-1B-7B Q4_0 | 32 | 512MB (35.6% hit, heavy eviction) | BIT-IDENTICAL |
| CPU | OLMoE-1B-7B Q4_0 | 48, second prompt | all resident | BIT-IDENTICAL |
| CPU | Qwen3-30B-A3B Q4_K_M | 24 | all resident (2 pool groups, mixed Q4_K/Q6_K) | BIT-IDENTICAL |
| CUDA `-ngl 99` | OLMoE-1B-7B Q4_0 | 32 | all resident in VRAM | BIT-IDENTICAL |
| CUDA `-ngl 99` | OLMoE-1B-7B Q4_0 | 32 | 512MB VRAM (34% hit, heavy eviction) | BIT-IDENTICAL |
| CUDA `-ngl 8` | OLMoE-1B-7B Q4_0 | 32 | pools split CPU + VRAM | BIT-IDENTICAL |

The gate tool disables CPU weight repacking (`use_extra_bufts = false`):
the repacked interleaved kernels sum in a different order than the plain
kernels the pool tensors use, which is a kernel-choice difference, not a
data difference — see docs/INTEGRATION.md.

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

**Cache budget → throughput** (same pack, decode-only `-p 0 -n 128 -r 3`,
includes cold-start misses; the floor is `n_expert` slots per pool group):

| NVMOE_CACHE_MB | share of experts | tg128 |
|---|---|---|
| 4096 | ~24% | 21.3 ± 0.2 |
| 6144 | ~36% | 28.0 ± 2.5 |
| 8192 | ~48% | 53.1 ± 9.2 |
| 12288 | ~73% | 97.4 ± 55.3 cold-start / **166.1 ± 2.2 warm** |

Hit rate becomes throughput, exactly as the Phase 1 trace curves
predicted. Even the 8GB row beats the 17GB-of-RAM exps=CPU recipe.
(The 4GB/8GB rows include the batched-miss overlap of the third patch —
misses in one op are fetched by up to `NVMOE_QD` concurrent readers,
default 4, the measured NVMe sweet spot; sequential fetching measured
17.0 and 47.6 on the same rows. The 6144 row predates it.)

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
