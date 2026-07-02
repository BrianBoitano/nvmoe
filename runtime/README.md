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
