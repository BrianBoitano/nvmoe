# How nvmoe hooks into llama.cpp (Phase 2.3)

The patch series in `runtime/patches/` teaches llama.cpp to run a model from
an nvmoe pack ([PACK_FORMAT.md](PACK_FORMAT.md)) with the routed-expert
weights paged from NVMe instead of loaded — and, since the fourth patch, to
prefetch the next layer's experts before its router has run. This documents
where it hooks in and why — enough to re-do the surgery on a different
llama.cpp commit by hand if the patches ever stop applying.

## The trick in one paragraph

`ggml_mul_mat_id(as, b, ids)` computes, for every token, `as[ids[k]] @ b`
— it indexes the expert dimension of the weight tensor `as` with runtime
ids. It doesn't care whether `as` holds *all* the experts or just the ones
currently needed, as long as the ids point at the right slices. So: give it
a fixed-size **pool tensor** `[ne0, ne1, n_slots]` that caches hot experts,
and rewrite the ids from expert-space to slot-space just before the matmul.
The matmul kernel — CPU or CUDA — is untouched, which is what makes the
patch small and portable across llama.cpp versions.

## Hook points (5 files touched, 1 added)

**1. Pack detection — `llama_model_loader` ctor** (`src/llama-model-loader.cpp`).
The repacker stamps `resident.gguf` with `nvmoe.pack.version` /
`nvmoe.pack.manifest` KVs. The loader checks for them and records the pack
directory. No new CLI flags: `-m pack/resident.gguf` on any tool just works.

**2. Skipping the expert weights — `llama_model_base::create_tensor`**
(`src/llama-model.cpp`). In pack mode, requests for
`blk.*.ffn_{gate,up,down}_exps.weight` return `nullptr` before touching the
GGUF (they aren't in it). This is the one chokepoint every architecture's
tensor-loading code funnels through, so no per-arch edits are needed. The
loader's `n_created == n_tensors` accounting stays balanced because the
paged tensors aren't in the file's tensor count either.

**3. The runtime — `src/llama-nvmoe.{h,cpp}`** (new, ~400 lines). Owns:
- the parsed manifest (geometry per layer, extent offsets per expert),
- **pool tensors**, one per matrix kind per *(geometry, buffer type) group*
  (layers whose gate/up/down types+shapes match share pools and cache slots
  — Qwen3-30B's Q4_K_M has two geometry groups because half its layers
  quantize to Q6_K). Each pool lives in the buffer type of the device that
  runs its layer: full offload → VRAM, partial offload → split per device,
- the **LRU cache** keyed `(layer, expert) → slot`, with per-op pinning so
  an expert fetched for the current `mul_mat_id` can't be evicted by a
  later miss in the same op,
- the **fetch path**: one O_DIRECT `pread` of the expert's extent into an
  aligned bounce buffer, then `ggml_backend_tensor_set` per matrix kind
  into the pool slots — a memcpy on CPU pools, a synchronous H2D copy on
  device pools. When any pool is on a GPU the bounce buffer comes from the
  device's *host buffer type* (pinned), so the copies are straight DMA and
  O_DIRECT accepts the pages. No CUDA API appears anywhere in the runtime;
  the same code serves every backend ggml has.
  An expert occupies the same slot index in all three pools (gate/up share
  shape; down is transposed but same nbytes), so one id remap serves all
  three matmuls.

State lives on `llama_model` and defaults to *all experts resident*;
`NVMOE_CACHE_MB` caps it, floored at `n_expert` slots per group — the
worst case one `mul_mat_id` can reference — so any single op always fits.

**4. Graph surgery — `build_moe_ffn`** (`src/llama-graph.cpp`). After
`selected_experts` (the `ffn_moe_topk` tensor) is built, pack mode inserts:

```
ids_mm = ggml_map_custom1(cont(selected_experts), llama_nvmoe_fetch_op, lyr)
```

The custom op runs on the CPU during graph execution, *after* the router
has picked experts and *before* the expert matmuls (the graph dependency
enforces the order). It looks each id up in the cache, fetches misses
synchronously (0.7ms/extent measured, `paging/`), and writes slot indices.
The three `mul_mat_id` calls then use the pool tensors + `ids_mm`.
Everything indexed by *real* expert id — routing probabilities
(`get_rows`), per-expert biases (`add_id`) — keeps the original ids.
Per-expert weight *scales* and grovemoe's id arithmetic are asserted
unsupported (nothing nvmoe targets uses them).

**5. Plumbing** — `llm_graph_params`/`llm_graph_context` gain an `nvmoe`
pointer (`src/llama-graph.h`, `src/llama-context.cpp`), and
`llama_model_base::load_tensors` initializes the runtime after weights load
(`src/llama-model.cpp`).

## The correctness gate

`examples/nvmoe-logits` + `tools/compare_logits.py` — see
[runtime/README.md](../runtime/README.md) for the exact commands and the
verified table. The claim is the strongest one available: **bit-identical
logits** vs stock on the CPU backend, over dozens of greedy steps, including
under heavy cache eviction (every fetch/evict/remap path exercised).

Two things we learned the hard way, kept here so nobody re-learns them:

- **CPU weight repacking changes the math.** With `use_extra_bufts` on
  (the default), llama.cpp rewrites Q4_0/Q4_K weights into an interleaved
  layout at load and uses different matmul kernels whose summation order
  differs — stock-vs-pack logits then diverge at ~1e-6 per op (amplified
  over autoregressive steps) *even though the weight bytes are identical*.
  The gate tool sets `use_extra_bufts = false` on both sides so both use
  the plain kernels. (Pool tensors are plain-layout; teaching the fetch
  path to repack extents on the fly is possible but pointless — the real
  target is the GPU backend.)
- The ids tensor from `argsort_top_k` is a strided view; the custom op
  takes a `ggml_cont` of it.

Known quirk, not yet chased: upstream's `llama-eval-callback` example
segfaults on a pack model (our own dump mode in `llama-nvmoe-logits -d`
does the same job and works).

## The lookahead prefetcher (fourth patch)

The routing decision for layer L+1 is only known after layer L's FFN has
run — too late to hide a ~0.7ms extent read. But the residual stream
changes slowly per layer, so L+1's router applied to L's *input* is a good
approximation of where L+1 will route. Two facts make this nearly free:

- `build_moe_ffn` receives `cur = (x / rms(x)) ⊙ w_L` — this layer's
  RMS-normed FFN input. L+1's router wants `(x' / rms(x')) ⊙ w_{L+1}` where
  `x' = x + ffn_L(x) + attn_{L+1}(...)`. Approximate `x' ≈ x` and the norm
  difference becomes a per-channel constant: fold
  `W'[h,e] = gate_inp_{L+1}[h,e] · w_{L+1}[h] / w_L[h]` into a **lookahead
  router** at load time (`llama_nvmoe_init_lookahead`), and the graph pays
  exactly one extra `mul_mat` per layer.
- The custom op already runs on the CPU with the scheduler copying its
  inputs across — handing it the raw lookahead logits costs one more small
  host copy, and the top-k happens in the op (a 128-float `partial_sort`).
  An in-graph argsort was measured first and cost ~0.6ms/token at 48
  layers: with CUDA graphs disabled by the custom-op splits, every extra
  kernel launch is paid in full.

Measured accuracy (both reference models, printed at teardown): the top-8
prediction contains ~86% of the ids the real router then picks one layer
ahead, ~79.5% two ahead. Prediction is input-dependent, which is why it
beats any offline table — the misses at healthy budgets are the
long tail that aggregate statistics rank last (the analysis that
rejected the table approach: `tools/analyze_lookahead.py`).

Fetches move to a persistent pool of `NVMOE_QD` workers. The op resolves
its ids, enqueues real misses as *mandatory*, enqueues not-yet-cached
predictions for the next layer as *speculative*, and blocks only until its
own slots land. Three scheduling rules earned their keep in benchmarks:
mandatory pops before speculative; speculation may never occupy the last
free worker (a miss must not queue behind a full pipe of guesses); and a
queued speculative fetch the op starts waiting on is promoted. All cache
maps are mutated only on the graph thread — workers just read extents,
`ggml_backend_tensor_set` into pool slots, and clear an in-flight flag —
so bit-identity is preserved by construction (the matmul ids always come
from the real router; prefetch only warms slots).

What was tried and measured *slower* on the reference box, kept behind
envs: top-16 prediction (`NVMOE_LOOKAHEAD=16`, ~2x wasted bytes on an
I/O-bound pipeline) and a two-layer horizon (`NVMOE_LOOKAHEAD_DEPTH=2`,
compounding error + speculative evictions). Lookahead auto-disables when
the cache holds >60% of the paged experts — at near-resident budgets the
48 extra matmul launches cost ~4% and there is almost nothing to hide.
`NVMOE_PREFETCH=0` computes predictions and stats without speculative I/O.

## Constraints inherited by later stages

- One `llama_context` per pack-loaded model: the custom op mutates cache
  state without locking across contexts.
- Prefill sweeps experts by design (see DESIGN.md); the cache floor
  guarantees correctness, not speed, for big ubatches — the fetch op skips
  speculation for ubatches over 4 tokens.
- The custom op runs on the CPU backend even in GPU builds (ggml custom ops
  are CPU-only). That is *correct* by construction — the scheduler copies
  the tiny ids tensor to host and back — and it is where the fetch pool is
  driven from. It does cost graph splits (two per MoE layer), which
  disables CUDA graphs; that launch overhead is also why the lookahead
  top-k lives in the op, not the graph.
- `graph_max_nodes()` budgets by `n_tensors`, which the pack *shrinks*
  (the expert weights aren't file tensors) while the nvmoe path *adds*
  nodes per layer — the patch gives the budget back explicitly. Symptom
  when it bites: `GGML_ASSERT(obj_new)` in `ggml_new_object` during the
  first graph build on many-layer models.
