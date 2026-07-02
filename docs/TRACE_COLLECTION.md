# Collecting real expert-routing traces from llama.cpp

Phase 1 replaces the synthetic traces in `sim/trace_gen.py` with measured
routing decisions. The cache simulator consumes them via `--trace` (JSONL,
one line per token, each line a JSON array of `[layer, expert_id]` pairs).

## Approach: eval callback, no fork needed

llama.cpp exposes `ggml_backend_sched_eval_callback` (wired through
`common_params.cb_eval`) — the same hook the `llama-imatrix` tool uses to
observe intermediate tensors. During graph execution the MoE routing tensors
are observable by name:

- `ffn_moe_topk-<il>` — the selected expert ids per token for layer `<il>`
  (output of the top-k over router logits in `build_moe_ffn`)
- `ffn_moe_probs-<il>` — router probabilities (useful later for training the
  prefetch predictor, not needed for cache simulation)

Collector sketch (C++, modeled on `examples/imatrix/imatrix.cpp`):

1. Register a `cb_eval` that returns true (ask for data) for tensors whose
   name starts with `ffn_moe_topk-`.
2. In the callback, copy the tensor to host (`ggml_backend_tensor_get`),
   decode the int32 expert ids — shape is `[top_k, n_tokens]`.
3. Buffer per-token selections across layers; flush one JSONL line per token
   position: `[[layer, e0], [layer, e1], ...]` in layer order.
4. Run a normal generation (`llama-cli` flow) over representative workloads.

## Calibration models for the PlexyLady box (no dedicated RAM)

Trace collection only needs the model to RUN, not run fast. mmap-loaded
weights live in the page cache, which the kernel reclaims under pressure, so
a slow trace run does not permanently take RAM from Docker.

| Model | GGUF size (Q4) | Why |
|---|---|---|
| Qwen3-30B-A3B | ~18GB | same fine-grained-expert family as Qwen3-Next; fastest to trace |
| GPT-OSS-20B | ~13GB | same architecture family as GPT-OSS-120B |
| Mixtral-8x7B | ~26GB | coarse-expert contrast case |
| DeepSeek-V2-Lite (16B) | ~10GB | closest small proxy for DeepSeek routing (shared expert + fine-grained) |

## Workloads to trace

Hit rates depend on the workload's routing entropy. Trace at least:

1. Multi-turn chat (the primary local use case)
2. Long-form generation from a short prompt (best-case temporal locality)
3. Code generation (reportedly different expert distribution)
4. Long-document summarization (prefill-heavy worst case)

## What to extract

- Per-layer expert popularity distribution → fit `zipf_s`
- Token-to-token working-set overlap at windows 8/32/128 → fit `locality`
- Cross-layer co-activation matrix → seeds the Phase 2 prefetch predictor
- Replayed LRU / LRU+pin hit rates vs the synthetic predictions → publish the
  delta in the README (honesty budget: if synthetic was optimistic, say so)
