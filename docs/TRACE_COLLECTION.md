# Collecting real expert-routing traces from llama.cpp

Phase 1 replaces the synthetic traces in `sim/trace_gen.py` with measured
routing decisions. The cache simulator consumes them via `--trace` (JSONL,
one line per token, each line a JSON array of `[layer, expert_id]` pairs).

## The collector (built, working)

`collector/nvmoe-trace.cpp` is a small llama.cpp example tool that registers
a `cb_eval` scheduler callback (`llama_context_params.cb_eval`) and dumps the
`ffn_moe_topk-<il>` tensors — the selected expert ids per token per MoE layer
(I32, shape `[top_k, n_tokens]`) — as raw JSONL. Install and build it inside
any llama.cpp checkout:

```bash
./collector/install.sh /path/to/llama.cpp
# then:
llama-nvmoe-trace -m model.gguf -f prompts/chat.txt -o traces/run.raw.jsonl -n 400
python3 sim/trace_post.py traces/run.raw.jsonl --stats
python3 sim/trace_post.py traces/run.raw.jsonl --decode-only -o traces/run.tokens.jsonl
```

Gotcha handled by `trace_post.py`: during prefill llama.cpp computes the
final layer only for output rows, so that layer's record has a smaller
`n_tokens` than the rest of the step — short records align to the trailing
token positions. Related tensor for the Phase 2 prefetch predictor:
`ffn_moe_probs-<il>` (router probabilities).

Two collector gotchas learned on GPT-OSS-120B (both fixed in the tool):
`ffn_moe_topk` is a strided *view* of the argsort tensor, so a flat read
garbles any multi-token (prefill) record — the collector reads row by row
through `nb[1]`; and the name match must be exact (`ffn_moe_topk-<digits>`),
because nvmoe pack graphs also carry `ffn_moe_topk_nvmoe-<il>` (cache-slot
ids, not expert ids) which a prefix match happily logs as a duplicate layer.

`tools/collect_qwen_traces.sh` runs the standard four-workload suite.
`sim/calibrate.py` fits the synthetic generator to a real trace and reports
the real-vs-synthetic LRU hit-rate curve.

## Tracing models too big for CPU (GPU + pack)

A 63GB model decodes unusably slowly on CPU, but its routing is observable
at full pack speed: point the collector at an **nvmoe pack** with `-ngl 99`.
Pack routing is bit-identical to stock (that's the runtime's correctness
gate), so the trace is too, and the whole four-workload suite runs in ~4
minutes on the reference box. That is how the GPT-OSS-120B trace
(`traces/gptoss-all.tokens.jsonl`) was collected:

```bash
BIN=<fork>/build-cuda/bin/llama-nvmoe-trace \
MODEL=models/gpt-oss-120b-mxfp4.nvmoe/resident.gguf \
PREFIX=gptoss NGL=99 NVMOE_CACHE_MB=11264 bash tools/collect_qwen_traces.sh
```

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
