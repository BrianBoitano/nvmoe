# Contributing to nvmoe

Thanks for looking. This project is early and the surface area is small on
purpose — most contributions are self-contained and don't require touching
anything else.

## Good first contributions

**Add a model preset.** Edit `sim/presets.py`. You need five numbers from the
model's config.json: MoE layer count, experts per layer, top_k, and the expert
FFN dimensions (expert params = 3 x hidden_size x moe_intermediate_size).
Estimate `always_on_gb` as attention + shared experts + embeddings at your
quant level. Run `python3 sim/run_sim.py --model your-model` and sanity-check
the totals against the GGUF file size.

**Trace a model on your hardware.** Follow the Quickstart's optional section
(README) or docs/TRACE_COLLECTION.md. Traces from different model families
(especially DeepSeek-style fine-grained 256-expert routing) and different
workloads are directly useful — open a PR with the `*.tokens.jsonl` file and
the stats output. Keep committed traces under ~5MB.

**Run the NVMe probe on your SSD.** `python3 tools/nvme_probe.py <big file>`.
PRs adding a results table (drive model, PCIe gen, the QD/block matrix) to
docs/ help everyone calibrate expectations for their own hardware.

**Extend the cache simulator.** `sim/cache_sim.py` implements LRU and
oracle-pinned LRU. Interesting additions: LFU/2Q/ARC, mixed-precision tiers
(hot experts at higher bit-width, HOBBIT-style), and lookahead prefetch using
the `ffn_moe_probs` tensors the collector can also observe.

**Phase 2 (the runtime) needs:** llama.cpp internals experience
(`build_moe_ffn`, backend scheduler), io_uring / O_DIRECT plumbing, and CUDA
pinned-memory + async-copy experience. Open an issue to coordinate before
writing large amounts of code.

## Ground rules

- Python is 3.10+ stdlib only — no dependencies in `sim/` or `tools/`.
- Measured numbers beat simulated numbers; simulated numbers beat vibes.
  Every performance claim in the README links to a command that reproduces it.
- State your hardware when reporting numbers (GPU, RAM channels/speed, SSD,
  PCIe generation).

## Provenance and the llama.cpp upstream policy

This project was bootstrapped with heavy AI assistance (Claude), openly. The
maintainer reviews and owns everything merged. Two consequences:

1. If you use AI tools for a contribution, say so in the PR — same standard.
2. The `collector/` code is a *private-fork example tool* for llama.cpp.
   llama.cpp upstream does not accept predominantly AI-generated PRs
   (see their CONTRIBUTING.md); private forks are explicitly exempt, which is
   what this is. Do not submit nvmoe code upstream to llama.cpp unless you
   personally authored it and can defend it there.

## License

MIT. By contributing you agree your contribution is MIT-licensed.
