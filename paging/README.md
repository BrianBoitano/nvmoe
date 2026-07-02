# paging/ — the NVMe→VRAM paging library (Phase 2.2)

One benchmark tool, built in stages, each measured before any llama.cpp
surgery (docs/DESIGN.md, component 2):

- **2.2a — the I/O half, done.** Random expert-extent fetches from a real
  `experts.pack` via io_uring + O_DIRECT + registered buffers, timed.
- **2.2b — the GPU half, done.** `--gpu` extends every fetch to the full
  runtime path: NVMe → pinned staging (`cuMemHostAlloc`) →
  `cuMemcpyHtoDAsync` → VRAM slab, pipelined per slot with CUDA events,
  with a byte-verified NVMe→pinned→VRAM→host round trip at startup.
- What remains for 2.3 is policy, not plumbing: the LRU slot cache and the
  router-guided prefetcher, which live with the llama.cpp integration.

## Build and run

```bash
make -C paging                                        # needs gcc + kernel 5.4+
python3 tools/pack_extents.py models/olmoe-q4_0.nvmoe # manifest -> extents.tsv
./paging/nvmoe-iobench models/olmoe-q4_0.nvmoe        # I/O half, sweeps QD 1..32
./paging/nvmoe-iobench models/olmoe-q4_0.nvmoe --gpu  # full NVMe->VRAM path
```

No liburing needed — the ring is set up with raw syscalls (~80 lines).
No CUDA toolkit needed either: `--gpu` dlopens the *driver* API
(`libcuda.so.1`, ships with the NVIDIA driver) against ~20 self-declared
prototypes, so one plain-gcc build runs CPU-only anywhere and does the VRAM
path wherever `nvidia-smi` works. **Docker note:** the default seccomp
profile blocks `io_uring_setup` (EPERM). Run on the host or with a seccomp
profile that allows io_uring; the tool prints a hint when it happens.

## Measured results (reference box)

Samsung 990 PRO 2TB (PCIe 4.0 x4), btrfs (no compression), kernel 6.18,
RTX 5070 Ti on PCIe 4.0 x16, uniform-random extents, 3,000 fetches per
point. Reproduce with the commands above. At queue depth >1, latency
includes queue wait by design — it is the time from "runtime wants this
expert" to "bytes are usable".

**Qwen3-30B-A3B pack** (6,144 extents of 2.5–2.9MB). `uring` = into host
RAM; `uring+h2d` = all the way into VRAM:

| mode | qd | GB/s | fetch/s | p50 ms | p99 ms |
|---|---|---|---|---|---|
| uring | 1 | 4.14 | 1446 | 0.680 | 0.939 |
| uring+h2d | 1 | 3.99 | 1394 | 0.707 | 1.072 |
| uring | 2 | **6.91** | 2414 | 0.821 | 1.142 |
| uring+h2d | 2 | **6.59** | 2302 | 0.860 | 1.177 |
| uring | 4 | 5.74 | 2007 | 1.997 | 2.872 |
| uring+h2d | 4 | 5.76 | 2011 | 1.995 | 2.810 |
| uring+h2d | 32 | 5.73 | 2004 | 15.977 | 16.943 |

**OLMoE-1B-7B pack** (1,024 extents of 3.4MB), the same shape: I/O-only
peaks 6.96 GB/s @ QD2 (p50 0.99ms), end-to-end 6.80 GB/s @ QD2 (p50 1.00ms);
QD1 miss-to-VRAM p50 0.76ms, p99 1.08ms. pread QD1 baselines land within a
few percent of uring QD1 in both modes.

## What this settles

**One expert miss, disk to VRAM, costs ~0.7ms** (p50 at QD1; p99 ~1.1ms).
That is the stall the Phase 2.3 MVP's synchronous fetch-on-miss adds per
missed expert — small enough that correctness-first integration is viable
before prefetch exists. At Qwen3-30B's measured 99.6% hit rate on a 16GB
card (~1.5 misses/token), miss stalls add ~1ms/token even with zero overlap.

**The GPU hop is nearly free.** End-to-end throughput is 95-98% of the
I/O-only number at every queue depth: PCIe H2D (~26GB/s effective) hides
behind the ~7GB/s NVMe reads in the per-slot pipeline. The NVMe stays the
bottleneck, which is exactly the design's premise.

**Peak paging bandwidth is ~7GB/s at QD2** (~6.6 GB/s to VRAM, ~2,300
experts/s) — the number the Phase 0 simulator assumed for its tok/s
ceilings, confirmed on the real pack file with the real access pattern.

**Deep queues buy nothing at expert-sized reads.** Throughput peaks at QD2
and *falls* to ~5.8GB/s beyond QD4 while latency grows linearly (pure
queueing). Multi-MB requests are already parallel inside the device, so the
prefetcher should issue shallow (2–4 in flight) and spend its cleverness on
*which* extents to fetch, not how many to keep in flight.

**The staging ring stays tiny.** 64 slots × 3.4MB ≈ 220MB of pinned host
RAM sustained the peak numbers — the ≤4GB budget has an order of magnitude
of headroom, and io_uring accepts driver-pinned (`cuMemHostAlloc`) pages as
registered buffers, so one allocation serves both DMA directions.

**io_uring ≈ pread at QD1** — the win is not syscall overhead at these
sizes. It is that one thread can have N reads in flight *while doing other
work*: exactly what the runtime needs (submit prefetches, go back to
compute), and what blocking pread cannot do without a thread pool.
