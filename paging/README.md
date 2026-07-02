# paging/ — the NVMe→VRAM paging library (Phase 2.2)

Being built in stages, each measured standalone before any llama.cpp surgery:

- **2.2a `nvmoe_iobench.c` — the I/O half, done.** Random expert-extent
  fetches from a real `experts.pack` via io_uring + O_DIRECT + registered
  buffers, timed. Answers the two questions the runtime design hangs on:
  what does one cache miss cost, and how fast can a prefetcher stream misses?
- **2.2b — the GPU half, next.** Pinned staging ring (≤4GB, the only host
  RAM the runtime uses) + `cudaMemcpyAsync` into a VRAM slab pool with LRU.
  Extends this benchmark to end-to-end NVMe→VRAM.

## Build and run

```bash
make -C paging                                        # needs gcc + kernel 5.4+
python3 tools/pack_extents.py models/olmoe-q4_0.nvmoe # manifest -> extents.tsv
./paging/nvmoe-iobench models/olmoe-q4_0.nvmoe        # sweeps QD 1..32
```

No liburing needed — the ring is set up with raw syscalls (~80 lines; the
repo promise is "gcc and a kernel is all you need"). **Docker note:** the
default seccomp profile blocks `io_uring_setup` (EPERM). Run on the host or
with a profile that allows io_uring; the tool prints a hint when it happens.

## Measured results (reference box)

Samsung 990 PRO 2TB (PCIe 4.0 x4), btrfs (no compression), kernel 6.18,
uniform-random extents, 3,000 fetches per point. Reproduce with the commands
above. At queue depth >1, latency includes queue wait by design — it is the
time from "runtime wants this expert" to "bytes are in the staging buffer".

**OLMoE-1B-7B pack** (1,024 extents of 3.4MB):

| mode | qd | GB/s | fetch/s | p50 ms | p99 ms |
|---|---|---|---|---|---|
| pread | 1 | 4.84 | 1368 | 0.696 | 0.924 |
| uring | 1 | 4.92 | 1389 | 0.701 | 0.898 |
| uring | 2 | **6.96** | 1967 | 0.992 | 1.645 |
| uring | 4 | 5.74 | 1623 | 2.277 | 3.117 |
| uring | 32 | 5.77 | 1629 | 19.737 | 30.952 |

**Qwen3-30B-A3B pack** (6,144 extents of 2.5–2.9MB):

| mode | qd | GB/s | fetch/s | p50 ms | p99 ms |
|---|---|---|---|---|---|
| pread | 1 | 4.20 | 1468 | 0.664 | 0.979 |
| uring | 1 | 4.14 | 1446 | 0.680 | 0.939 |
| uring | 2 | **6.91** | 2414 | 0.821 | 1.142 |
| uring | 4 | 5.74 | 2007 | 1.997 | 2.872 |
| uring | 32 | 5.81 | 2031 | 15.982 | 17.034 |

## What this settles

**One expert miss costs ~0.7ms** (p50, QD1; p99 under 1ms). That is the
stall the Phase 2.3 MVP's synchronous fetch-on-miss adds per missed expert —
small enough that correctness-first integration is viable before prefetch
exists. At Qwen3-30B's measured 99.6% hit rate on a 16GB card (~1.5 misses
per token), miss stalls add ~1ms/token even with zero overlap.

**Peak paging bandwidth is ~7GB/s at QD2** — the number the Phase 0
simulator assumed for its tok/s ceilings, now confirmed on the real pack
file with the real access pattern. ~2,000–2,400 experts/s.

**Deep queues buy nothing at expert-sized reads.** Throughput peaks at QD2
and *falls* to ~5.8GB/s beyond QD4 while latency grows linearly (pure
queueing). Multi-MB requests are already parallel inside the device, so the
design's original "QD 16–32" guess was wrong — the prefetcher should issue
shallow (2–4 in flight) and spend its cleverness on *which* extents to
fetch, not how many to keep in flight.

**io_uring ≈ pread at QD1** — the win is not syscall overhead at these
sizes. It is that one thread can have N reads in flight *while doing other
work*: exactly what the runtime needs (submit prefetches, go back to
compute), and what blocking pread cannot do without a thread pool.
