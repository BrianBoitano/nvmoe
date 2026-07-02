#!/usr/bin/env python3
"""Measure sustained NVMe read bandwidth at MoE-expert-sized reads.

nvmoe's decode ceiling assumes the SSD delivers near-sequential bandwidth on
expert-sized random reads. This probe checks that assumption on the actual
drive: O_DIRECT (page cache bypassed — same I/O path the runtime will use),
random 4K-aligned offsets, expert-sized blocks, at several queue depths
(emulated with reader threads; os.preadv releases the GIL).

Usage:
    python3 tools/nvme_probe.py <big_file_on_target_drive> [--seconds 5]

Block sizes map to real experts: 2MB Qwen3-Next-80B, 5MB Qwen3-30B-A3B,
9MB DeepSeek-R1 @1.58bit, 13MB GPT-OSS-120B, 99MB Mixtral-8x7B.
"""

import argparse
import mmap
import os
import random
import threading
import time
from pathlib import Path

BLOCK_SIZES_MB = [2, 5, 9, 13, 99]
QUEUE_DEPTHS = [1, 4, 8]
ALIGN = 4096


def reader(path: Path, block: int, file_size: int, stop: threading.Event,
           counter: list[int], lock: threading.Lock, seed: int) -> None:
    rng = random.Random(seed)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, block)  # page-aligned buffer, required by O_DIRECT
    max_off = (file_size - block) // ALIGN
    local = 0
    try:
        while not stop.is_set():
            offset = rng.randrange(0, max_off) * ALIGN
            got = os.preadv(fd, [buf], offset)
            local += got
    finally:
        os.close(fd)
        with lock:
            counter[0] += local


def probe(path: Path, block_mb: int, depth: int, seconds: float) -> float:
    block = block_mb * 1024 * 1024
    # O_DIRECT wants length aligned too; round to 4K
    block = (block // ALIGN) * ALIGN
    file_size = path.stat().st_size
    if file_size < block * 2:
        raise SystemExit(f"test file too small for {block_mb}MB blocks")

    stop = threading.Event()
    counter, lock = [0], threading.Lock()
    threads = [
        threading.Thread(target=reader, args=(path, block, file_size, stop, counter, lock, i))
        for i in range(depth)
    ]
    start = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    return counter[0] / elapsed / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="large file on the drive to test")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    size_gb = args.file.stat().st_size / 1e9
    print(f"file: {args.file} ({size_gb:.1f}GB), O_DIRECT random reads, "
          f"{args.seconds:.0f}s per cell\n")
    header = f"{'block':>7} | " + " | ".join(f"QD{d:<2}" for d in QUEUE_DEPTHS)
    print(header + "   (GB/s)")
    print("-" * len(header))
    for block_mb in BLOCK_SIZES_MB:
        cells = [f"{probe(args.file, block_mb, d, args.seconds):4.2f}" for d in QUEUE_DEPTHS]
        print(f"{block_mb:>5}MB | " + " | ".join(cells))


if __name__ == "__main__":
    main()
