#!/usr/bin/env python3
"""Compare two llama-nvmoe-logits dumps -- the Phase 2.3 correctness gate.

The fork's `llama-nvmoe-logits` tool greedy-decodes a fixed prompt and dumps
the full logits vector at every step. Run it on the original GGUF and on the
pack's resident.gguf, then:

    python3 tools/compare_logits.py stock.bin pack.bin

Exit 0 iff both dumps decode the same tokens AND the logits agree within
--tol (default 0.0: bit-identical, the CPU-backend requirement; GPU backends
may need a small tolerance, which the tool prints so the report is honest).

Dump format (little-endian):
    char[8]  magic "NVMLOG01"
    u32      n_vocab
    u32      n_prompt, u32 prompt_tokens[n_prompt]
    u32      n_steps
    per step: u32 chosen_token, f32 logits[n_vocab]
"""

import argparse
import struct
import sys


def read_dump(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"NVMLOG01":
            sys.exit(f"{path}: bad magic {magic!r}")
        (n_vocab,) = struct.unpack("<I", f.read(4))
        (n_prompt,) = struct.unpack("<I", f.read(4))
        prompt = struct.unpack(f"<{n_prompt}I", f.read(4 * n_prompt))
        (n_steps,) = struct.unpack("<I", f.read(4))
        steps = []
        for _ in range(n_steps):
            (tok,) = struct.unpack("<I", f.read(4))
            logits = f.read(4 * n_vocab)
            if len(logits) != 4 * n_vocab:
                sys.exit(f"{path}: truncated dump")
            steps.append((tok, logits))
    return n_vocab, prompt, steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", help="reference dump (stock GGUF run)")
    ap.add_argument("test", help="dump under test (pack run)")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max allowed |logit diff| (default 0 = bit-identical)")
    args = ap.parse_args()

    nv_a, prompt_a, steps_a = read_dump(args.ref)
    nv_b, prompt_b, steps_b = read_dump(args.test)

    if nv_a != nv_b:
        sys.exit(f"FAIL: n_vocab differs ({nv_a} vs {nv_b})")
    if prompt_a != prompt_b:
        sys.exit("FAIL: prompt tokens differ (different tokenizer state?)")
    if len(steps_a) != len(steps_b):
        sys.exit(f"FAIL: step count differs ({len(steps_a)} vs {len(steps_b)})")

    max_diff = 0.0
    max_at = (-1, -1)
    identical_bytes = True
    for i, ((tok_a, la), (tok_b, lb)) in enumerate(zip(steps_a, steps_b)):
        if tok_a != tok_b:
            sys.exit(f"FAIL: step {i} chose different tokens ({tok_a} vs {tok_b})")
        if la != lb:
            identical_bytes = False
            fa = struct.unpack(f"<{nv_a}f", la)
            fb = struct.unpack(f"<{nv_a}f", lb)
            for j, (x, y) in enumerate(zip(fa, fb)):
                d = abs(x - y)
                if d > max_diff:
                    max_diff = d
                    max_at = (i, j)

    n = len(steps_a)
    if identical_bytes:
        print(f"PASS: {n} steps, {nv_a} logits/step -- BIT-IDENTICAL")
        return
    print(f"logits differ: max |diff| = {max_diff:.3e} at step {max_at[0]}, vocab id {max_at[1]}")
    if max_diff <= args.tol:
        print(f"PASS: {n} steps, same tokens, within --tol {args.tol:g}")
    else:
        sys.exit(f"FAIL: exceeds --tol {args.tol:g}")


if __name__ == "__main__":
    main()
