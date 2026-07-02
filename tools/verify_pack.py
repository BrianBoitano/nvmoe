#!/usr/bin/env python3
"""Prove a repack is lossless: byte-compare every extent against the source.

The repacker (tools/repack_gguf.py) claims that resident.gguf + experts.pack
together contain exactly the source GGUF's tensor bytes, rearranged. This tool
checks that claim the hard way, trusting only the manifest and the two files:

  1. the source file is the one the manifest was built from (size + sha256);
  2. every source tensor is accounted for exactly once (paged XOR resident);
  3. every expert extent in experts.pack is byte-identical to the slice of the
     source tensor it claims to be (full read of both files, chunked memcmp);
  4. every tensor in resident.gguf is byte-identical to its source tensor, and
     the metadata KV section is a verbatim copy (nvmoe provenance KVs may
     follow it).

Nothing here is derived from the repacker's planning code — offsets and sizes
come from the manifest, bytes come from the disks — so a systematic repacker
bug shows up as a mismatch instead of being replicated on both sides.

Usage:
    python3 tools/verify_pack.py models/olmoe-q4_0.nvmoe models/olmoe-q4_0.gguf
    python3 tools/verify_pack.py <pack_dir> <source.gguf> --sample 64   # spot-check
    python3 tools/verify_pack.py <pack_dir> <source.gguf> --quick      # skip sha256

Exit code 0 = PASS (every compared byte identical), 1 = any mismatch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_lite import read_gguf  # noqa: E402
from repack_gguf import PACK_MAGIC, PACK_VERSION, human, sha256_file  # noqa: E402


class Mismatch(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"FAIL: {msg}")


def compare_ranges(fa, off_a: int, fb, off_b: int, nbytes: int, what: str,
                   chunk: int = 8 << 20) -> int:
    """memcmp two file ranges; on mismatch report the first differing byte."""
    fa.seek(off_a)
    fb.seek(off_b)
    done = 0
    while done < nbytes:
        n = min(chunk, nbytes - done)
        a, b = fa.read(n), fb.read(n)
        if len(a) != n or len(b) != n:
            raise Mismatch(f"{what}: short read at +{done} "
                           f"(got {len(a)}/{len(b)} of {n})")
        if a != b:
            i = next(i for i in range(n) if a[i] != b[i])
            raise Mismatch(f"{what}: bytes differ at +{done + i} "
                           f"({a[i]:#04x} != {b[i]:#04x})")
        done += n
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify an nvmoe repack against its source GGUF")
    ap.add_argument("pack_dir", help="directory written by repack_gguf.py")
    ap.add_argument("source", help="the original .gguf")
    ap.add_argument("--sample", type=int, metavar="N",
                    help="compare only N randomly chosen expert extents (seeded; "
                    "default: all of them)")
    ap.add_argument("--quick", action="store_true", help="skip the source sha256 check")
    args = ap.parse_args()

    t0 = time.time()
    pack_dir, src_path = Path(args.pack_dir), Path(args.source)
    man = json.loads((pack_dir / "manifest.json").read_text())
    if man.get("format") != "nvmoe-pack" or man.get("format_version") != PACK_VERSION:
        raise Mismatch(f"manifest format {man.get('format')!r} "
                       f"v{man.get('format_version')} not nvmoe-pack v{PACK_VERSION}")

    # 1. right source file?
    if src_path.stat().st_size != man["source"]["size"]:
        raise Mismatch(f"source size {src_path.stat().st_size} != "
                       f"manifest {man['source']['size']}")
    if not args.quick and man["source"]["sha256"]:
        print("sha256(source) ...", flush=True)
        got = sha256_file(src_path)
        if got != man["source"]["sha256"]:
            raise Mismatch(f"source sha256 {got[:16]}... != manifest "
                           f"{man['source']['sha256'][:16]}...")

    src = read_gguf(src_path)
    res = read_gguf(pack_dir / man["files"]["resident_gguf"])

    # 2. every tensor accounted for exactly once
    paged_names = {f"blk.{L}.{kind}.weight"
                   for L in man["model"]["moe_layers"] for kind in man["kinds"]}
    src_names = {t.name for t in src.tensors}
    res_names = {t.name for t in res.tensors}
    if missing := paged_names - src_names:
        raise Mismatch(f"manifest pages tensors the source lacks: {sorted(missing)[:3]}")
    if overlap := paged_names & res_names:
        raise Mismatch(f"tensors both paged and resident: {sorted(overlap)[:3]}")
    if lost := src_names - paged_names - res_names:
        raise Mismatch(f"source tensors not in pack or resident: {sorted(lost)[:3]}")
    if invented := res_names - src_names:
        raise Mismatch(f"resident tensors not in source: {sorted(invented)[:3]}")

    with open(src_path, "rb") as sf, \
         open(pack_dir / man["files"]["expert_pack"], "rb") as pf, \
         open(pack_dir / man["files"]["resident_gguf"], "rb") as rf:

        # 3. expert extents
        hdr = pf.read(len(PACK_MAGIC))
        if hdr != PACK_MAGIC:
            raise Mismatch(f"pack magic {hdr!r} != {PACK_MAGIC!r}")
        pf.seek(0, 2)
        if pf.tell() != man["totals"]["pack_file_bytes"]:
            raise Mismatch(f"pack file is {pf.tell()} bytes, manifest says "
                           f"{man['totals']['pack_file_bytes']}")

        groups = man["groups"]
        picked = groups
        if args.sample and args.sample < len(groups):
            picked = random.Random(0).sample(groups, args.sample)
        pack_bytes = 0
        for L, e, off in picked:
            lay = man["layers"][str(L)]
            for kind, part in lay["parts"].items():
                t = src.tensor(f"blk.{L}.{kind}.weight")
                pack_bytes += compare_ranges(
                    pf, off + part["rel_off"],
                    sf, src.abs_offset(t) + e * part["slice_bytes"],
                    part["slice_bytes"], f"extent L{L}/E{e}/{kind}")
        print(f"expert extents  {len(picked)}/{len(groups)} compared, "
              f"{human(pack_bytes)} — all byte-identical")

        # 4. resident tensors + verbatim metadata
        kv_len = src.kv_raw[1] - src.kv_raw[0]
        if res.kv_raw[1] - res.kv_raw[0] < kv_len:
            raise Mismatch("resident KV section shorter than source's")
        compare_ranges(rf, res.kv_raw[0], sf, src.kv_raw[0], kv_len,
                       "metadata KV section")
        res_bytes = 0
        for t in res.tensors:
            s = src.tensor(t.name)
            if (t.ne, t.ggml_type) != (s.ne, s.ggml_type):
                raise Mismatch(f"{t.name}: shape/type changed "
                               f"({t.ne}/{t.type_name} vs {s.ne}/{s.type_name})")
            res_bytes += compare_ranges(rf, res.abs_offset(t),
                                        sf, src.abs_offset(s), t.nbytes,
                                        f"resident {t.name}")
        print(f"resident        {len(res.tensors)} tensors, {human(res_bytes)} "
              f"+ verbatim metadata — all byte-identical")

    sampled = f" (sampled {args.sample} of {len(groups)} extents)" if picked is not groups else ""
    print(f"PASS in {time.time() - t0:.1f}s: repack is lossless{sampled}")


if __name__ == "__main__":
    main()
