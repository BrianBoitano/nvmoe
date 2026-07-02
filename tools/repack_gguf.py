#!/usr/bin/env python3
"""Offline repacker: split a MoE GGUF into a resident GGUF + an expert pack.

This is Phase 2 step 1 (see docs/DESIGN.md). The runtime wants two things a
stock GGUF can't give it:

  1. Routed-expert weights addressable per (layer, expert) at 4KB-aligned
     offsets, so a cache miss becomes ONE O_DIRECT read of exactly the bytes
     that expert needs (its gate+up+down matrices, stored back-to-back).
  2. Everything else (attention, dense FFN, shared experts, router, norms,
     embeddings) in a normal GGUF that loads to VRAM the boring way.

So we repack, offline, once per model:

  models/foo.gguf  ->  out_dir/resident.gguf    all non-routed-expert tensors,
                                                metadata copied verbatim
                       out_dir/experts.pack     one aligned extent per
                                                (layer, expert), layer-major
                       out_dir/manifest.json    offsets/sizes/types for every
                                                extent + model geometry

The fetch unit is one expert's full FFN (gate+up+down slices concatenated):
routing always needs the three matrices together, so packing them adjacent
turns a miss into a single sequential read. In a GGUF, `blk.L.ffn_*_exps.weight`
is a 3D tensor with the expert index as the outermost (slowest) dimension,
which makes each expert's slice contiguous: slicing is pure byte math, no
dequantization, and the output is provably byte-identical to the source
(prove it with tools/verify_pack.py).

Usage:
    python3 tools/repack_gguf.py models/olmoe-q4_0.gguf
    python3 tools/repack_gguf.py models/qwen3-30b-a3b-q4_k_m.gguf --out-dir /fast/qwen3
    python3 tools/repack_gguf.py anything.gguf --dry-run   # plan + sizes only

Stdlib only. Reads and writes are streamed (peak RAM well under 100MB
regardless of model size).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_lite import (  # noqa: E402
    GGUFInfo, TensorInfo, T_STRING, T_UINT32,
    align_up, copy_range, ggml_type_name, read_gguf, write_subset_gguf,
)

PACK_MAGIC = b"NVMOEPK1"
PACK_VERSION = 1
PACK_HEADER_BYTES = 4096          # one aligned block; extents start after it
EXTENT_ALIGN = 4096               # O_DIRECT-friendly: covers 512B and 4Kn LBAs

# Routed-expert weights use llama.cpp's merged 3D layout: blk.<L>.ffn_<kind>_exps.weight
# with ne = (in_features, out_features, n_expert). Shared experts ("shexp"),
# the router ("ffn_gate_inp"), and per-expert biases do not match and stay
# resident — biases are tiny and some archs (GPT-OSS) keep them; paging them
# would buy nothing.
EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
KIND_ORDER = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]


def classify(info: GGUFInfo):
    """Split tensors into paged expert weights (by layer/kind) and resident."""
    paged: dict[int, dict[str, TensorInfo]] = {}
    resident: list[TensorInfo] = []
    oddities: list[str] = []
    for t in info.tensors:
        m = EXPS_RE.match(t.name)
        if m:
            if len(t.ne) != 3:
                raise SystemExit(
                    f"error: {t.name} is {len(t.ne)}-D, expected 3-D merged experts. "
                    f"Old per-expert GGUFs (blk.N.ffn_gate.E.weight) are not supported; "
                    f"re-convert the model with current llama.cpp."
                )
            layer, kind = int(m.group(1)), f"ffn_{m.group(2)}_exps"
            paged.setdefault(layer, {})[kind] = t
        else:
            resident.append(t)
            if "_exps." in t.name:
                oddities.append(t.name)
    return paged, resident, oddities


def plan_pack(info: GGUFInfo, paged: dict[int, dict[str, TensorInfo]]):
    """Decide the on-disk layout of every extent; returns (layers, groups).

    layers: per-layer geometry (kinds present, per-expert slice bytes, types).
    groups: one record per (layer, expert) with its absolute pack offset.
    Layer-major order matches the runtime's access pattern (decode walks
    layers in order, so a layer's experts end up disk-adjacent).
    """
    moe_layers = sorted(paged)
    kinds = [k for k in KIND_ORDER if k in paged[moe_layers[0]]]
    for L in moe_layers:
        have = [k for k in KIND_ORDER if k in paged[L]]
        if have != kinds:
            raise SystemExit(f"error: layer {L} has expert kinds {have}, "
                             f"layer {moe_layers[0]} has {kinds} — mixed archs unsupported")

    n_expert = paged[moe_layers[0]][kinds[0]].ne[2]
    layers: dict[int, dict] = {}
    for L in moe_layers:
        parts, rel = {}, 0
        for kind in kinds:
            t = paged[L][kind]
            if t.ne[2] != n_expert:
                raise SystemExit(f"error: {t.name} has {t.ne[2]} experts, expected {n_expert}")
            slice_bytes = t.nbytes // t.ne[2]
            parts[kind] = {
                "ggml_type": t.ggml_type,
                "type_name": t.type_name,
                "ne": list(t.ne[:2]),
                "slice_bytes": slice_bytes,
                "rel_off": rel,
            }
            rel += slice_bytes
        layers[L] = {"parts": parts, "group_bytes": rel,
                     "group_stride": align_up(rel, EXTENT_ALIGN)}

    groups, off = [], PACK_HEADER_BYTES
    for L in moe_layers:
        for e in range(n_expert):
            groups.append({"layer": L, "expert": e, "offset": off})
            off += layers[L]["group_stride"]
    return layers, groups, kinds, n_expert, off  # off == final pack size


def write_pack(info: GGUFInfo, src_f, out_path: Path, paged, layers, groups) -> None:
    with open(out_path, "wb") as out:
        out.write(PACK_MAGIC + struct.pack("<I", PACK_VERSION))
        out.write(b"\x00" * (PACK_HEADER_BYTES - out.tell()))
        for g in groups:
            L = g["layer"]
            lay = layers[L]
            assert out.tell() == g["offset"], "pack writer out of sync with plan"
            for kind, part in lay["parts"].items():
                t = paged[L][kind]
                src_off = info.abs_offset(t) + g["expert"] * part["slice_bytes"]
                copy_range(src_f, out, src_off, part["slice_bytes"])
            out.write(b"\x00" * (lay["group_stride"] - lay["group_bytes"]))


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split a MoE GGUF into resident.gguf + experts.pack + manifest.json "
                    "(single-file GGUFs only; merge split GGUFs first)")
    ap.add_argument("gguf", help="source .gguf (any MoE model with merged _exps tensors)")
    ap.add_argument("--out-dir", help="output directory "
                    "(default: <source dir>/<source stem>.nvmoe/)")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip the source sha256 (saves one full read of the file)")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse + plan + print the summary, write nothing")
    args = ap.parse_args()

    src = Path(args.gguf)
    t0 = time.time()
    info = read_gguf(src)
    if int(info.kv.get("split.count", 0) or 0) > 1:
        raise SystemExit("error: this is one shard of a split GGUF; merge shards first "
                         "(llama-gguf-split --merge)")

    paged, resident, oddities = classify(info)
    if not paged:
        raise SystemExit("error: no blk.*.ffn_*_exps.weight tensors found — "
                         "not a MoE GGUF (or a dense model; nothing to page)")
    layers, groups, kinds, n_expert, pack_size = plan_pack(info, paged)

    arch = info.kv.get("general.architecture", "unknown")
    n_expert_kv = info.kv.get(f"{arch}.expert_count")
    if n_expert_kv is not None and int(n_expert_kv) != n_expert:
        raise SystemExit(f"error: {arch}.expert_count={n_expert_kv} but expert tensors "
                         f"have {n_expert} slices")

    paged_bytes = sum(lay["group_bytes"] for lay in layers.values()) * n_expert
    resident_bytes = sum(t.nbytes for t in resident)
    moe_layers = sorted(layers)

    print(f"source          {src}  ({human(info.file_size)}, GGUF v{info.version}, "
          f"align {info.alignment})")
    print(f"model           arch={arch}  layers={info.kv.get(f'{arch}.block_count', '?')} "
          f"(MoE: {len(moe_layers)})  experts/layer={n_expert} "
          f"top_k={info.kv.get(f'{arch}.expert_used_count', '?')}")
    print(f"paged           {len(groups)} extents = {len(moe_layers)} layers x {n_expert} experts, "
          f"kinds {'+'.join(k.split('_')[1] for k in kinds)}")
    per_layer_sizes = sorted({lay["group_bytes"] for lay in layers.values()})
    print(f"expert size     {' / '.join(human(s) for s in per_layer_sizes)} per expert "
          f"({human(paged_bytes)} total, {100 * paged_bytes / info.file_size:.1f}% of file)")
    types_used = sorted({p['type_name'] for lay in layers.values() for p in lay['parts'].values()})
    print(f"expert quants   {', '.join(types_used)}")
    print(f"resident        {len(resident)} tensors, {human(resident_bytes)}")
    if oddities:
        print(f"note            kept resident (expert-adjacent but not paged): "
              f"{', '.join(oddities[:4])}{' ...' if len(oddities) > 4 else ''}")
    pad_waste = pack_size - PACK_HEADER_BYTES - paged_bytes
    print(f"pack layout     {human(pack_size)} file, {human(pad_waste)} alignment padding "
          f"({100 * pad_waste / pack_size:.3f}%)")

    if args.dry_run:
        print("dry run: nothing written")
        return

    out_dir = Path(args.out_dir) if args.out_dir else src.parent / f"{src.stem}.nvmoe"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path, pack_path, man_path = (out_dir / "resident.gguf",
                                     out_dir / "experts.pack",
                                     out_dir / "manifest.json")

    src_sha = None
    if not args.no_hash:
        print("hashing source ...", flush=True)
        src_sha = sha256_file(src)

    print(f"writing {pack_path} ...", flush=True)
    with open(src, "rb") as src_f:
        write_pack(info, src_f, pack_path, paged, layers, groups)
        print(f"writing {res_path} ...", flush=True)
        write_subset_gguf(info, src_f, res_path, resident, extra_kvs=[
            ("nvmoe.pack.version", T_UINT32, PACK_VERSION),
            ("nvmoe.pack.manifest", T_STRING, man_path.name),
        ])

    manifest = {
        "format": "nvmoe-pack",
        "format_version": PACK_VERSION,
        "alignment": EXTENT_ALIGN,
        "pack_header_bytes": PACK_HEADER_BYTES,
        "source": {"file": src.name, "size": info.file_size,
                   "sha256": src_sha, "gguf_version": info.version},
        "model": {
            "arch": arch,
            "n_layers_total": int(info.kv.get(f"{arch}.block_count", 0) or 0),
            "moe_layers": moe_layers,
            "n_expert": n_expert,
            "n_expert_used": int(info.kv.get(f"{arch}.expert_used_count", 0) or 0),
        },
        "files": {"resident_gguf": res_path.name, "expert_pack": pack_path.name},
        "kinds": kinds,
        "layers": {str(L): layers[L] for L in moe_layers},
        "groups": [[g["layer"], g["expert"], g["offset"]] for g in groups],
        "totals": {"n_groups": len(groups), "paged_bytes": paged_bytes,
                   "pack_file_bytes": pack_size,
                   "resident_tensors": len(resident), "resident_bytes": resident_bytes},
    }
    man_path.write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"done in {time.time() - t0:.1f}s -> {out_dir}/")
    print(f"verify: python3 tools/verify_pack.py {out_dir} {src}")


if __name__ == "__main__":
    main()
