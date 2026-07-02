#!/usr/bin/env python3
"""Export a pack's extent table as TSV for tools that shouldn't parse JSON.

The paging benchmark (paging/nvmoe_iobench.c) is deliberately dependency-free
C; teaching it JSON would be all liability and no insight. This exports the
one thing it needs from manifest.json — where every extent lives:

    <absolute_offset>\t<read_nbytes>        one line per (layer, expert)

read_nbytes is the padded stride (offset and length both 4KB-multiples), i.e.
exactly what an O_DIRECT read of that extent must ask for.

Usage:
    python3 tools/pack_extents.py models/olmoe-q4_0.nvmoe        # writes extents.tsv
    python3 tools/pack_extents.py <pack_dir> -o /dev/stdout      # or anywhere else
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="manifest.json -> extents.tsv")
    ap.add_argument("pack_dir", help="directory written by repack_gguf.py")
    ap.add_argument("-o", "--out", help="output path (default: <pack_dir>/extents.tsv)")
    args = ap.parse_args()

    pack_dir = Path(args.pack_dir)
    man = json.loads((pack_dir / "manifest.json").read_text())
    out_path = Path(args.out) if args.out else pack_dir / "extents.tsv"

    lines = []
    for layer, expert, offset in man["groups"]:
        stride = man["layers"][str(layer)]["group_stride"]
        lines.append(f"{offset}\t{stride}\n")
    out_path.write_text("".join(lines))
    if out_path != Path("/dev/stdout"):
        print(f"{len(lines)} extents -> {out_path}")


if __name__ == "__main__":
    main()
