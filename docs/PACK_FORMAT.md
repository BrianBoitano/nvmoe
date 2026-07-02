# nvmoe pack format (v1)

The Phase 2 runtime does not read GGUF files at inference time. It reads a
**pack**: the output of `tools/repack_gguf.py`, which rearranges a MoE GGUF
into the layout the paging path actually wants. This document specifies that
layout precisely enough to write an independent reader.

## Why repack at all

A GGUF stores each layer's routed experts as one merged 3D tensor
(`blk.L.ffn_gate_exps.weight` with shape `[in, out, n_expert]`, expert index
outermost). That is the wrong unit for paging twice over:

- A cache miss needs **one expert of one layer** — and it always needs that
  expert's gate, up, and down matrices *together*, because routing selected
  the expert, not a matrix. In the GGUF those three slices live megabytes
  apart (three separate tensors).
- O_DIRECT reads want 4KB-aligned offsets and lengths. GGUF aligns tensors
  (default 32B), not expert slices.

So the repacker makes the fetch unit physical: one extent per
`(layer, expert)` holding gate+up+down back-to-back, starting on a 4KB
boundary. A miss becomes a single aligned read of exactly the needed bytes.

Slicing is pure byte arithmetic — expert `e`'s slice of a merged tensor is
`[e * nbytes/n_expert, (e+1) * nbytes/n_expert)` because the expert dimension
is outermost and ggml tensors are contiguous. No dequantization, no float
handling: the pack is byte-identical source data, rearranged, at any quant
type. `tools/verify_pack.py` proves this for every extent.

## A pack is a directory of three files

```
<model>.nvmoe/
  resident.gguf     everything that is NOT a routed-expert weight
  experts.pack      the routed-expert extents
  manifest.json     where everything is (the runtime's source of truth)
```

### resident.gguf

A normal, well-formed GGUF (readable by ggml's own reader) containing every
source tensor except the paged `blk.*.ffn_{gate,up,down}_exps.weight` ones.
Attention, dense FFN, shared experts (`*_shexp`), the router
(`ffn_gate_inp`), norms, embeddings, and any per-expert *biases* (tiny, not
worth paging) all stay here.

The metadata KV section is copied **byte-for-byte** from the source — same
architecture, tokenizer, hparams — with two KVs appended:

| key | type | value |
|---|---|---|
| `nvmoe.pack.version` | uint32 | `1` |
| `nvmoe.pack.manifest` | string | manifest filename |

Stock llama.cpp will parse this file but refuse to load it as a model (the
expert tensors are "missing" — they're in the pack). That is expected; only
the nvmoe runtime consumes it.

### experts.pack

```
offset 0        8 bytes   magic "NVMOEPK1"
offset 8        4 bytes   uint32 LE format version (1)
offset 12       zeros up to 4096
offset 4096     extent for (layer 0, expert 0)
...             extents, layer-major: all experts of MoE layer 0, then 1, ...
```

Each extent is that expert's matrices concatenated in fixed kind order
(gate, up, down — whichever exist for the arch), zero-padded to the next
4KB boundary. Extent offsets/strides are explicit in the manifest; within a
layer, extents are uniform, but they may differ **across** layers (mixed
quants like Q4_K_M put Q6_K on some layers — Qwen3-30B-A3B packs at both
2.5MB and 2.9MB per expert).

4KB alignment is the largest common logical block size (512e and 4Kn drives
both accept 4KB-aligned O_DIRECT reads), and expert slices happen to be
4KB-multiples for every quant we've packed, so the padding overhead is ~0%.

### manifest.json

Everything the runtime needs to service a miss without touching GGUF parsing:

```jsonc
{
  "format": "nvmoe-pack",
  "format_version": 1,
  "alignment": 4096,
  "pack_header_bytes": 4096,
  "source": { "file": "...", "size": ..., "sha256": "...", "gguf_version": 3 },
  "model": {
    "arch": "qwen3moe",
    "n_layers_total": 48,
    "moe_layers": [0, 1, ...],       // layers that have routed experts
    "n_expert": 128,
    "n_expert_used": 8               // top-k
  },
  "files": { "resident_gguf": "resident.gguf", "expert_pack": "experts.pack" },
  "kinds": ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"],
  "layers": {                        // per-layer extent geometry
    "0": {
      "parts": {                     // in on-disk order within the extent
        "ffn_gate_exps": { "ggml_type": 12, "type_name": "Q4_K",
                           "ne": [2048, 768],   // ggml order, expert dim dropped
                           "slice_bytes": 884736,
                           "rel_off": 0 },      // offset inside the extent
        "ffn_up_exps":   { ... },
        "ffn_down_exps": { ... }
      },
      "group_bytes": 2654208,        // payload bytes per extent
      "group_stride": 2654208        // padded to alignment
    }, ...
  },
  "groups": [ [layer, expert, absolute_pack_offset], ... ],  // layer-major
  "totals": { "n_groups": 6144, "paged_bytes": ..., "pack_file_bytes": ...,
              "resident_tensors": 435, "resident_bytes": ... }
}
```

To read expert `e` of layer `L`: look up its `[L, e, offset]` group, read
`layers[L].group_stride` bytes at `offset` (already aligned on both ends),
then hand each part to the compute side at `offset + parts[k].rel_off`.
`groups` is technically derivable from `layers` + ordering; it is explicit so
a future mixed-precision repack (hot experts at higher bits, per-extent
types) only has to bump `format_version` and move `parts` into each group.

## Guarantees (and how they're checked)

1. **Lossless**: every extent and every resident tensor is byte-identical to
   the corresponding source bytes. `verify_pack.py` compares all of them
   (not a sample) plus the metadata section, against the source file, using
   only the manifest — offsets are never re-derived from repacker code, so a
   planning bug cannot vouch for itself.
2. **Complete**: paged ∪ resident == source tensors, exactly, no overlap.
3. **Well-formed resident**: `resident.gguf` passes ggml's strict reader
   (`llama-gguf <file> r n` reads all KVs + full tensor data).

```bash
python3 tools/repack_gguf.py models/olmoe-q4_0.gguf          # ~seconds
python3 tools/verify_pack.py models/olmoe-q4_0.nvmoe models/olmoe-q4_0.gguf
python3 tests/test_repack.py                                 # no model needed
```

## Limits (v1)

- Single-file GGUFs only — merge split shards first (`llama-gguf-split --merge`).
- Merged-expert GGUFs only (`blk.*.ffn_*_exps.weight`, 3D). The ancient
  per-expert layout (`blk.N.ffn_gate.E.weight`) predates every model nvmoe
  targets; re-convert with current llama.cpp if you hit it.
- Little-endian GGUF v2/v3.
