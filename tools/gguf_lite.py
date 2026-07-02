"""Minimal GGUF reader/writer for the nvmoe repacker. Python stdlib only.

This is NOT a general GGUF library. It does exactly what the offline repacker
(tools/repack_gguf.py) and verifier (tools/verify_pack.py) need:

  - parse the header, all metadata KVs, and all tensor infos of a GGUF v2/v3
    little-endian file, without loading tensor data;
  - compute each tensor's exact byte size from its dims + ggml type (validated
    against the file layout, so a wrong entry in the type table cannot slip
    through silently);
  - write a new GGUF that keeps a subset of the tensors, copying the metadata
    section verbatim (plus optional appended KVs) and re-packing tensor data.

Why not the `gguf` pip package: the repo promise is "nothing to install", and
tools/ stays importable-from-nowhere. The format is stable and small enough
that a single-file reader is the more auditable choice.

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF metadata value types
(
    T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32,
    T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64,
) = range(13)

_SCALAR_FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}

# ggml tensor types: id -> (name, elements per block, bytes per block).
# Mirrors ggml.h / ggml.c type traits. The parser cross-checks every tensor's
# computed size against the gap to the next tensor's offset, so a mistake here
# fails loudly at parse time instead of corrupting a repack.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    # 4, 5: Q4_2/Q4_3 removed from ggml
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    # 31-33: Q4_0_4_4 etc. removed from ggml
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    # 36-38: IQ4_NL_4_4 etc. removed from ggml
    39: ("MXFP4", 32, 17),
}


def ggml_type_name(type_id: int) -> str:
    if type_id in GGML_TYPES:
        return GGML_TYPES[type_id][0]
    return f"UNKNOWN_{type_id}"


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def tensor_nbytes(ne: tuple[int, ...], ggml_type: int) -> int:
    """Exact byte size of a contiguous ggml tensor (GGUF stores ne[0] fastest)."""
    if ggml_type not in GGML_TYPES:
        raise ValueError(
            f"unknown ggml type id {ggml_type} — add it to GGML_TYPES "
            f"(see ggml.h type traits)"
        )
    name, blk, blk_bytes = GGML_TYPES[ggml_type]
    if ne[0] % blk != 0:
        raise ValueError(f"ne[0]={ne[0]} not divisible by {name} block size {blk}")
    row_bytes = ne[0] // blk * blk_bytes
    n_rows = 1
    for d in ne[1:]:
        n_rows *= d
    return row_bytes * n_rows


@dataclass
class TensorInfo:
    name: str
    ne: tuple[int, ...]      # ggml order: ne[0] is the fastest-moving dim
    ggml_type: int
    offset: int              # relative to the file's tensor-data start
    nbytes: int              # exact size from dims + type

    @property
    def type_name(self) -> str:
        return ggml_type_name(self.ggml_type)


@dataclass
class GGUFInfo:
    path: Path
    version: int
    alignment: int
    tensor_count: int
    kv_count: int
    kv: dict[str, object]            # scalar/string values only; arrays -> summary str
    kv_raw: tuple[int, int]          # absolute [start, end) of the KV section bytes
    tensors: list[TensorInfo] = field(default_factory=list)
    data_start: int = 0              # absolute file offset of the tensor-data section
    file_size: int = 0

    def tensor(self, name: str) -> TensorInfo:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(name)

    def abs_offset(self, t: TensorInfo) -> int:
        return self.data_start + t.offset


# ---------------------------------------------------------------- reading ---

def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"unexpected EOF (wanted {n} bytes, got {len(b)})")
    return b


def _read_scalar(f: BinaryIO, vtype: int):
    fmt = _SCALAR_FMT[vtype]
    return struct.unpack(fmt, _read_exact(f, struct.calcsize(fmt)))[0]


def _read_string(f: BinaryIO) -> str:
    (n,) = struct.unpack("<Q", _read_exact(f, 8))
    return _read_exact(f, n).decode("utf-8")


def _skip_value(f: BinaryIO, vtype: int) -> None:
    """Walk past a KV value without keeping it (used for big arrays)."""
    if vtype == T_STRING:
        (n,) = struct.unpack("<Q", _read_exact(f, 8))
        f.seek(n, 1)
    elif vtype == T_ARRAY:
        etype, count = struct.unpack("<IQ", _read_exact(f, 12))
        if etype in _SCALAR_FMT:
            f.seek(struct.calcsize(_SCALAR_FMT[etype]) * count, 1)
        else:
            for _ in range(count):
                _skip_value(f, etype)
    elif vtype in _SCALAR_FMT:
        f.seek(struct.calcsize(_SCALAR_FMT[vtype]), 1)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf(path: str | Path) -> GGUFInfo:
    """Parse header, KVs, and tensor infos. Never reads tensor data."""
    path = Path(path)
    with open(path, "rb") as f:
        if _read_exact(f, 4) != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file (bad magic)")
        version, = struct.unpack("<I", _read_exact(f, 4))
        if version not in (2, 3):
            raise ValueError(
                f"{path}: GGUF version {version} unsupported (need 2 or 3; "
                f"v1 and big-endian files are not handled)"
            )
        tensor_count, kv_count = struct.unpack("<QQ", _read_exact(f, 16))

        kv_start = f.tell()
        kv: dict[str, object] = {}
        for _ in range(kv_count):
            key = _read_string(f)
            vtype, = struct.unpack("<I", _read_exact(f, 4))
            if vtype == T_STRING:
                kv[key] = _read_string(f)
            elif vtype in _SCALAR_FMT:
                kv[key] = _read_scalar(f, vtype)
            elif vtype == T_ARRAY:
                pos = f.tell()
                etype, count = struct.unpack("<IQ", _read_exact(f, 12))
                f.seek(pos)
                _skip_value(f, vtype)
                kv[key] = f"<array type={etype} n={count}>"
            else:
                raise ValueError(f"{path}: KV {key!r} has unknown value type {vtype}")
        kv_end = f.tell()

        alignment = int(kv.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))

        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = _read_string(f)
            n_dims, = struct.unpack("<I", _read_exact(f, 4))
            ne = struct.unpack(f"<{n_dims}Q", _read_exact(f, 8 * n_dims))
            ggml_type, offset = struct.unpack("<IQ", _read_exact(f, 12))
            tensors.append(
                TensorInfo(name, tuple(int(d) for d in ne), ggml_type, offset,
                           tensor_nbytes(ne, ggml_type))
            )

        data_start = align_up(f.tell(), alignment)
        f.seek(0, 2)
        file_size = f.tell()

    info = GGUFInfo(path, version, alignment, tensor_count, kv_count,
                    kv, (kv_start, kv_end), tensors, data_start, file_size)
    _validate_layout(info)
    return info


def _validate_layout(info: GGUFInfo) -> None:
    """Cross-check computed tensor sizes against the actual file layout.

    Each tensor must occupy [offset, offset+nbytes) and the next tensor must
    start at the aligned end. This catches a wrong GGML_TYPES entry (or a
    corrupt file) immediately, which matters because the repacker slices
    expert tensors by these computed sizes.
    """
    by_off = sorted(info.tensors, key=lambda t: t.offset)
    for i, t in enumerate(by_off):
        if t.offset % info.alignment != 0:
            raise ValueError(f"{t.name}: offset {t.offset} not {info.alignment}-aligned")
        end = (by_off[i + 1].offset if i + 1 < len(by_off)
               else info.file_size - info.data_start)
        gap = end - t.offset
        if not (t.nbytes <= gap < t.nbytes + info.alignment):
            raise ValueError(
                f"{t.name}: computed nbytes {t.nbytes} ({t.type_name}, ne={t.ne}) "
                f"does not fit file layout gap {gap} — GGML_TYPES table wrong "
                f"or file corrupt"
            )


# ---------------------------------------------------------------- writing ---

def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pack_kv(key: str, vtype: int, value) -> bytes:
    out = _pack_string(key) + struct.pack("<I", vtype)
    if vtype == T_STRING:
        out += _pack_string(value)
    elif vtype in _SCALAR_FMT:
        out += struct.pack(_SCALAR_FMT[vtype], value)
    else:
        raise ValueError(f"cannot serialize KV type {vtype}")
    return out


def copy_range(src: BinaryIO, dst: BinaryIO, src_off: int, nbytes: int,
               chunk: int = 8 << 20) -> None:
    """Stream nbytes from src@src_off to dst's current position."""
    src.seek(src_off)
    left = nbytes
    while left:
        b = src.read(min(chunk, left))
        if not b:
            raise EOFError(f"EOF while copying (still needed {left} bytes)")
        dst.write(b)
        left -= len(b)


def write_subset_gguf(
    src_info: GGUFInfo,
    src_f: BinaryIO,
    out_path: str | Path,
    keep: list[TensorInfo],
    extra_kvs: list[tuple[str, int, object]] | None = None,
) -> None:
    """Write a GGUF containing only `keep` tensors (data copied from src).

    The metadata KV section is copied byte-for-byte from the source, so the
    output stays loadable by anything that loaded the source; `extra_kvs`
    (e.g. nvmoe provenance) are appended after it. Tensor data is re-packed
    tightly in the order given, zero-padded to the source's alignment, which
    is exactly the layout llama.cpp's GGUF reader validates.
    """
    extra_kvs = extra_kvs or []
    align = src_info.alignment

    with open(out_path, "wb") as out:
        out.write(GGUF_MAGIC)
        out.write(struct.pack("<I", src_info.version))
        out.write(struct.pack("<QQ", len(keep), src_info.kv_count + len(extra_kvs)))

        kv_start, kv_end = src_info.kv_raw
        copy_range(src_f, out, kv_start, kv_end - kv_start)
        for key, vtype, value in extra_kvs:
            out.write(_pack_kv(key, vtype, value))

        new_offset = 0
        for t in keep:
            out.write(_pack_string(t.name))
            out.write(struct.pack("<I", len(t.ne)))
            out.write(struct.pack(f"<{len(t.ne)}Q", *t.ne))
            out.write(struct.pack("<IQ", t.ggml_type, new_offset))
            new_offset = align_up(new_offset + t.nbytes, align)

        out.write(b"\x00" * (align_up(out.tell(), align) - out.tell()))
        for t in keep:
            copy_range(src_f, out, src_info.abs_offset(t), t.nbytes)
            # pad every tensor, including the last: ggml's strict reader slurps
            # sum-of-padded-sizes in one read, and gguf-py writes files this way
            out.write(b"\x00" * (align_up(t.nbytes, align) - t.nbytes))
