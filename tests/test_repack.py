"""End-to-end tests for the offline repacker, no real model required.

Builds a tiny synthetic MoE GGUF from scratch (2 MoE layers x 4 experts, with
deliberately mixed quant types and a per-expert bias tensor to exercise the
keep-resident path), repacks it, and checks that:

  - the verifier passes on an honest repack;
  - the verifier FAILS when a single byte is flipped in either output file
    (a verifier that can't fail proves nothing);
  - the resident GGUF is a well-formed GGUF with the source metadata intact.

Run:  python3 tests/test_repack.py
"""

import contextlib
import io
import json
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gguf_lite  # noqa: E402
import repack_gguf  # noqa: E402
import verify_pack  # noqa: E402

ALIGN = 32  # GGUF default alignment (no general.alignment KV in the fixture)


def _kv_str(key: str, val: str) -> bytes:
    k, v = key.encode(), val.encode()
    return (struct.pack("<Q", len(k)) + k + struct.pack("<I", gguf_lite.T_STRING)
            + struct.pack("<Q", len(v)) + v)


def _kv_u32(key: str, val: int) -> bytes:
    k = key.encode()
    return (struct.pack("<Q", len(k)) + k
            + struct.pack("<II", gguf_lite.T_UINT32, val))


def _kv_str_array(key: str, vals: list) -> bytes:
    k = key.encode()
    out = (struct.pack("<Q", len(k)) + k
           + struct.pack("<IIQ", gguf_lite.T_ARRAY, gguf_lite.T_STRING, len(vals)))
    for v in vals:
        b = v.encode()
        out += struct.pack("<Q", len(b)) + b
    return out


# (name, ne in ggml order, ggml type id)  — mixed types on purpose
F32, F16, Q8_0 = 0, 1, 8
FIXTURE_TENSORS = [
    ("token_embd.weight", (8, 16), F32),
    ("blk.0.attn_q.weight", (8, 8), F16),
    ("blk.0.ffn_gate_inp.weight", (8, 4), F32),      # router: stays resident
    ("blk.0.ffn_gate_exps.weight", (32, 2, 4), Q8_0),
    ("blk.0.ffn_up_exps.weight", (8, 3, 4), F16),
    ("blk.0.ffn_down_exps.weight", (4, 8, 4), F32),
    ("blk.0.ffn_up_exps.bias", (3, 4), F32),         # expert-adjacent, not paged
    ("blk.1.ffn_gate_exps.weight", (32, 2, 4), Q8_0),
    ("blk.1.ffn_up_exps.weight", (8, 3, 4), F16),
    ("blk.1.ffn_down_exps.weight", (4, 8, 4), F32),
    ("output_norm.weight", (8,), F32),
]


def write_fixture_gguf(path: Path) -> None:
    kv_blob = (
        _kv_str("general.architecture", "moetest")
        + _kv_u32("moetest.block_count", 2)
        + _kv_u32("moetest.expert_count", 4)
        + _kv_u32("moetest.expert_used_count", 2)
        + _kv_str_array("tokenizer.ggml.tokens", ["<s>", "</s>", "hello", "world"])
    )
    infos, offset = b"", 0
    for name, ne, ttype in FIXTURE_TENSORS:
        n = name.encode()
        infos += (struct.pack("<Q", len(n)) + n + struct.pack("<I", len(ne))
                  + struct.pack(f"<{len(ne)}Q", *ne)
                  + struct.pack("<IQ", ttype, offset))
        offset = gguf_lite.align_up(offset + gguf_lite.tensor_nbytes(ne, ttype), ALIGN)

    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(FIXTURE_TENSORS), 5))
        f.write(kv_blob + infos)
        f.write(b"\x00" * (gguf_lite.align_up(f.tell(), ALIGN) - f.tell()))
        rng = random.Random(1234)  # deterministic, distinct bytes per tensor
        for name, ne, ttype in FIXTURE_TENSORS:
            nbytes = gguf_lite.tensor_nbytes(ne, ttype)
            f.write(rng.randbytes(nbytes))
            f.write(b"\x00" * (gguf_lite.align_up(nbytes, ALIGN) - nbytes))


def run_cli(module, argv) -> str:
    out = io.StringIO()
    old = sys.argv
    sys.argv = [module.__name__] + [str(a) for a in argv]
    try:
        with contextlib.redirect_stdout(out):
            module.main()
    finally:
        sys.argv = old
    return out.getvalue()


class RepackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "tiny-moe.gguf"
        self.out = self.dir / "tiny-moe.nvmoe"
        write_fixture_gguf(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def repack(self):
        return run_cli(repack_gguf, [self.src, "--out-dir", self.out])

    def verify(self):
        return run_cli(verify_pack, [self.out, self.src])

    def test_parser_reads_fixture(self):
        info = gguf_lite.read_gguf(self.src)  # includes layout self-validation
        self.assertEqual(info.kv["general.architecture"], "moetest")
        self.assertEqual(info.kv["moetest.expert_count"], 4)
        self.assertEqual(info.tensor("blk.0.ffn_gate_exps.weight").nbytes, 272)  # Q8_0
        self.assertEqual(info.tensor("blk.0.ffn_up_exps.weight").nbytes, 192)    # F16
        self.assertEqual(info.tensor("blk.0.ffn_down_exps.weight").nbytes, 512)  # F32

    def test_repack_then_verify_passes(self):
        out = self.repack()
        self.assertIn("ffn_up_exps.bias", out)  # oddity reported, kept resident
        vout = self.verify()
        self.assertIn("PASS", vout)
        self.assertIn("8/8 compared", vout)     # 2 layers x 4 experts

    def test_manifest_geometry(self):
        self.repack()
        man = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(man["totals"]["n_groups"], 8)
        self.assertEqual(man["model"]["moe_layers"], [0, 1])
        for L, e, off in man["groups"]:
            self.assertEqual(off % 4096, 0)
        # group = 68 (Q8_0) + 48 (F16) + 128 (F32) bytes, padded to one 4KB block
        lay = man["layers"]["0"]
        self.assertEqual(lay["group_bytes"], 244)
        self.assertEqual(lay["group_stride"], 4096)
        parts = lay["parts"]
        self.assertEqual(list(parts), ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"])
        self.assertEqual(parts["ffn_down_exps"]["rel_off"], 68 + 48)

    def test_resident_gguf_is_valid_and_complete(self):
        self.repack()
        res = gguf_lite.read_gguf(self.out / "resident.gguf")  # validates layout
        names = {t.name for t in res.tensors}
        self.assertNotIn("blk.0.ffn_gate_exps.weight", names)
        self.assertIn("blk.0.ffn_gate_inp.weight", names)   # router kept
        self.assertIn("blk.0.ffn_up_exps.bias", names)      # bias kept
        self.assertEqual(res.kv["nvmoe.pack.version"], 1)
        self.assertEqual(res.kv["general.architecture"], "moetest")
        self.assertEqual(len(names), len(FIXTURE_TENSORS) - 6)

    def test_verify_catches_pack_corruption(self):
        self.repack()
        man = json.loads((self.out / "manifest.json").read_text())
        off = man["groups"][3][2] + 100  # inside an extent, past the first part
        self._flip_byte(self.out / "experts.pack", off)
        with self.assertRaises(SystemExit) as cm:
            self.verify()
        self.assertIn("FAIL", str(cm.exception))
        self.assertIn("extent", str(cm.exception))

    def test_verify_catches_resident_corruption(self):
        self.repack()
        res = gguf_lite.read_gguf(self.out / "resident.gguf")
        t = res.tensor("token_embd.weight")
        self._flip_byte(self.out / "resident.gguf", res.abs_offset(t) + 5)
        with self.assertRaises(SystemExit) as cm:
            self.verify()
        self.assertIn("FAIL", str(cm.exception))
        self.assertIn("token_embd", str(cm.exception))

    def test_dry_run_writes_nothing(self):
        run_cli(repack_gguf, [self.src, "--out-dir", self.out, "--dry-run"])
        self.assertFalse(self.out.exists())

    def _flip_byte(self, path: Path, off: int):
        with open(path, "r+b") as f:
            f.seek(off)
            b = f.read(1)
            f.seek(off)
            f.write(bytes([b[0] ^ 0xFF]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
