"""Tests for the Phase 3 planner, no real model or trace required.

Reuses test_repack's synthetic tiny-MoE fixture for the GGUF-geometry path
and a hand-built trace for the hit-rate transfer math. The postdiction
against the measured reference models is a separate, model-download-needing
check: `python3 tools/plan.py --postdict`.

Run:  python3 tests/test_plan.py
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from test_repack import write_fixture_gguf  # noqa: E402
import plan  # noqa: E402


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "tiny-moe.gguf"
        write_fixture_gguf(self.src)
        self.geo = plan.geometry_from_gguf(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fixture_geometry(self):
        g = self.geo
        self.assertEqual(g.arch, "moetest")
        self.assertEqual(g.moe_layers, 2)
        self.assertEqual(g.n_expert, 4)
        self.assertEqual(g.top_k, 2)
        self.assertEqual(g.total_experts, 8)
        self.assertEqual(g.cliff_frac, 0.5)
        # both layers share one extent size, 4KB-aligned
        self.assertEqual(len(g.strides), 1)
        stride, n_layers = g.strides[0]
        self.assertEqual(n_layers, 2)
        self.assertEqual(stride % plan.EXTENT_ALIGN, 0)
        # per-token active bytes: top_k experts in every MoE layer
        self.assertEqual(g.active_bytes, 2 * 2 * stride)
        self.assertEqual(g.paged_bytes, 8 * stride)
        # resident = everything that isn't a routed-expert weight
        self.assertGreater(g.resident_bytes, 0)

    def test_preset_geometry_matches_sim(self):
        g = plan.geometry_from_preset("deepseek-r1-671b")
        self.assertEqual(g.moe_layers, 58)
        self.assertEqual(g.total_experts, 58 * 256)
        self.assertAlmostEqual(g.cliff_frac, 8 / 256)
        # ~129GB of experts, as the sim preset says
        self.assertAlmostEqual(g.paged_bytes / 1e9, 129, delta=2)


class HitModelTests(unittest.TestCase):
    # 4 layers x 8 experts, top-2; token t uses experts (t%8, (t+1)%8) in
    # every layer -- perfectly cyclic, the LRU worst case.
    def make_trace(self, n=600):
        return [[(l, t % 8) for l in range(4)] + [(l, (t + 1) % 8) for l in range(4)]
                for t in range(n)]

    def test_below_cliff_is_zero(self):
        trace = self.make_trace()
        self.assertEqual(plan.hit_rate(trace, 0.1, cliff_target=0.25), 0.0)

    def test_full_cache_hits_everything(self):
        trace = self.make_trace()
        self.assertGreater(plan.hit_rate(trace, 1.0, cliff_target=0.25), 0.99)

    def test_cliff_normalization_is_identity_when_cliffs_match(self):
        # target cliff == source cliff (2/8): the transfer must run the sim
        # at the target fraction itself
        trace = self.make_trace()
        total_src, cliff_src = plan.trace_geometry(trace)
        self.assertEqual(total_src, 32)
        self.assertAlmostEqual(cliff_src, 0.25)
        direct = plan.hit_rate(trace, 0.5, cliff_target=0.25)
        from cache_sim import simulate_lru
        expected = simulate_lru(trace, 16, warmup_tokens=200).hit_rate
        self.assertAlmostEqual(direct, expected)

    def test_nvme_eff_scales_with_ssd(self):
        slow = plan.nvme_eff(3e6, 3.5)
        fast = plan.nvme_eff(3e6, 7.0)
        self.assertAlmostEqual(fast / slow, 2.0)
        # bigger extents fetch faster per byte, up to the plateau
        self.assertGreater(plan.nvme_eff(6e6, 7.0), plan.nvme_eff(3e6, 7.0))

    def test_predict_monotonic_in_cache(self):
        # skewed random routing (cyclic would be LRU's flat worst case:
        # 0% hits at every capacity below the full cycle)
        import random
        rng = random.Random(7)
        trace = [[(l, min(int(8 * rng.random() ** 2), 7)) for l in range(4)
                  for _ in range(2)] for _ in range(600)]
        geo = plan.Geometry("t.gguf", "moetest", 4, 8, 2,
                            [(4096 * 1024, 4)], int(1e9), 0.1)
        tps = [plan.predict_tps(geo, geo.paged_bytes * f, trace, 7.0, 420.0)[3]
               for f in (0.3, 0.6, 1.0)]
        self.assertLess(tps[0], tps[1])
        self.assertLessEqual(tps[1], tps[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
