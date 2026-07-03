#!/usr/bin/env python3
"""nvmoe planner (Phase 3): read a MoE GGUF, probe/accept hardware numbers,
and emit a placement plan — cache budget, prefetch setting, expected decode
speed, and the exact commands to repack, verify, gate, and bench.

Usage:
    python3 tools/plan.py models/qwen3-30b-a3b-q4_k_m.gguf
    python3 tools/plan.py model.gguf --vram-gb 24 --nvme-gbps 12
    python3 tools/plan.py --preset deepseek-r1-671b        # model not on disk yet
    python3 tools/plan.py --postdict                       # validate vs measured

The tok/s model and its calibration:

    t_token   = t_compute + t_io
    t_compute = (resident_bytes + active_expert_bytes) / GPU_EFF_GBPS
    t_io      = active_expert_bytes * (1 - hit) / nvme_eff(extent_size)

  * hit comes from an LRU simulation over a REAL routing trace of the same
    routing family (traces/*.tokens.jsonl, committed), at the cache fraction
    the budget buys. Live hit-rate anchors measured on the runtime (real
    prompt, 128 decode steps, 2026-07-02) agree with the trace curves to a
    few points: qwen3 @4GB 83.6% live vs 80.5% sim; dsv2-lite @4GB 63.5%
    live vs 61.1% sim; gpt-oss @11GB 70.2% live.
  * GPU_EFF_GBPS = 420 GB/s effective decode-read bandwidth, fit on two
    anchors from runtime/README.md (Qwen3 warm 169.7 tok/s @12GB cache,
    V2-Lite pack-all-resident 234.7 tok/s) on the RTX 5070 Ti (896 GB/s
    spec). Scale it for your card with --gpu-eff-gbps.
  * nvme_eff: measured fetch bandwidth of the QD-4 runtime path by extent
    size on the 7 GB/s-class 990 PRO (4.4 GB/s at 2.9MB extents, 5.7 at
    5.7MB, 5.65 at 12.6MB); scaled by --nvme-gbps / 7.0.

Honesty note, verified by --postdict: this predicts REAL-WORKLOAD decode.
llama-bench's `-p 0` decode (generation from BOS, short horizon) routes far
more repetitively than real prompts and can land up to ~2x ABOVE the
prediction on flat-routing (DeepSeek-family) models and at small caches
on fine-grained-expert models; on skewed-routing models near-resident it
lands within ~±25%. The planner's range reflects that spread.
"""

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "sim"))

from cache_sim import simulate_lru               # noqa: E402
from trace_gen import load_trace_jsonl           # noqa: E402
from gguf_lite import read_gguf, align_up        # noqa: E402
from presets import MODELS as SIM_PRESETS        # noqa: E402

MIB = 1 << 20
EXTENT_ALIGN = 4096
EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")

GPU_EFF_GBPS = 420.0        # see module docstring
REFERENCE_NVME_GBPS = 7.0   # the SSD the nvme_eff anchors were measured on
PACK_OVERHEAD_WHEN_FITS = 0.13   # measured: V2-Lite pack-resident vs stock
DEFAULT_HEADROOM_GB = 1.5   # compute buffers + fragmentation, ballpark
PRED_RANGE = (0.7, 2.5)     # measured/predicted spread over the 11 anchors

# routing family -> committed real trace (all: 4 workloads, 1396 decode
# tokens). Curves transfer between models as hit(cache fraction of total
# experts), with the thrash cliff (top_k/n_expert) aligned between source
# and target (fine-grained models cliff lower). Measured decode-routing
# stats, most- to least-cacheable:
#   gptoss   top-10% experts carry 56.7% of traffic, token overlap 50.4%
#   qwen3    34.7% / 43.4%
#   glm      26.7% / 37.0%   (GLM-4.5-Air — between qwen3 and deepseek)
#   deepseek 17.7% / 24.2%   (fine-grained routing is flat — R1's family)
FAMILIES = {
    "qwen3":    "qwen3-all.tokens.jsonl",
    "gptoss":   "gptoss-all.tokens.jsonl",
    "glm":      "glm-all.tokens.jsonl",
    "deepseek": "dsv2lite-all.tokens.jsonl",
}
ARCH_FAMILY = {
    "qwen3moe": "qwen3", "qwen2moe": "qwen3", "olmoe": "qwen3",
    "qwen3next": "qwen3",
    "gpt-oss": "gptoss",
    "glm4moe": "glm",
    "deepseek2": "deepseek", "deepseek3": "deepseek",
    # sim/presets.py keys, for --preset plans of models not on disk
    "qwen3-30b-a3b": "qwen3", "qwen3-next-80b": "qwen3", "olmoe-7b": "qwen3",
    "gpt-oss-120b": "gptoss",
    "deepseek-r1-671b": "deepseek", "deepseek-v2-lite": "deepseek",
}


# --------------------------------------------------------------- geometry ---

class Geometry:
    """Everything the plan needs to know about one MoE model."""

    def __init__(self, name, arch, moe_layers, n_expert, top_k,
                 strides, resident_bytes, kv_gb_estimate):
        self.name = name
        self.arch = arch
        self.moe_layers = moe_layers          # count
        self.n_expert = n_expert
        self.top_k = top_k
        self.strides = strides                # [(extent_bytes, n_layers), ...]
        self.resident_bytes = resident_bytes
        self.kv_gb_estimate = kv_gb_estimate

    @property
    def total_experts(self):
        return self.moe_layers * self.n_expert

    @property
    def paged_bytes(self):
        return sum(s * n for s, n in self.strides) * self.n_expert

    @property
    def avg_extent(self):
        return sum(s * n for s, n in self.strides) / self.moe_layers

    @property
    def active_refs(self):
        return self.moe_layers * self.top_k

    @property
    def active_bytes(self):
        """Expert bytes one decode token touches (top_k in every MoE layer)."""
        return self.top_k * sum(s * n for s, n in self.strides)

    @property
    def cliff_frac(self):
        """Cache fraction below which LRU thrashes to 0%: one token's set."""
        return self.top_k / self.n_expert


def geometry_from_gguf(path: Path) -> Geometry:
    info = read_gguf(path)
    arch = info.kv.get("general.architecture", "?")
    by_layer: dict[int, int] = {}     # layer -> extent bytes (gate+up+down, aligned)
    n_expert = top_k = None
    resident = 0
    for t in info.tensors:
        m = EXPS_RE.match(t.name)
        if not m:
            resident += t.nbytes
            continue
        layer = int(m.group(1))
        if len(t.ne) != 3:
            raise SystemExit(f"{t.name} is not 3-D merged experts — "
                             f"re-convert with current llama.cpp")
        n_expert = t.ne[2]
        by_layer[layer] = by_layer.get(layer, 0) + t.nbytes // t.ne[2]
    if not by_layer:
        raise SystemExit(f"{path}: no routed-expert tensors found — not a "
                         f"merged-experts MoE GGUF (dense models can't page)")
    top_k = info.kv.get(f"{arch}.expert_used_count")
    if top_k is None:
        raise SystemExit(f"{path}: missing {arch}.expert_used_count KV")

    stride_counts: dict[int, int] = {}
    for layer, raw in sorted(by_layer.items()):
        stride = align_up(raw, EXTENT_ALIGN)
        stride_counts[stride] = stride_counts.get(stride, 0) + 1
    strides = sorted(stride_counts.items())

    kv_gb = estimate_kv_gb(info, arch)
    return Geometry(path.name, arch, len(by_layer), int(n_expert), int(top_k),
                    strides, resident, kv_gb)


def geometry_from_preset(key: str) -> Geometry:
    p = SIM_PRESETS[key]
    stride = align_up(p.expert_bytes, EXTENT_ALIGN)
    return Geometry(p.name, key, p.moe_layers, p.experts_per_layer, p.top_k,
                    [(stride, p.moe_layers)], int(p.always_on_gb * 1e9), p.kv_gb)


def estimate_kv_gb(info, arch, n_ctx=4096) -> float:
    """F16 KV cache at n_ctx. MLA models (deepseek2) store compressed KV;
    this generic estimate overshoots there — override with --kv-gb."""
    kv = info.kv
    layers = kv.get(f"{arch}.block_count")
    heads_kv = kv.get(f"{arch}.attention.head_count_kv")
    embd = kv.get(f"{arch}.embedding_length")
    heads = kv.get(f"{arch}.attention.head_count")
    if not all((layers, heads_kv, embd, heads)):
        return 1.0
    head_dim = kv.get(f"{arch}.attention.key_length", embd // heads)
    return 2 * layers * heads_kv * head_dim * 2 * n_ctx / 1e9


# ------------------------------------------------------------- hit model ---

def load_family_trace(family: str):
    path = _HERE.parent / "traces" / FAMILIES[family]
    if not path.exists():
        return None
    return load_trace_jsonl(path)


def trace_geometry(trace):
    """(total_experts, cliff_frac) of the traced model, from the trace itself."""
    layers = max(k[0] for tok in trace[:50] for k in tok) + 1
    n_expert = max(k[1] for tok in trace for k in tok) + 1
    top_k = round(sum(len(t) for t in trace[:50]) / 50 / layers)
    return layers * n_expert, top_k / n_expert


def hit_rate(trace, frac_target, cliff_target) -> float:
    """Steady-state LRU hit rate at a cache fraction, transferred from the
    family trace with the thrash cliff aligned (a model whose per-token
    active set is 3% of its experts and one whose set is 9% hit their
    floors at different fractions; normalize to distance above the cliff)."""
    if frac_target <= cliff_target:
        return 0.0
    total_src, cliff_src = trace_geometry(trace)
    x = (frac_target - cliff_target) / (1.0 - cliff_target)
    frac_src = cliff_src + x * (1.0 - cliff_src)
    capacity = min(int(frac_src * total_src), total_src)
    return simulate_lru(trace, capacity, warmup_tokens=200).hit_rate


def nvme_eff(extent_bytes: float, nvme_gbps: float) -> float:
    """Effective QD-4 fetch bandwidth (bytes/s) at an extent size. Anchors
    measured live on the runtime, 990 PRO, 2026-07-02."""
    anchors = [(2.9e6, 4.4e9), (5.7e6, 5.7e9), (12.6e6, 5.65e9)]
    scale = nvme_gbps / REFERENCE_NVME_GBPS
    if extent_bytes <= anchors[0][0]:
        return anchors[0][1] * (extent_bytes / anchors[0][0]) * scale
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if extent_bytes <= x1:
            return (y0 + (y1 - y0) * (extent_bytes - x0) / (x1 - x0)) * scale
    return anchors[-1][1] * scale


def predict_tps(geo: Geometry, cache_bytes: float, trace,
                nvme_gbps: float, gpu_eff_gbps: float):
    """Returns (hit, t_compute_ms, t_io_ms, tok/s)."""
    frac = cache_bytes / geo.paged_bytes
    hit = hit_rate(trace, frac, geo.cliff_frac) if trace else 0.0
    miss_bytes = geo.active_bytes * (1.0 - hit)
    t_io = miss_bytes / nvme_eff(geo.avg_extent, nvme_gbps)
    t_c = (geo.resident_bytes + geo.active_bytes) / (gpu_eff_gbps * 1e9)
    return hit, t_c * 1e3, t_io * 1e3, 1.0 / (t_c + t_io)


# ------------------------------------------------------------------ plan ---

def emit_plan(geo: Geometry, args) -> None:
    vram = args.vram_gb * 1e9
    kv_gb = args.kv_gb if args.kv_gb is not None else geo.kv_gb_estimate
    fixed = geo.resident_bytes + kv_gb * 1e9 + args.headroom_gb * 1e9
    cache_bytes = vram - fixed
    total_gb = (geo.resident_bytes + geo.paged_bytes) / 1e9

    print(f"\n=== nvmoe plan: {geo.name} ===")
    print(f"arch                : {geo.arch}  "
          f"({geo.moe_layers} MoE layers x {geo.n_expert} experts, top-{geo.top_k})")
    print(f"model size          : {total_gb:.1f}GB "
          f"({geo.paged_bytes / 1e9:.1f}GB routed experts in "
          f"{geo.total_experts} extents of {geo.avg_extent / 1e6:.1f}MB, "
          f"{geo.resident_bytes / 1e9:.1f}GB resident)")
    print(f"hardware            : {args.vram_gb:.0f}GB VRAM, "
          f"{args.nvme_gbps:.1f}GB/s NVMe"
          + (" (measure yours: tools/nvme_probe.py)" if args.nvme_gbps == 7.0 else ""))
    print(f"VRAM budget         : {geo.resident_bytes / 1e9:.1f} resident "
          f"+ {kv_gb:.1f} KV(est) + {args.headroom_gb:.1f} headroom "
          f"-> {max(cache_bytes, 0) / 1e9:.1f}GB expert cache")
    print(f"prefill sweep       : ~{geo.paged_bytes / (args.nvme_gbps * 1e9):.0f}s "
          f"per pass over all experts (long prompts stream the whole store; "
          f"decode-optimized by design)")

    # 1. does it just fit? then don't page.
    if geo.resident_bytes + geo.paged_bytes + kv_gb * 1e9 + args.headroom_gb * 1e9 <= vram:
        print(f"\nVERDICT: fits in VRAM entirely — use stock llama.cpp "
              f"(-ngl 99). Paging a model that fits costs "
              f"~{PACK_OVERHEAD_WHEN_FITS:.0%} (measured, runtime/README.md).")
        return

    if cache_bytes <= 0:
        print(f"\nVERDICT: resident weights + KV alone exceed {args.vram_gb:.0f}GB "
              f"VRAM — no room for an expert cache. Partial offload of the "
              f"resident layers (-ngl < max) is untested with packs; a bigger "
              f"card or a smaller quant is the honest answer.")
        return

    # 2. thrash-cliff floor.
    frac = cache_bytes / geo.paged_bytes
    floor_bytes = geo.cliff_frac * geo.paged_bytes
    marginal = frac < 1.5 * geo.cliff_frac
    if frac <= geo.cliff_frac:
        print(f"\nWARNING: the {cache_bytes / 1e9:.1f}GB cache is BELOW the "
              f"thrash cliff ({geo.cliff_frac:.1%} of experts = "
              f"{floor_bytes / 1e9:.1f}GB — one token's active set). "
              f"Steady-state hit rate will be ~0%.")
    elif marginal:
        print(f"\nNOTE: cache is only {frac / geo.cliff_frac:.1f}x the thrash "
              f"cliff ({floor_bytes / 1e9:.1f}GB) — expect the steep part "
              f"of the curve; every extra GB pays off directly.")

    # 3. family curve -> hit rate -> tok/s.
    family = args.family
    if family == "auto":
        family = ARCH_FAMILY.get(geo.arch)
    if args.trace:
        traces = {"your trace": load_trace_jsonl(Path(args.trace))}
    elif family:
        traces = {family: load_family_trace(family)}
    else:
        print(f"\nNOTE: unknown architecture {geo.arch!r} — showing all three "
              f"routing families. Collect a real trace for a tight answer:\n"
              f"  BIN=<llama-nvmoe-trace> MODEL=<pack>/resident.gguf NGL=99 \\\n"
              f"  PREFIX={geo.arch} bash tools/collect_qwen_traces.sh")
        traces = {f: load_family_trace(f) for f in FAMILIES}

    print(f"\n{'family':<12} {'hit rate':>9} {'GB/token':>9} "
          f"{'t_io':>8} {'tok/s':>7}  expected range")
    rows = []
    for fam, trace in traces.items():
        if trace is None:
            print(f"{fam:<12} trace file missing (traces/{FAMILIES.get(fam, '?')})")
            continue
        hit, tc, tio, tps = predict_tps(geo, cache_bytes, trace,
                                        args.nvme_gbps, args.gpu_eff_gbps)
        rows.append((fam, hit, tps))
        miss_gb = geo.active_bytes * (1 - hit) / 1e9
        ceiling = 1e3 / tc                      # all-resident compute ceiling
        lo, hi = tps * PRED_RANGE[0], min(tps * PRED_RANGE[1], ceiling)
        print(f"{fam:<12} {hit:>9.1%} {miss_gb:>9.3f} {tio:>7.0f}ms "
              f"{tps:>7.1f}  {lo:.0f}-{hi:.0f} tok/s")

    # 4. prefetch recommendation.
    print()
    if geo.avg_extent <= 6e6:
        print(f"prefetch            : ON (default) — {geo.avg_extent / 1e6:.1f}MB "
              f"extents are under the 6MB speculation gate; lookahead is "
              f"worth ~+5% when fetch-bound and auto-disables near-resident")
    else:
        print(f"prefetch            : OFF (automatic) — {geo.avg_extent / 1e6:.1f}MB "
              f"extents exceed the 6MB speculation gate; wasted guesses cost "
              f"more than they hide (measured on GPT-OSS-120B). "
              f"NVMOE_PREFETCH=1 forces it if your SSD is much faster.")

    # 5. the plan.
    cache_mb = int(cache_bytes / MIB) // 256 * 256
    stem = Path(geo.name).stem
    print(f"recommended cache   : NVMOE_CACHE_MB={cache_mb}")
    print(f"\ncommands:")
    print(f"  python3 tools/repack_gguf.py models/{stem}.gguf")
    print(f"  python3 tools/verify_pack.py models/{stem}.nvmoe models/{stem}.gguf")
    print(f"  # correctness gate (bit-identical logits vs stock):")
    print(f"  <bin>/llama-nvmoe-logits -m models/{stem}.gguf -o /tmp/stock.bin -n 24")
    print(f"  NVMOE_CACHE_MB={cache_mb} <bin>/llama-nvmoe-logits "
          f"-m models/{stem}.nvmoe/resident.gguf -o /tmp/pack.bin -n 24")
    print(f"  python3 tools/compare_logits.py /tmp/stock.bin /tmp/pack.bin")
    print(f"  # decode benchmark:")
    print(f"  NVMOE_CACHE_MB={cache_mb} <bin>/llama-bench "
          f"-m models/{stem}.nvmoe/resident.gguf -ngl 99 -p 512 -n 128 -r 5 -t 8")


# -------------------------------------------------------------- postdict ---

# geometry constants copied from the packs' manifest.json (reproducible from
# the public GGUFs via tools/repack_gguf.py) and measured llama-bench decode
# from runtime/README.md. (cache_mb, tok/s) per model.
POSTDICT = [
    ("qwen3-30b-a3b-q4_k_m", "qwen3",
     Geometry("qwen3-30b-a3b-q4_k_m.gguf", "qwen3moe", 48, 128, 8,
              [(2654208, 24), (3059712, 24)], int(1.00e9), 1.0),
     [(12288, 169.7), (8192, 60.2), (4096, 21.9)]),
    ("gpt-oss-120b-mxfp4", "gptoss",
     Geometry("gpt-oss-120b-mxfp4.gguf", "gpt-oss", 36, 128, 4,
              [(13221888, 36)], int(2.47e9), 1.5),
     [(11264, 24.5), (8192, 18.7), (4096, 10.5)]),
    # sync-fetch (NVMOE_LOOKAHEAD=0) rows of the qwen3-next sweep — the
    # prediction models synchronous misses; prefetch added +12-17% on top
    ("qwen3-next-80b-a3b-q4_k_m", "qwen3",
     Geometry("qwen3-next-80b-a3b-instruct-q4_k_m.gguf", "qwen3next", 48, 512, 10,
              [(1769472, 24), (2039808, 24)], int(1.70e9), 0.4),
     [(11776, 35.4), (8192, 27.3), (4096, 18.0)]),
    # A12B cautionary tale: 4.2GB of expert reads per token; the planner
    # said 2.4 before the bench said 2.80 — active params are the wall
    ("glm-4.5-air-q4_k_m", "glm",
     Geometry("glm-4.5-air-q4_k_m.gguf", "glm4moe", 45, 128, 8,
              [(11460608, 45)], int(5.0e9), 0.8),
     [(7936, 2.80), (4096, 2.29)]),
    ("deepseek-v2-lite-chat-q4_k_m", "deepseek",
     Geometry("deepseek-v2-lite-chat-q4_k_m.gguf", "deepseek2", 26, 64, 6,
              [(5226496, 14), (6307840, 12)], int(0.84e9), 0.5),
     [(4096, 32.3), (2048, 14.6)]),
]


def postdict(args) -> None:
    print("predicted vs measured decode (llama-bench, RTX 5070 Ti + 990 PRO;"
          "\nmeasured numbers and their commands: runtime/README.md)\n")
    print(f"{'model':<30} {'cache':>7} {'hit':>7} {'pred':>7} "
          f"{'measured':>9} {'meas/pred':>10}")
    worst = (1.0, 1.0)
    for name, family, geo, points in POSTDICT:
        trace = load_family_trace(family)
        if trace is None:
            print(f"{name:<30} trace missing for family {family!r}")
            continue
        for mb, measured in points:
            hit, _, _, tps = predict_tps(geo, mb * MIB, trace,
                                         args.nvme_gbps, args.gpu_eff_gbps)
            ratio = measured / tps
            worst = (min(worst[0], ratio), max(worst[1], ratio))
            print(f"{name:<30} {mb:>5}MB {hit:>7.1%} {tps:>7.1f} "
                  f"{measured:>9.1f} {ratio:>9.2f}x")
    print(f"\nspread: measured lands {worst[0]:.2f}x-{worst[1]:.2f}x of "
          f"predicted. The >1x tail is llama-bench's `-p 0` generation "
          f"routing more repetitively than the real-workload traces the "
          f"prediction uses (largest on flat-routing models); live hit "
          f"rates on real prompts match the curves to a few points.")


# ------------------------------------------------------------------ main ---

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf", nargs="?", help="path to a MoE GGUF (or use --preset)")
    parser.add_argument("--preset", choices=sorted(SIM_PRESETS),
                        help="plan a model you haven't downloaded, from sim/presets.py")
    parser.add_argument("--vram-gb", type=float, default=16.0)
    parser.add_argument("--nvme-gbps", type=float, default=7.0,
                        help="your SSD's peak read GB/s (tools/nvme_probe.py)")
    parser.add_argument("--gpu-eff-gbps", type=float, default=GPU_EFF_GBPS,
                        help="effective decode-read bandwidth; default is the "
                             "RTX 5070 Ti fit — scale by your card's memory "
                             "bandwidth relative to 896GB/s")
    parser.add_argument("--kv-gb", type=float, help="override the KV-cache estimate")
    parser.add_argument("--headroom-gb", type=float, default=DEFAULT_HEADROOM_GB)
    parser.add_argument("--family", default="auto",
                        choices=["auto"] + sorted(FAMILIES),
                        help="routing family override (auto = by architecture)")
    parser.add_argument("--trace", help="use your own trace (tokens.jsonl) as the curve")
    parser.add_argument("--postdict", action="store_true",
                        help="print predicted-vs-measured for the reference models")
    args = parser.parse_args()

    if args.postdict:
        postdict(args)
        return
    if args.preset:
        emit_plan(geometry_from_preset(args.preset), args)
        return
    if not args.gguf:
        parser.error("give a GGUF path, --preset, or --postdict")
    emit_plan(geometry_from_gguf(Path(args.gguf)), args)


if __name__ == "__main__":
    main()
