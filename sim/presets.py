"""Model and hardware presets for the nvmoe cache simulator.

All model numbers are derived from published architecture configs:
  - DeepSeek-R1 671B: 61 layers (58 MoE), 256 routed experts/layer, top-8 routing,
    expert FFN = 3 matrices of 7168 x 2048. Unsloth dynamic quants keep routed
    experts at ~1.58 bpw while dense/attention/shared stay at 4-6 bpw.
  - Qwen3-Next-80B-A3B: 48 layers, 512 experts/layer, top-10, expert FFN
    = 3 x 2048 x 512 (ultra-sparse: only ~3B active of 80B total).
  - GPT-OSS-120B: 36 layers, 128 experts/layer, top-4, expert FFN
    = 3 x 2880 x 2880, shipped natively in MXFP4 (~4.25 bpw).
  - Mixtral-8x7B: 32 layers, 8 experts/layer, top-2, expert FFN
    = 3 x 4096 x 14336. Small enough to collect REAL routing traces on one box,
    so it is the calibration model for the synthetic trace generator.

"always_on_gb" is the VRAM-resident floor: attention + dense layers + shared
experts + embeddings at their quant level. "kv_gb" is a working KV-cache budget
at a typical local context (R1 uses MLA so its KV is small).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    name: str
    moe_layers: int
    experts_per_layer: int
    top_k: int
    expert_params: int      # parameters in ONE routed expert
    expert_bits: float      # bpw for routed experts (the tier that gets paged)
    always_on_gb: float     # dense + attention + shared experts, resident in VRAM
    kv_gb: float            # KV cache budget at a typical local context

    @property
    def expert_bytes(self) -> int:
        return int(self.expert_params * self.expert_bits / 8)

    @property
    def total_experts(self) -> int:
        return self.moe_layers * self.experts_per_layer

    @property
    def total_expert_gb(self) -> float:
        return self.total_experts * self.expert_bytes / 1e9

    @property
    def active_experts_per_token(self) -> int:
        return self.moe_layers * self.top_k


@dataclass(frozen=True)
class HardwarePreset:
    name: str
    vram_gb: float
    nvme_gbps: float        # sustained sequential read, GB/s


MODELS = {
    "deepseek-r1-671b": ModelPreset(
        name="DeepSeek-R1 671B (dyn ~1.58-bit experts)",
        moe_layers=58, experts_per_layer=256, top_k=8,
        expert_params=3 * 7168 * 2048, expert_bits=1.58,
        always_on_gb=9.5, kv_gb=1.0,
    ),
    "qwen3-next-80b": ModelPreset(
        name="Qwen3-Next-80B-A3B (4.5-bit)",
        moe_layers=48, experts_per_layer=512, top_k=10,
        expert_params=3 * 2048 * 512, expert_bits=4.5,
        always_on_gb=2.5, kv_gb=1.0,
    ),
    "gpt-oss-120b": ModelPreset(
        name="GPT-OSS-120B (native MXFP4)",
        moe_layers=36, experts_per_layer=128, top_k=4,
        expert_params=3 * 2880 * 2880, expert_bits=4.25,
        always_on_gb=2.5, kv_gb=1.5,
    ),
    # calibration models — small enough to trace on one box (see docs/TRACE_COLLECTION.md)
    "qwen3-30b-a3b": ModelPreset(
        name="Qwen3-30B-A3B (4.5-bit, calibration model)",
        moe_layers=48, experts_per_layer=128, top_k=8,
        expert_params=3 * 2048 * 768, expert_bits=4.5,
        always_on_gb=1.5, kv_gb=1.0,
    ),
    "olmoe-7b": ModelPreset(
        name="OLMoE-1B-7B (4-bit, collector smoke-test model)",
        moe_layers=16, experts_per_layer=64, top_k=8,
        expert_params=3 * 2048 * 1024, expert_bits=4.5,
        always_on_gb=0.7, kv_gb=0.5,
    ),
    "deepseek-v2-lite": ModelPreset(
        name="DeepSeek-V2-Lite (4.8-bit, fine-grained-routing calibration model)",
        moe_layers=26, experts_per_layer=64, top_k=6,
        expert_params=3 * 2048 * 1408, expert_bits=4.8,
        always_on_gb=1.6, kv_gb=0.5,
    ),
    "mixtral-8x7b": ModelPreset(
        name="Mixtral-8x7B (4.5-bit, calibration model)",
        moe_layers=32, experts_per_layer=8, top_k=2,
        expert_params=3 * 4096 * 14336, expert_bits=4.5,
        always_on_gb=1.5, kv_gb=1.0,
    ),
}

HARDWARE = {
    "5070ti-990pro": HardwarePreset(
        name="RTX 5070 Ti 16GB + Samsung 990 PRO (PCIe 4.0 x4)",
        vram_gb=16.0, nvme_gbps=7.0,
    ),
}
