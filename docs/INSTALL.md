# Installing nvmoe

Three paths. **Path A needs no compiler and takes about two minutes** —
use it unless you have a reason not to. All paths end at the same check.

What you need in all cases:

- Linux, x86_64
- Python 3.10+ (standard library only — nothing to `pip install`)
- An NVMe SSD with room for the models you want (18-65GB each)
- For GPU speed: an NVIDIA GPU (8GB VRAM minimum, 16GB recommended) with a
  current driver (`nvidia-smi` works). No CUDA toolkit needed for Path A.

---

## Path A — prebuilt binaries (recommended)

```bash
# 1. get the repo (the tools, planner, and CLI live here)
git clone https://github.com/BrianBoitano/nvmoe
cd nvmoe
```

**2.** Open the [Releases page](https://github.com/BrianBoitano/nvmoe/releases)
and download ONE tarball from the latest release:

- have an NVIDIA GPU → `nvmoe-bin-<version>-linux-x86_64-cuda.tar.gz`
- CPU only / just testing → `nvmoe-bin-<version>-linux-x86_64-cpu.tar.gz`

Or from the terminal (replace `v0.4.0` with the latest version, and
`cuda` with `cpu` if you have no GPU):

```bash
curl -fLO https://github.com/BrianBoitano/nvmoe/releases/download/v0.4.0/nvmoe-bin-v0.4.0-linux-x86_64-cuda.tar.gz
tar xzf nvmoe-bin-v0.4.0-linux-x86_64-cuda.tar.gz
mv nvmoe-bin bin        # the CLI looks for ./bin automatically

# 3. confirm
./nvmoe doctor
```

(`curl -f` fails loudly on a bad URL instead of feeding an error page to
tar.)

`doctor` should show green for python, gpu (if you have one), and runtime.
If it does, you're installed — go run a model:

```bash
./nvmoe list
./nvmoe run qwen3-30b-2507
```

The CUDA binaries cover sm_80 through sm_120 (RTX 30-series through
50-series, A100/H100) and link the CUDA runtime statically — **your NVIDIA
driver is the only requirement**. If the release download step confuses
your shell, just open the [Releases page](https://github.com/BrianBoitano/nvmoe/releases)
in a browser, download the tarball, and extract it into the repo as `bin/`.

## Path B — build from source with CUDA

For a GPU older/newer than the prebuilt arch list, or if you want to
hack on the runtime. Extra prerequisites: `git`, `cmake` ≥ 3.14, a C++17
compiler, and the CUDA toolkit 12.x (`nvcc`). No toolkit on the machine?
Run these same commands inside an `nvidia/cuda:12.8.1-devel-ubuntu24.04`
container with the repo mounted — that is exactly how the reference box
builds it.

```bash
git clone https://github.com/BrianBoitano/nvmoe
cd nvmoe

# clone llama.cpp at the pinned commit and apply the nvmoe patch series
./runtime/apply.sh

# build (15-40 min depending on machine; "native" = just your GPU's arch)
cmake -B llama.cpp-nvmoe/build-cuda -S llama.cpp-nvmoe \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
      -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
cmake --build llama.cpp-nvmoe/build-cuda -j --target \
      llama-server llama-bench llama-gguf-split llama-nvmoe-logits

./nvmoe doctor
```

## Path C — CPU only

Same as Path B without the CUDA flags (or take the `-cpu` release
tarball). Everything works — the correctness gate, the repacker, the
planner, the server — just slowly. This is the right path for testing the
pipeline on a laptop before committing your GPU box.

```bash
./runtime/apply.sh
cmake -B llama.cpp-nvmoe/build -S llama.cpp-nvmoe -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
cmake --build llama.cpp-nvmoe/build -j --target llama-server llama-nvmoe-logits llama-bench
```

---

## Verify the install (any path)

```bash
./nvmoe doctor                 # all green?
./nvmoe run olmoe-1b-7b        # 3.9GB test model: downloads, repacks,
                               # byte-verifies, serves on :8901
curl http://localhost:8901/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
./nvmoe stop
```

If that answered, the whole pipeline works on your machine.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `doctor` says "no built llama-server" | `bin/` missing or wrong place — it must sit at the repo root (or export `NVMOE_BIN_DIR=/path/to/bin`) |
| llama-server exits instantly (CUDA tarball) | no NVIDIA driver visible (`nvidia-smi` fails). Fix the driver, or use the `-cpu` tarball |
| `illegal instruction` on start (prebuilt) | very old CPU (pre-AVX2). Build from source (Path B/C) — it compiles for your machine |
| out-of-memory at model load | `NVMOE_CACHE_MB` + KV cache must fit VRAM together; lower the cache or `-c`. `python3 tools/plan.py <gguf>` prints a budget that fits |
| decode is much slower than the published number | check the SSD: the pack must live on the NVMe you measured (`python3 tools/nvme_probe.py`), and another process may be holding GPU or SSD bandwidth |
| port 8901 in use | `./nvmoe run <model> --port 8902` |

Something else: open an issue with the `./nvmoe doctor` output and the
first 30 lines of the server log.
