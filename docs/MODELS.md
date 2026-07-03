# Verified models — download one and go

Every model on this page has been repacked, byte-verified, gated
**bit-identical against stock llama.cpp**, and benchmarked on the reference
box (RTX 5070 Ti 16GB, Samsung 990 PRO, CUDA 12.8). Speeds are measured
decode, not estimates; the planner (`tools/plan.py`) reprints the receipts
with `--postdict`. For a different GPU/SSD, run the planner on the GGUF —
it reads the geometry and tells you what to expect on *your* numbers.

## The lineup (16GB-class GPUs)

| model | download (Q4_K_M unless noted) | size | serve config | measured decode |
|---|---|---|---|---|
| **Qwen3-Next-80B-A3B-Instruct** — the flagship pick | [unsloth GGUF](https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF) | 48.5GB | cache 10240MB, `-c 65536` | **44.8 tok/s** warm @11.5GB cache; ~40 at this config |
| **Qwen3-30B-A3B-Instruct-2507** — long-context workhorse | [unsloth GGUF](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) | 18.6GB | cache 6144MB, `-c 131072 -ctk q8_0 -ctv q8_0 -fa 1` | **80.5 tok/s** warm @9GB; 128k ctx fits in 14.2GB |
| **Qwen3-Coder-30B-A3B-Instruct** — coding agent | [unsloth GGUF](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | 18.6GB | same as 2507 | same geometry/speed class |
| **GPT-OSS-120B** (native MXFP4) | [3 HF splits](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) — download all 3, then `llama-gguf-split --merge` (the one model `./nvmoe run` cannot fetch for you) | 63GB | cache 11264MB, `-c 8192` | **24.5 tok/s** @11GB cache |
| Qwen3-30B-A3B (original, `./nvmoe run qwen3-30b`) | [unsloth GGUF](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF) | 18.6GB | cache 12288MB | **166 tok/s** warm @12GB — superseded by the 2507 refresh unless you want maximum speed at 32k ctx |

Small models for testing the pipeline without a big download:
OLMoE-1B-7B (3.9GB) and DeepSeek-V2-Lite (10.4GB) — note V2-Lite *fits*
in 16GB VRAM, where stock llama.cpp is ~13% faster (the planner will tell
you exactly this).

**Measured and rejected — don't bother on a 16GB card:**

| model | why not |
|---|---|
| GLM-4.5-Air 106B-A12B | 2.8 tok/s measured. 12B *active* params = 4.2GB of expert reads per token. Active size is destiny; A3B-class or bust |
| DeepSeek-R1 671B (1.58-bit) | pencils out to 1-3 tok/s: flat routing family + cache pinned at its 3.1% thrash cliff, plus visible quant degradation |
| Mixtral-class coarse MoE | 99MB experts cache terribly (simulated; the geometry argument is in the README) |
| Any dense model | physics: every weight streams every token (~0.04 tok/s) |

## The short way: the CLI

```bash
./nvmoe run qwen3-30b-2507     # does every step below, resumable, idempotent
```

`./nvmoe doctor` first if you're not sure the machine is ready. The manual
path below is what the CLI automates — useful when you want to see or
change any stage.

## Zero to chatting, copy-paste (Linux + NVIDIA)

Prereqs: git, cmake, CUDA toolkit (or run the build inside an
`nvidia/cuda:12.8+-devel` container), python3, ~25GB disk for the smallest
recommended model.

```bash
# 1. clone + build the patched llama.cpp (the CUDA compile dominates: 15-40 min.
#    Skip entirely with a prebuilt tarball: docs/INSTALL.md Path A)
git clone https://github.com/BrianBoitano/nvmoe && cd nvmoe
./runtime/apply.sh
cmake -B llama.cpp-nvmoe/build-cuda -S llama.cpp-nvmoe \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
      -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
cmake --build llama.cpp-nvmoe/build-cuda -j --target llama-server llama-nvmoe-logits llama-bench

# 2. download a verified model (the 128k-context 30B; ~18.6GB)
mkdir -p models
curl -L -o models/qwen3-30b-2507.gguf \
  "https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"

# 3. plan it on YOUR hardware (tells you cache size + expected speed)
python3 tools/plan.py models/qwen3-30b-2507.gguf --vram-gb 16 --nvme-gbps 7

# 4. repack into an expert pack + prove it lossless (byte-for-byte)
python3 tools/repack_gguf.py models/qwen3-30b-2507.gguf
python3 tools/verify_pack.py models/qwen3-30b-2507.nvmoe models/qwen3-30b-2507.gguf

# 5. serve it (OpenAI-compatible; point any chat UI at http://localhost:8901/v1)
NVMOE_CACHE_MB=6144 ./llama.cpp-nvmoe/build-cuda/bin/llama-server \
    -m models/qwen3-30b-2507.nvmoe/resident.gguf \
    -ngl 99 -c 131072 -ctk q8_0 -ctv q8_0 -fa 1 --host 0.0.0.0 --port 8901

# 6. talk to it
curl http://localhost:8901/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

Notes that save an afternoon:

- **The pack lives on the SSD you'll page from.** Put `models/` on your
  fastest NVMe; measure it honestly with `python3 tools/nvme_probe.py`.
- **`NVMOE_CACHE_MB` + KV must fit VRAM together.** The server's `--fit`
  estimate doesn't count the expert cache yet, so size `-c` and the cache
  explicitly (the planner prints a starting point).
- **HF throttles single-stream downloads** to a few MB/s; split the file
  into parallel byte ranges if you're in a hurry.
- **One model per server process**, one process per pack. Whatever is
  loaded answers all requests regardless of the requested model name.
- **First tokens after a cold start are slow** while the cache warms;
  steady-state arrives within a few dozen tokens.
- Every knob and its measured rationale: [runtime/README.md](../runtime/README.md).

## Testing a model that isn't on this page

Any **MoE GGUF with merged expert tensors** (`blk.N.ffn_{gate,up,down}_exps.weight`
— every modern llama.cpp conversion) can go through the same pipeline that
validated the models above. This is also exactly how to contribute a new
row to this page.

**Step 0 — will it even be worth it?** Before downloading 50GB, ask the
planner about the architecture class. If the model matches a preset
(`python3 tools/plan.py --preset ...` / see `sim/presets.py`), you get an
answer instantly. The two rules the measurements taught us:

- **Active params must be A3B-class** on a 16GB card. ~3B active = fast;
  12B active = 2.8 tok/s (we measured it so you don't have to).
- If the model **fits entirely in your VRAM**, use stock llama.cpp —
  paging a model that fits costs ~13%.

**Step 1 — plan it** (reads the GGUF header only, instant):

```bash
python3 tools/plan.py models/your-model.gguf --vram-gb 16 --nvme-gbps 7
```

You get the geometry, the thrash-cliff check, a hit-rate estimate from the
closest measured routing family, an expected tok/s range, and every command
below pre-filled. Unknown architecture? It shows all four family curves and
tells you how to collect the model's own trace.

**Step 2 — repack + prove it lossless:**

```bash
python3 tools/repack_gguf.py models/your-model.gguf
python3 tools/verify_pack.py models/your-model.nvmoe models/your-model.gguf
```

(Split GGUFs: merge first with `llama-gguf-split --merge part1 out.gguf`.
The repacker refuses dense models and fused `gate_up` conversions with an
explanation rather than a broken pack.)

**Step 3 — the correctness gate.** This is the step that makes a result
trustworthy: the pack must produce **bit-identical logits** to the stock
GGUF over a greedy decode:

```bash
# where your binaries live, by install path:
#   Path A (release tarball):  B=./bin
#   Path B (CUDA from source): B=./llama.cpp-nvmoe/build-cuda/bin
#   Path C (CPU from source):  B=./llama.cpp-nvmoe/build/bin
B=./bin
$B/llama-nvmoe-logits -m models/your-model.gguf              -o /tmp/stock.bin -n 16
NVMOE_CACHE_MB=4096 $B/llama-nvmoe-logits -m models/your-model.nvmoe/resident.gguf -o /tmp/pack.bin -n 16
python3 tools/compare_logits.py /tmp/stock.bin /tmp/pack.bin
# want: PASS ... -- BIT-IDENTICAL
```

Run the stock side on CPU if the stock model doesn't fit your VRAM (both
sides must be the same backend). A model too big for your RAM to gate
all-resident is fine — cap `NVMOE_CACHE_MB`; eviction is part of what the
gate exercises. If a new architecture fails the gate, that's a real
finding — open an issue with the model name and the first divergence.

**Step 4 — measure honestly:**

```bash
NVMOE_CACHE_MB=<from step 1> $B/llama-bench     -m models/your-model.nvmoe/resident.gguf -ngl 99 -p 512 -n 128 -r 5 -t 8
```

Bench at least the planner's recommended cache plus one smaller budget, and
note that `-p 0` decode flatters the cache (the README explains the
llama-bench-vs-real-workload gap — it's why the planner reports a range).

**Step 5 (optional, the most useful contribution) — trace its routing:**

`llama-nvmoe-trace` ships in the release tarballs (Path A: already in
`./bin`). Built from source? It isn't a default target — add it once:

```bash
mkdir -p llama.cpp-nvmoe/examples/nvmoe-trace
cp collector/nvmoe-trace.cpp collector/CMakeLists.txt llama.cpp-nvmoe/examples/nvmoe-trace/
sed -i 's/    add_subdirectory(simple)/    add_subdirectory(simple)\n    add_subdirectory(nvmoe-trace)/' \
    llama.cpp-nvmoe/examples/CMakeLists.txt
cmake --build llama.cpp-nvmoe/build-cuda -j --target llama-nvmoe-trace
```

Then trace through the pack (bit-identical routing at pack speed — how
models too big for CPU get traced):

```bash
BIN=$B/llama-nvmoe-trace MODEL=models/your-model.nvmoe/resident.gguf NGL=99 NVMOE_CACHE_MB=<budget> PREFIX=yourmodel bash tools/collect_qwen_traces.sh
python3 sim/trace_post.py traces/yourmodel-all.raw.jsonl --stats
```

Four workloads, ~1400 decode tokens, a few minutes on GPU. If the top-10%
traffic share and token-overlap numbers don't match any committed family
(qwen3 34.7%/43.4%, gptoss 56.7%/50.4%, glm 26.7%/37.0%, deepseek
17.7%/24.2%), you've found a fifth routing family — PR the
`traces/*-all.tokens.jsonl` and a `FAMILIES` entry in `tools/plan.py`, and
the planner gets smarter for everyone.
