#!/bin/bash
# Collect Qwen3-30B-A3B routing traces across four workloads, post-process,
# and print stats. Run from the nvmoe repo root:
#
#   BIN=/path/to/llama.cpp/build/bin/llama-nvmoe-trace \
#   MODEL=/path/to/Qwen3-30B-A3B-Q4_K_M.gguf \
#   bash tools/collect_qwen_traces.sh
#
# BIN is the collector binary built by collector/install.sh.
# Works with any MoE GGUF — smaller models (OLMoE, Qwen3-30B-A3B,
# DeepSeek-V2-Lite) trace fine on CPU; no GPU required.
set -e
BIN=${BIN:-llama-nvmoe-trace}
MODEL=${MODEL:?set MODEL=/path/to/model.gguf}

command -v "$BIN" >/dev/null || [ -x "$BIN" ] || {
    echo "collector binary not found: $BIN"
    echo "build it with: ./collector/install.sh /path/to/llama.cpp"
    exit 1
}
[ -f "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; }

mkdir -p traces
for w in chat code story summarize; do
    n=400
    [ "$w" = summarize ] && n=200
    echo "=== workload: $w (n=$n) ==="
    "$BIN" -m "$MODEL" -f "prompts/$w.txt" -o "traces/qwen3-$w.raw.jsonl" -n $n 2>&1 | tail -1
done

cat traces/qwen3-chat.raw.jsonl traces/qwen3-code.raw.jsonl \
    traces/qwen3-story.raw.jsonl traces/qwen3-summarize.raw.jsonl \
    > traces/qwen3-all.raw.jsonl

python3 sim/trace_post.py traces/qwen3-all.raw.jsonl --stats
python3 sim/trace_post.py traces/qwen3-all.raw.jsonl --decode-only -o traces/qwen3-all.tokens.jsonl
