#!/bin/bash
# Collect Qwen3-30B-A3B routing traces across four workloads, post-process,
# and print stats. Run from the nvmoe repo root.
set -e
BIN=${BIN:-/workspace/projects/llama.cpp-nvmoe/build/bin/llama-nvmoe-trace}
MODEL=${MODEL:-models/qwen3-30b-a3b-q4_k_m.gguf}

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
