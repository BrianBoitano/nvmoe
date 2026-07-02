#!/usr/bin/env bash
# Stand up the nvmoe llama.cpp fork: clone at the pinned base commit and
# apply the patch series. nvmoe is a standalone repo + patch set; the
# patches are never submitted upstream (llama.cpp does not accept
# predominantly AI-generated PRs; private forks are exempt).
#
# usage: ./runtime/apply.sh [target-dir]     (default: ./llama.cpp-nvmoe)
set -euo pipefail

BASE_COMMIT=4fc4ec5541b243957ae5099edb67372f8f3b550e   # upstream master the series is based on
TARGET=${1:-llama.cpp-nvmoe}
HERE=$(cd "$(dirname "$0")" && pwd)

if [ ! -d "$TARGET" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$TARGET"
fi
cd "$TARGET"

git fetch origin "$BASE_COMMIT" 2>/dev/null || git fetch --unshallow origin 2>/dev/null || true
git checkout -B nvmoe "$BASE_COMMIT"
git am "$HERE"/patches/*.patch

echo
echo "nvmoe branch ready at $(git rev-parse --short HEAD). Build with:"
echo "  cmake -B build && cmake --build build -j --target llama-nvmoe-logits llama-bench"
