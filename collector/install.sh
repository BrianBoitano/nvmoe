#!/bin/bash
# Build the trace collector inside a llama.cpp checkout
# usage: ./collector/install.sh /path/to/llama.cpp
set -e
LLAMA="$1"
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$LLAMA/examples/nvmoe-trace"
cp "$DIR/nvmoe-trace.cpp" "$DIR/CMakeLists.txt" "$LLAMA/examples/nvmoe-trace/"
grep -q nvmoe-trace "$LLAMA/examples/CMakeLists.txt" || sed -i "s/    add_subdirectory(simple)/    add_subdirectory(simple)\n    add_subdirectory(nvmoe-trace)/" "$LLAMA/examples/CMakeLists.txt"
cmake -B "$LLAMA/build" -S "$LLAMA" -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA/build" --target llama-nvmoe-trace -j
