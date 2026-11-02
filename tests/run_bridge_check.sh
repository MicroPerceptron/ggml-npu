#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGML_DIR="${GGML_DIR:-$ROOT/../llama.cpp}"
GGML_LIB_DIR="${GGML_LIB_DIR:-$GGML_DIR/build/bin}"
PROVIDER="${GGML_NPU_TEST_PROVIDER:-VIRTIO_VULKAN}"
case "$(uname -s)" in
    Darwin) EXT=dylib ;;
    Linux) EXT=so ;;
    *) echo 'Run bridge_check.cpp with your platform compiler on this host.' >&2; exit 1 ;;
esac
PLUGIN="${1:-$ROOT/target/release/libggml_npu.$EXT}"
CPU="${GGML_CPU_MODULE:-$GGML_LIB_DIR/libggml-cpu.so}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
    -I"$GGML_DIR/ggml/include" "$ROOT/tests/bridge_check.cpp" \
    -L"$GGML_LIB_DIR" -lggml -lggml-base -Wl,-rpath,"$GGML_LIB_DIR" \
    -o "$OUT/bridge_check"
"$OUT/bridge_check" "$PLUGIN" "$CPU" "$PROVIDER"
