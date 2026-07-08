#!/usr/bin/env bash
# Build the OpenVINO compile shim -> libggml_npu_ovc.so.
#
# The Rust backend dlopens this on a cache miss to compile native NPU blobs at runtime.
# Only needed on Intel hosts. Override the OpenVINO location with OPENVINO_DIR.
set -euo pipefail

OV="${OPENVINO_DIR:-/opt/intel/openvino_2026.2.1}"
DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -d "$OV/runtime/include" ]]; then
    echo "error: OpenVINO not found at $OV (set OPENVINO_DIR)" >&2
    exit 1
fi

g++ -std=c++17 -O2 -shared -fPIC \
    -I"$OV/runtime/include" \
    "$DIR/ov_compile.cpp" \
    -o "$DIR/libggml_npu_ovc.so" \
    -L"$OV/runtime/lib/intel64" -lopenvino

echo "built $DIR/libggml_npu_ovc.so"
