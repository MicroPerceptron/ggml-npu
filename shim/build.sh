#!/usr/bin/env bash
# Build the OpenVINO compile shim -> libggml_npu_ovc.so.
#
# The Rust backend dlopens this on a cache miss to compile native NPU blobs at runtime.
# Only needed on Intel hosts. CMake discovers OpenVINO from its standard package paths,
# CMAKE_PREFIX_PATH, or an optional OPENVINO_DIR installation/configuration directory.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${GGML_NPU_SHIM_BUILD_DIR:-$DIR/build}"
cmake_args=(-S "$DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release)

if [[ -n "${OPENVINO_DIR:-}" ]]; then
    config_dir=""
    for candidate in "$OPENVINO_DIR" "$OPENVINO_DIR/runtime/cmake"; do
        if [[ -f "$candidate/OpenVINOConfig.cmake" ]]; then
            config_dir="$candidate"
            break
        fi
    done
    if [[ -z "$config_dir" ]]; then
        echo "error: OpenVINOConfig.cmake not found below OPENVINO_DIR=$OPENVINO_DIR" >&2
        exit 1
    fi
    cmake_args+=("-DOpenVINO_DIR=$config_dir")
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --config Release

echo "built $DIR/libggml_npu_ovc.so"
