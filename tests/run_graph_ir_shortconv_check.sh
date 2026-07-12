#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGML="${GGML_DIR:-$ROOT/../llama.cpp}"
SHIM="${GGML_NPU_OVC:-$ROOT/shim/libggml_npu_ovc.so}"
BACKEND="${1:-$ROOT/target/release/libggml_npu.so}"

[[ -f "$SHIM" ]] || { echo "missing $SHIM; run shim/build.sh" >&2; exit 2; }
[[ -f "$BACKEND" ]] || { echo "missing $BACKEND; run cargo build --release" >&2; exit 2; }

OUTDIR="$(mktemp -d)"
trap 'rm -rf "$OUTDIR"' EXIT
"${CC:-cc}" -rdynamic -Wall -I"$GGML/ggml/include" -I"$GGML/ggml/src" \
    "$ROOT/tests/graph_ir_shortconv_check.c" -o "$OUTDIR/check" -ldl -lm

runtime_lib="${OPENVINO_RUNTIME_LIB_DIR:-}"
if [[ -z "$runtime_lib" && -n "${OPENVINO_DIR:-}" ]]; then
    for candidate in \
        "$OPENVINO_DIR/runtime/lib/intel64" \
        "$OPENVINO_DIR/runtime/lib" \
        "$OPENVINO_DIR/lib64" \
        "$OPENVINO_DIR/lib"; do
        if [[ -d "$candidate" ]]; then
            runtime_lib="$candidate"
            break
        fi
    done
fi
if [[ -n "$runtime_lib" ]]; then
    [[ -d "$runtime_lib" ]] || {
        echo "OpenVINO runtime library directory not found at $runtime_lib" >&2
        exit 2
    }
    export LD_LIBRARY_PATH="$runtime_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$OUTDIR/check" "$SHIM" "$BACKEND"
