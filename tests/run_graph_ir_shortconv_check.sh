#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGML="${GGML_DIR:-$ROOT/../llama.cpp}"
SHIM="${GGML_NPU_OVC:-$ROOT/shim/libggml_npu_ovc.so}"
BACKEND="${1:-$ROOT/target/release/libggml_npu.so}"
OV="${OPENVINO_DIR:-/opt/intel/openvino_2026.2.1}"

[[ -f "$SHIM" ]] || { echo "missing $SHIM; run shim/build.sh" >&2; exit 2; }
[[ -f "$BACKEND" ]] || { echo "missing $BACKEND; run cargo build --release" >&2; exit 2; }

OUTDIR="$(mktemp -d)"
trap 'rm -rf "$OUTDIR"' EXIT
"${CC:-cc}" -rdynamic -Wall -I"$GGML/ggml/include" -I"$GGML/ggml/src" \
    "$ROOT/tests/graph_ir_shortconv_check.c" -o "$OUTDIR/check" -ldl -lm

export LD_LIBRARY_PATH="$OV/runtime/lib/intel64:${LD_LIBRARY_PATH:-}"
exec "$OUTDIR/check" "$SHIM" "$BACKEND"
