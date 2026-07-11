#!/usr/bin/env bash
# Registration gate: compile the self-contained loader harness and run it against the
# built cdylib. Mimics ggml's dynamic-backend loader without building llama.cpp.
#
# Usage: ./tests/run_registration_check.sh [path/to/libggml_npu.so]
set -euo pipefail

CRATE="$(cd "$(dirname "$0")/.." && pwd)"
GGML="${GGML_DIR:-$CRATE/../llama.cpp}"
SO="${1:-$CRATE/target/release/libggml_npu.so}"

if [[ ! -f "$SO" ]]; then
    echo "error: $SO not found — run 'cargo build --release' first" >&2
    exit 1
fi
if [[ ! -f "$GGML/ggml/include/ggml-backend.h" ]]; then
    echo "error: ggml headers not found under $GGML — set GGML_DIR" >&2
    exit 1
fi

OUTDIR="$(mktemp -d)"
trap 'rm -rf "$OUTDIR"' EXIT
CC="${CC:-cc}"

# -rdynamic exports our stub ggml symbol so the dlopened .so (RTLD_NOW) can resolve it,
# exactly as a real host exports ggml's symbols to loaded backends.
"$CC" -rdynamic -Wall -I"$GGML/ggml/include" -I"$GGML/ggml/src" \
    "$CRATE/tests/registration_check.c" -o "$OUTDIR/registration_check" -ldl

exec "$OUTDIR/registration_check" "$SO"
