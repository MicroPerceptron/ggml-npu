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

# Feature-detect host NPU device typing (issue #13). Enum members are not preprocessor
# macros, so probe the header text the same way build.rs does.
NPU_DEFS=()
if grep -q 'GGML_BACKEND_DEVICE_TYPE_NPU' "$GGML/ggml/include/ggml-backend.h"; then
    NPU_DEFS+=(-DGGML_NPU_HEADERS_HAVE_NPU=1)
    echo "registration_check: host headers declare GGML_BACKEND_DEVICE_TYPE_NPU"
else
    echo "registration_check: host headers lack NPU device type — expecting ACCEL default"
fi

# -rdynamic exports our stub ggml symbol so the dlopened .so (RTLD_NOW) can resolve it,
# exactly as a real host exports ggml's symbols to loaded backends.
"$CC" -rdynamic -Wall -I"$GGML/ggml/include" -I"$GGML/ggml/src" "${NPU_DEFS[@]}" \
    "$CRATE/tests/registration_check.c" -o "$OUTDIR/registration_check" -ldl

env -u GGML_NPU_DEVICE_CLASS "$OUTDIR/registration_check" "$SO"
for cls in accel gpu igpu npu; do
    GGML_NPU_DEVICE_CLASS="$cls" "$OUTDIR/registration_check" "$SO"
done
