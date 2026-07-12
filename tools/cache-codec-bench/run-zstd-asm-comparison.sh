#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS="${1:?usage: run-zstd-asm-comparison.sh CORPUS_DIR OUTPUT_DIR}"
OUTPUT="${2:?usage: run-zstd-asm-comparison.sh CORPUS_DIR OUTPUT_DIR}"
mkdir -p "${OUTPUT}"

ARTIFACTS=(
  "${CORPUS}/ir2-2fc11a4f22b54cfc-n3.blob"
  "${CORPUS}/ir2-25db148dace09539-n8.blob"
  "${CORPUS}/ir2-082d22694112e163-n8.blob"
  "${CORPUS}/ir2-27d892a231afd50b-n8.blob"
  "${CORPUS}/ir2-00ca97ae85b4313b-n3.blob"
)

FEATURES="bundled-zstd,structured-simd,zrip-simd,lz4-safe"
ASM_TARGET="${ROOT}/target/matrix-portable-accelerated"
NO_ASM_TARGET="${ROOT}/target/matrix-zstd-no-asm"
CARGO_TARGET_DIR="${ASM_TARGET}" cargo build --manifest-path "${ROOT}/Cargo.toml" \
  --release --no-default-features --features "${FEATURES}"
CARGO_TARGET_DIR="${NO_ASM_TARGET}" cargo build --manifest-path "${ROOT}/Cargo.toml" \
  --release --no-default-features --features "${FEATURES},zstd-no-asm"

RUSTC_VERSION="$(rustc -V)"
CARGO_VERSION="$(cargo -V)"
CC_VERSION="$(gcc -dumpfullversion)"
TARGET="$(rustc -vV | awk '/^host:/ { print $2 }')"

run_binary() {
  local variant="$1"
  local executable="$2"
  local output="$3"
  GGML_NPU_BENCH_VARIANT="${variant}" \
  GGML_NPU_BENCH_RUSTC="${RUSTC_VERSION}" \
  GGML_NPU_BENCH_CARGO="${CARGO_VERSION}" \
  GGML_NPU_BENCH_CC="gcc ${CC_VERSION}" \
  GGML_NPU_BENCH_TARGET="${TARGET}" \
  GGML_NPU_BENCH_RUSTFLAGS="" \
  GGML_NPU_BENCH_CFLAGS="" \
  GGML_NPU_BENCH_AFFINITY="2-3" \
  GGML_NPU_BENCH_EXECUTION_MODE="sequential-authoritative-pair" \
  GGML_NPU_BENCH_LOCK="/tmp/ggml-npu-bench-global.lock" \
  GGML_NPU_BENCH_LOCK_MODE="exclusive" \
  GGML_NPU_CODEC_CANDIDATES="zstd-bundled-1" \
    taskset -c 2-3 "${executable}" --suite "${output}" "${ARTIFACTS[@]}"
}

exec 9>/tmp/ggml-npu-bench-global.lock
flock -x 9
run_binary \
  "authoritative-zstd-asm" \
  "${ASM_TARGET}/release/ggml-npu-cache-codec-bench" \
  "${OUTPUT}/authoritative-zstd-asm.json"
run_binary \
  "authoritative-zstd-no-asm" \
  "${NO_ASM_TARGET}/release/ggml-npu-cache-codec-bench" \
  "${OUTPUT}/authoritative-zstd-no-asm.json"
