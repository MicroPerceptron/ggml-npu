#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS="${1:?usage: run-matrix.sh CORPUS_DIR OUTPUT_DIR}"
OUTPUT="${2:?usage: run-matrix.sh CORPUS_DIR OUTPUT_DIR}"
mkdir -p "${OUTPUT}"

ARTIFACTS=(
  "${CORPUS}/ir2-2fc11a4f22b54cfc-n3.blob"
  "${CORPUS}/ir2-25db148dace09539-n8.blob"
  "${CORPUS}/ir2-082d22694112e163-n8.blob"
  "${CORPUS}/ir2-27d892a231afd50b-n8.blob"
  "${CORPUS}/ir2-00ca97ae85b4313b-n3.blob"
)

RUSTC_VERSION="$(rustc -V)"
CARGO_VERSION="$(cargo -V)"
CC_VERSION="$(gcc -dumpfullversion)"
TARGET="$(rustc -vV | awk '/^host:/ { print $2 }')"

run_variant() {
  local variant="$1"
  local features="$2"
  local rustflags="$3"
  local candidates="$4"
  local execution_mode="${5:-parallel-shared}"
  local lock_flag="${6:--s}"
  local lock_mode="shared"
  if [[ "${lock_flag}" == "-x" ]]; then
    lock_mode="exclusive"
  fi
  local target_dir="${ROOT}/target/matrix-${variant}"
  CARGO_TARGET_DIR="${target_dir}" RUSTFLAGS="${rustflags}" CFLAGS="" \
    cargo build --manifest-path "${ROOT}/Cargo.toml" --release \
      --no-default-features --features "${features}"
  GGML_NPU_BENCH_VARIANT="${variant}" \
  GGML_NPU_BENCH_RUSTC="${RUSTC_VERSION}" \
  GGML_NPU_BENCH_CARGO="${CARGO_VERSION}" \
  GGML_NPU_BENCH_CC="gcc ${CC_VERSION}" \
  GGML_NPU_BENCH_TARGET="${TARGET}" \
  GGML_NPU_BENCH_RUSTFLAGS="${rustflags}" \
  GGML_NPU_BENCH_CFLAGS="" \
  GGML_NPU_BENCH_AFFINITY="2-3" \
  GGML_NPU_BENCH_EXECUTION_MODE="${execution_mode}" \
  GGML_NPU_BENCH_LOCK="/tmp/ggml-npu-bench-global.lock" \
  GGML_NPU_BENCH_LOCK_MODE="${lock_mode}" \
  GGML_NPU_CODEC_CANDIDATES="${candidates}" \
    flock "${lock_flag}" /tmp/ggml-npu-bench-global.lock \
      taskset -c 2-3 "${target_dir}/release/ggml-npu-cache-codec-bench" \
        --suite "${OUTPUT}/${variant}.json" "${ARTIFACTS[@]}"
}

PORTABLE_FEATURES="bundled-zstd,structured-simd,zrip-simd,lz4-safe"
run_variant \
  "portable-accelerated" \
  "${PORTABLE_FEATURES}" \
  "" \
  "zstd-system--3,zstd-system-1,zstd-bundled--3,zstd-bundled-1,structured-zstd--3,structured-zstd-1,zrip--3,zrip-1,lz4-flex-reference"
run_variant \
  "portable-decode-scalar" \
  "bundled-zstd,lz4-safe" \
  "" \
  "structured-zstd--3,structured-zstd-1,zrip--3,zrip-1"
run_variant \
  "system-provider-only" \
  "structured-simd,zrip-simd,lz4-safe" \
  "" \
  "zstd-system--3,zstd-system-1"
run_variant \
  "lz4-unsafe" \
  "bundled-zstd,structured-simd,zrip-simd" \
  "" \
  "lz4-flex-reference"
run_variant \
  "zstd-no-asm" \
  "${PORTABLE_FEATURES},zstd-no-asm" \
  "" \
  "zstd-bundled--3,zstd-bundled-1"
run_variant \
  "native-upper-bound" \
  "${PORTABLE_FEATURES}" \
  "-C target-cpu=native" \
  "structured-zstd--3,structured-zstd-1,zrip--3,zrip-1,lz4-flex-reference"
run_variant \
  "authoritative-zstd-provider" \
  "${PORTABLE_FEATURES}" \
  "" \
  "zstd-system-1,zstd-bundled-1" \
  "sequential-authoritative" \
  "-x"
