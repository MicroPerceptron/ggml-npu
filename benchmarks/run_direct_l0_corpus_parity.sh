#!/usr/bin/env bash
# Lab gate: Direct ↔ Level Zero corpus binding + numeric parity (#63).
# Prefer this over full pp512 for retirement evidence. Does not delete L0.
#
# Compared per manifest entry (tests/direct_l0_corpus/manifest.csv):
#   - SHA-256 + byte size identity
#   - Direct ELF/HPI structural fingerprint vs golden (bindings, relocs, loader/NNRT,
#     resident/placement accounting)
#   - Direct binding byte sizes vs Level Zero graph argument properties
#   - For mode=numeric_fp32: Direct vs L0 FP32 output tolerances (1e-5 abs/rel)
#
# Usage:
#   export GGML_DIR=/path/to/llama.cpp
#   export GGML_NPU_CORPUS_DIR=/path/to/dir   # must contain all manifest artifacts
#   # optional: also place ir2-8c91da… in ~/.cache/ggml-npu/
#   ./benchmarks/run_direct_l0_corpus_parity.sh
#
# THREADS=6 is the bench-machine default for LFM harnesses; this gate does not invoke
# llama-bench, so THREADS is unused here (documented for lab consistency).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${GGML_DIR:?set GGML_DIR to your llama.cpp checkout}"
: "${THREADS:=6}" # documented lab default; unused by this artifact gate
export THREADS

MANIFEST="tests/direct_l0_corpus/manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "FAIL: missing $MANIFEST" >&2
  exit 1
fi

if [[ -z "${GGML_NPU_CORPUS_DIR:-}" ]]; then
  for candidate in \
    "benchmarks/results/lfm-pp512-cache" \
    "${HOME}/.cache/ggml-npu" \
    "../../benchmarks/results/lfm-pp512-cache"
  do
    if [[ -d "$candidate" ]]; then
      export GGML_NPU_CORPUS_DIR="$(cd "$candidate" && pwd)"
      break
    fi
  done
fi

if [[ -z "${GGML_NPU_CORPUS_DIR:-}" ]]; then
  echo "FAIL: set GGML_NPU_CORPUS_DIR to a directory containing corpus blobs" >&2
  exit 1
fi

echo "GGML_NPU_CORPUS_DIR=$GGML_NPU_CORPUS_DIR THREADS=$THREADS"

missing=0
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* || "$line" == id,* ]] && continue
  artifact=$(printf '%s\n' "$line" | cut -d, -f2)
  if [[ ! -f "$GGML_NPU_CORPUS_DIR/$artifact" && ! -f "${HOME}/.cache/ggml-npu/$artifact" ]]; then
    echo "MISSING $artifact"
    missing=$((missing + 1))
  fi
done <"$MANIFEST"
if [[ "$missing" -ne 0 ]]; then
  echo "FAIL: $missing corpus artifacts missing under CORPUS_DIR / ~/.cache/ggml-npu" >&2
  exit 1
fi

echo "== hermetic structural goldens (require all present) =="
GGML_NPU_CORPUS_REQUIRE_ALL=1 cargo test --lib --features direct-ivpu \
  vendor::ivpu::direct::corpus::tests:: \
  -- --nocapture
GGML_NPU_CORPUS_REQUIRE_ALL=1 cargo test --lib --features direct-ivpu \
  vendor::ivpu::direct::oracle::tests:: \
  -- --nocapture

echo "== lab Direct↔L0 binding + numeric parity (ignored hardware) =="
cargo test --lib --features bench-hardware \
  vendor::ivpu::direct::corpus::tests::corpus_direct_level_zero_parity \
  -- --ignored --exact --nocapture --test-threads=1

echo "PASS: Direct↔L0 corpus parity gate"
