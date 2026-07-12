#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$ROOT/../llama.cpp}"

cd "$ROOT"
# Hermetic claim ⊆ coverage matrix (issue #12). Cheap; no MODEL / pp512.
./tests/run_claim_coverage_check.sh

cargo test
CARGO_TARGET_DIR="$ROOT/target" cargo build --release
./shim/build.sh
./tests/run_registration_check.sh
./tests/run_graph_ir_shortconv_check.sh

# Optional guarded hardware/full-model gate. The compact graph suite above is mandatory and
# numerically checks ShortConv, recurrent state continuity, RoPE, GQA FlashAttention and sparse-MoE
# routing on the NPU.
if [[ -n "${MODEL:-}" ]]; then
    export GGML_NPU_MOE=1
    export GGML_NPU_ATTENTION=full
    export GGML_NPU_LFM_SHORTCONV=1
    for _ in $(seq 1 "${MODEL_RUNS:-2}"); do
        RUN_NPU=1 \
        LLAMA_ROOT="$LLAMA_DIR" \
        MODE=npu \
        NGL="${NGL:-0}" \
        NPU_ONLY="${NPU_ONLY:-0}" \
        NPU_DEVICE_CLASS="${NPU_DEVICE_CLASS:-npu}" \
        ATTENTION_PLACEMENT=full \
        FLASH_ATTENTION=on \
        PROMPT_TOKENS="${N_PROMPT:-16}" \
        GEN_TOKENS="${N_GEN:-4}" \
        REPETITIONS=1 \
        NPU_TIMEOUT="${NPU_TIMEOUT:-180s}" \
        "$ROOT/benchmarks/run_lfm_baseline.sh"
    done
fi
