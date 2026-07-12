#!/usr/bin/env bash
# Offline calibration sweep for quant width, attention partition, and NPU/GPU layer boundary.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${MODEL:?set MODEL to an LFM GGUF}"
: "${RUN_NPU:?set RUN_NPU=1 after reviewing the NPU safety notes}"
[[ "$RUN_NPU" == 1 ]] || { echo "RUN_NPU must equal 1" >&2; exit 2; }

IFS=, read -r -a bits <<< "${BITS_CANDIDATES:-4,8,16}"
IFS=, read -r -a placements <<< "${ATTENTION_CANDIDATES:-qkv,full}"
IFS=, read -r -a layers <<< "${NGL_CANDIDATES:-0}"
IFS=, read -r -a async_modes <<< "${ASYNC_CANDIDATES:-0,1}"

for ngl in "${layers[@]}"; do
    for bit in "${bits[@]}"; do
        for placement in "${placements[@]}"; do
            for async_mode in "${async_modes[@]}"; do
                echo "calibrating ngl=$ngl bits=$bit attention=$placement async=$async_mode"
                if [[ "$async_mode" == 1 ]]; then
                    RUN_NPU=1 MODE=npu MODEL="$MODEL" NGL="$ngl" GGML_NPU_BITS="$bit" GGML_NPU_ASYNC=1 \
                        ATTENTION_PLACEMENT="$placement" "$ROOT/benchmarks/run_lfm_baseline.sh"
                else
                    env -u GGML_NPU_ASYNC RUN_NPU=1 MODE=npu MODEL="$MODEL" NGL="$ngl" \
                        GGML_NPU_BITS="$bit" ATTENTION_PLACEMENT="$placement" \
                        "$ROOT/benchmarks/run_lfm_baseline.sh"
                fi
            done
        done
    done
done

"$ROOT/benchmarks/pareto.py" "$ROOT/benchmarks/results"
