#!/usr/bin/env bash
# Reproducible LFM baseline runner. NPU execution is opt-in because a bad ivpu submission can
# hang the machine. Results and environment metadata are written below benchmarks/results/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-$ROOT/../llama.cpp}"
BENCH="${LLAMA_BENCH:-$LLAMA_ROOT/build/bin/llama-bench}"
MODEL="${MODEL:-}"
MODE="${MODE:-cpu}" # cpu | vulkan | npu
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/benchmarks/results/${STAMP}-${MODE}"

if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
    echo "MODEL must name an existing LFM GGUF" >&2
    exit 2
fi
if [[ ! -x "$BENCH" ]]; then
    echo "llama-bench not found at $BENCH (set LLAMA_BENCH)" >&2
    exit 2
fi
if [[ "$MODE" == npu && "${RUN_NPU:-0}" != 1 ]]; then
    echo "NPU runs require RUN_NPU=1 acknowledgement" >&2
    exit 2
fi

mkdir -p "$OUT"
{
    echo "timestamp_utc=$STAMP"
    echo "mode=$MODE"
    echo "model=$MODEL"
    echo "bench=$BENCH"
    echo "attention_placement=${ATTENTION_PLACEMENT:-qkv}"
    echo "npu_bits=${GGML_NPU_BITS:-auto}"
    echo "npu_async=${GGML_NPU_ASYNC:-0}"
    echo "prompt_tokens=${PROMPT_TOKENS:-512}"
    echo "generated_tokens=${GEN_TOKENS:-128}"
    echo "repetitions=${REPETITIONS:-5}"
    uname -a
    free -b
} > "$OUT/environment.txt"

args=(-m "$MODEL" -p "${PROMPT_TOKENS:-512}" -n "${GEN_TOKENS:-128}" -r "${REPETITIONS:-5}")
case "$MODE" in
    cpu)
        args+=(-ngl 0)
        ;;
    vulkan)
        args+=(-ngl 99)
        ;;
    npu)
        : "${GGML_BACKEND_PATH:=$ROOT/target/release/libggml_npu.so}"
        : "${GGML_NPU_OVC:=$ROOT/shim/libggml_npu_ovc.so}"
        export GGML_BACKEND_PATH GGML_NPU_OVC
        export GGML_NPU_PROFILE=1 GGML_NPU_TRACE=1
        if [[ "${ATTENTION_PLACEMENT:-qkv}" == full ]]; then
            export GGML_NPU_ATTENTION=full
        fi
        args+=(-ngl "${NGL:-0}")
        ;;
    *)
        echo "MODE must be cpu, vulkan, or npu" >&2
        exit 2
        ;;
esac

bench_pid=""
guard_pid=""
ENERGY_FILE="${ENERGY_FILE:-/sys/class/powercap/intel-rapl:0/energy_uj}"
energy_before=""
[[ -r "$ENERGY_FILE" ]] && energy_before="$(<"$ENERGY_FILE")"
cleanup() {
    [[ -n "$guard_pid" ]] && kill "$guard_pid" 2>/dev/null || true
}
trap cleanup EXIT

/usr/bin/time -v "$BENCH" "${args[@]}" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt" &
bench_pid=$!

if [[ "$MODE" == npu ]]; then
    (
        while sleep 0.2; do
            avail_kb="$(awk '/MemAvailable:/ { print $2 }' /proc/meminfo)"
            if (( avail_kb < ${MIN_AVAILABLE_KB:-400000} )); then
                echo "MemAvailable guard killed benchmark at ${avail_kb} KiB" >> "$OUT/guard.txt"
                kill -TERM "$bench_pid" 2>/dev/null || true
                exit 0
            fi
        done
    ) &
    guard_pid=$!
fi

wait "$bench_pid"
if [[ -n "$energy_before" && -r "$ENERGY_FILE" && -r "${ENERGY_FILE%/energy_uj}/max_energy_range_uj" ]]; then
    energy_after="$(<"$ENERGY_FILE")"
    energy_range="$(<"${ENERGY_FILE%/energy_uj}/max_energy_range_uj")"
    energy_delta=$((energy_after - energy_before))
    (( energy_delta < 0 )) && energy_delta=$((energy_delta + energy_range))
    echo "$energy_delta" > "$OUT/energy_uj.txt"
fi

echo "$OUT"
