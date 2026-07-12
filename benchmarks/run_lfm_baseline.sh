#!/usr/bin/env bash
# Reproducible LFM baseline runner. NPU execution is opt-in because a bad ivpu submission can
# hang the machine. Results and environment metadata are written below benchmarks/results/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-$ROOT/../llama.cpp}"
OPENVINO_DIR="${OPENVINO_DIR:-/opt/intel/openvino_2026.2.1}"
MODEL="${MODEL:-}"
MODE="${MODE:-cpu-native}" # cpu-native | cpu-openblas | cpu-flexiblas | cpu-mkl | vulkan | npu
THREADS="${THREADS:-$(nproc)}"
case "$MODE" in
    cpu|cpu-native) DEFAULT_BENCH="$LLAMA_ROOT/build-cpu-native/bin/llama-bench" ;;
    cpu-openblas) DEFAULT_BENCH="$LLAMA_ROOT/build-cpu-openblas/bin/llama-bench" ;;
    cpu-flexiblas) DEFAULT_BENCH="$LLAMA_ROOT/build-cpu-flexiblas/bin/llama-bench" ;;
    cpu-mkl) DEFAULT_BENCH="$LLAMA_ROOT/build-cpu-mkl/bin/llama-bench" ;;
    *) DEFAULT_BENCH="$LLAMA_ROOT/build-dl/bin/llama-bench" ;;
esac
BENCH="${LLAMA_BENCH:-$DEFAULT_BENCH}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/benchmarks/results/${STAMP}-${MODE}"

clear_npu_environment() {
    unset GGML_BACKEND_PATH GGML_NPU_OVC GGML_NPU_DEVICE_CLASS GGML_NPU_MEMORY_LIMIT
    unset GGML_NPU_PROFILE GGML_NPU_TRACE GGML_NPU_DEBUG GGML_NPU_ASYNC
    unset GGML_NPU_BITS GGML_NPU_ATTENTION GGML_NPU_MOE GGML_NPU_LFM_SHORTCONV
    unset GGML_NPU_FORCE_GRAPH GGML_NPU_MIN_OFFLOAD_WORK
}

if [[ "$MODE" == cpu || "$MODE" == cpu-* ]]; then
    clear_npu_environment
    : "${OMP_NUM_THREADS:=$THREADS}"
    : "${OPENBLAS_NUM_THREADS:=$THREADS}"
    : "${MKL_NUM_THREADS:=$THREADS}"
    export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS
fi

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
    echo "npu_device_class=${NPU_DEVICE_CLASS:-npu}"
    echo "npu_only=${NPU_ONLY:-0}"
    echo "npu_cache=${GGML_NPU_CACHE:-default}"
    echo "npu_timeout=${NPU_TIMEOUT:-180s}"
    echo "lfm_shortconv=${GGML_NPU_LFM_SHORTCONV:-0}"
    echo "flash_attention=${FLASH_ATTENTION:-auto}"
    echo "prompt_tokens=${PROMPT_TOKENS:-512}"
    echo "generated_tokens=${GEN_TOKENS:-128}"
    echo "repetitions=${REPETITIONS:-5}"
    echo "threads=$THREADS"
    echo "backend_git_head=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    uname -a
    free -b
} > "$OUT/environment.txt"

args=(-m "$MODEL" -p "${PROMPT_TOKENS:-512}" -n "${GEN_TOKENS:-128}" -r "${REPETITIONS:-5}"
    -fa "${FLASH_ATTENTION:-auto}" -t "$THREADS" -o jsonl)
case "$MODE" in
    cpu|cpu-native|cpu-openblas|cpu-flexiblas|cpu-mkl)
        args+=(-ngl 0)
        ;;
    vulkan)
        args+=(-ngl 99)
        ;;
    npu)
        : "${GGML_BACKEND_PATH:=$ROOT/target/release/libggml_npu.so}"
        : "${GGML_NPU_OVC:=$ROOT/shim/libggml_npu_ovc.so}"
        : "${GGML_NPU_DEVICE_CLASS:=${NPU_DEVICE_CLASS:-npu}}"
        export GGML_BACKEND_PATH GGML_NPU_OVC GGML_NPU_DEVICE_CLASS
        export LD_LIBRARY_PATH="$OPENVINO_DIR/runtime/lib/intel64:${LD_LIBRARY_PATH:-}"
        [[ "${NPU_PROFILE:-1}" == 1 ]] && export GGML_NPU_PROFILE=1
        [[ "${NPU_TRACE:-0}" == 1 ]] && export GGML_NPU_TRACE=1
        if [[ "${ATTENTION_PLACEMENT:-qkv}" == full ]]; then
            export GGML_NPU_ATTENTION=full
        fi
        if [[ "${NPU_ONLY:-0}" == 1 ]]; then
            : "${GGML_NPU_MEMORY_LIMIT:=8589934592}"
            export GGML_NPU_MEMORY_LIMIT
            args+=(-ngl 999 -dev NPU -sm none)
        else
            args+=(-ngl "${NGL:-0}")
        fi
        ;;
    *)
        echo "MODE must be cpu-native, cpu-openblas, cpu-flexiblas, cpu-mkl, vulkan, or npu" >&2
        exit 2
        ;;
esac

if [[ "$MODE" == npu && -f "$GGML_BACKEND_PATH" ]]; then
    echo "backend_so_sha256=$(sha256sum "$GGML_BACKEND_PATH" | awk '{print $1}')" >> "$OUT/environment.txt"
fi

bench_pid=""
guard_pid=""
ENERGY_FILE="${ENERGY_FILE:-/sys/class/powercap/intel-rapl:0/energy_uj}"
energy_before=""
[[ -r "$ENERGY_FILE" ]] && energy_before="$(<"$ENERGY_FILE")"
cleanup() {
    [[ -n "$guard_pid" ]] && kill "$guard_pid" 2>/dev/null || true
    if [[ -n "$bench_pid" ]] && kill -0 "$bench_pid" 2>/dev/null; then
        kill -TERM -- "-$bench_pid" 2>/dev/null || true
        sleep 1
        kill -KILL -- "-$bench_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

runner=()
if [[ "$MODE" == npu ]]; then
    runner=(timeout --signal=TERM --kill-after=10s "${NPU_TIMEOUT:-180s}")
fi
# One process group makes timeout, memory-guard, interrupt, and shell-exit cleanup reach every
# descendant (the old wrapper-only kill could leave llama-bench running after its parent exited).
setsid /usr/bin/time -f \
    '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M,"exit_status":%x}' \
    -o "$OUT/process.json" "${runner[@]}" "$BENCH" "${args[@]}" \
    > "$OUT/result.jsonl" 2> "$OUT/stderr.txt" &
bench_pid=$!

if [[ "$MODE" == npu ]]; then
    (
        while sleep 0.2; do
            avail_kb="$(awk '/MemAvailable:/ { print $2 }' /proc/meminfo)"
            if (( avail_kb < ${MIN_AVAILABLE_KB:-4194304} )); then
                echo "MemAvailable guard killed benchmark at ${avail_kb} KiB" >> "$OUT/guard.txt"
                kill -TERM -- "-$bench_pid" 2>/dev/null || true
                sleep 1
                kill -KILL -- "-$bench_pid" 2>/dev/null || true
                exit 0
            fi
        done
    ) &
    guard_pid=$!
fi

set +e
wait "$bench_pid"
status=$?
set -e
bench_pid=""
if [[ -n "$energy_before" && -r "$ENERGY_FILE" && -r "${ENERGY_FILE%/energy_uj}/max_energy_range_uj" ]]; then
    energy_after="$(<"$ENERGY_FILE")"
    energy_range="$(<"${ENERGY_FILE%/energy_uj}/max_energy_range_uj")"
    energy_delta=$((energy_after - energy_before))
    (( energy_delta < 0 )) && energy_delta=$((energy_delta + energy_range))
    printf '{"energy_uj":%s}\n' "$energy_delta" > "$OUT/energy.json"
fi

echo "$OUT"
exit "$status"
