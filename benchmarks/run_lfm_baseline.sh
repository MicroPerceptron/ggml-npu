#!/usr/bin/env bash
# Reproducible LFM baseline runner. NPU execution is opt-in because a bad ivpu submission can
# hang the machine. Results and environment metadata are written below benchmarks/results/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-$ROOT/../llama.cpp}"
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
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/benchmarks/results}"
OUT="${BENCHMARK_OUTPUT_DIR:-$RESULTS_ROOT/${STAMP}-${MODE}}"
TIME_BIN="${TIME_BIN:-$(type -P time || true)}"

path_label() {
    local value="${1%/}"
    printf '%s\n' "${value##*/}"
}

configure_openvino_runtime_path() {
    local runtime_lib="${OPENVINO_RUNTIME_LIB_DIR:-}"
    if [[ -z "$runtime_lib" && -n "${OPENVINO_DIR:-}" ]]; then
        local candidate
        for candidate in \
            "$OPENVINO_DIR/runtime/lib/intel64" \
            "$OPENVINO_DIR/runtime/lib" \
            "$OPENVINO_DIR/lib64" \
            "$OPENVINO_DIR/lib"; do
            if [[ -d "$candidate" ]]; then
                runtime_lib="$candidate"
                break
            fi
        done
    fi
    if [[ -n "$runtime_lib" ]]; then
        [[ -d "$runtime_lib" ]] || {
            echo "OpenVINO runtime library directory not found at $runtime_lib" >&2
            return 1
        }
        export LD_LIBRARY_PATH="$runtime_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
}

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
if [[ -z "$TIME_BIN" || ! -x "$TIME_BIN" ]]; then
    echo "GNU time executable not found (set TIME_BIN)" >&2
    exit 2
fi
if [[ "$MODE" == npu && "${RUN_NPU:-0}" != 1 ]]; then
    echo "NPU runs require RUN_NPU=1 acknowledgement" >&2
    exit 2
fi

npu_cache_label=default
if [[ -n "${GGML_NPU_CACHE:-}" ]]; then
    npu_cache_label="$(path_label "$GGML_NPU_CACHE")"
fi
cpu_affinity="$(taskset -pc $$ 2>/dev/null || echo unavailable)"
load_average_start="$(< /proc/loadavg)"
memory_pressure_start="$(tr '\n' ';' < /proc/pressure/memory 2>/dev/null || echo unavailable)"
mkdir -p "$OUT"
{
    echo "timestamp_utc=$STAMP"
    echo "mode=$MODE"
    echo "model=$(path_label "$MODEL")"
    echo "bench=$(path_label "$BENCH")"
    echo "attention_placement=${ATTENTION_PLACEMENT:-qkv}"
    echo "npu_bits=${GGML_NPU_BITS:-auto}"
    echo "npu_async=${GGML_NPU_ASYNC:-0}"
    echo "npu_device_class=${NPU_DEVICE_CLASS:-npu}"
    echo "npu_only=${NPU_ONLY:-0}"
    echo "npu_cache=$npu_cache_label"
    echo "npu_timeout=${NPU_TIMEOUT:-180s}"
    echo "lfm_shortconv=${GGML_NPU_LFM_SHORTCONV:-0}"
    echo "npu_moe=${GGML_NPU_MOE:-0}"
    echo "npu_debug=${GGML_NPU_DEBUG:-0}"
    echo "npu_trace=${GGML_NPU_TRACE:-0}"
    echo "flash_attention=${FLASH_ATTENTION:-auto}"
    echo "prompt_tokens=${PROMPT_TOKENS:-512}"
    echo "generated_tokens=${GEN_TOKENS:-128}"
    echo "repetitions=${REPETITIONS:-5}"
    echo "threads=$THREADS"
    echo "benchmark_lock=${GGML_NPU_BENCH_LOCK:-none}"
    echo "cpu_affinity=$cpu_affinity"
    echo "load_average_start=$load_average_start"
    echo "memory_pressure_start=$memory_pressure_start"
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
        configure_openvino_runtime_path
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
    [[ -f "$GGML_NPU_OVC" ]] &&
        echo "compiler_shim_sha256=$(sha256sum "$GGML_NPU_OVC" | awk '{print $1}')" >> "$OUT/environment.txt"
fi
echo "llama_bench_sha256=$(sha256sum "$BENCH" | awk '{print $1}')" >> "$OUT/environment.txt"

bench_pid=""
guard_pid=""
monitor_pid=""
ENERGY_FILE="${ENERGY_FILE:-/sys/class/powercap/intel-rapl:0/energy_uj}"
energy_before=""
[[ -r "$ENERGY_FILE" ]] && energy_before="$(<"$ENERGY_FILE")"
cleanup() {
    [[ -n "$guard_pid" ]] && kill "$guard_pid" 2>/dev/null || true
    [[ -n "$monitor_pid" ]] && kill "$monitor_pid" 2>/dev/null || true
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
setsid "$TIME_BIN" -f \
    '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M,"exit_status":%x}' \
    -o "$OUT/process.json" "${runner[@]}" "$BENCH" "${args[@]}" \
    > "$OUT/result.jsonl" 2> "$OUT/stderr.txt" &
bench_pid=$!

(
    printf 'monotonic_seconds\tllama_bench_rss_kib\tmem_available_kib\tswap_used_kib\tnpu_memory_bytes\tmemory_psi_some_total\tmemory_psi_full_total\n'
    while kill -0 "$bench_pid" 2>/dev/null; do
        read -r mem_available swap_total swap_free < <(
            awk '
                /^MemAvailable:/ { available = $2 }
                /^SwapTotal:/ { total = $2 }
                /^SwapFree:/ { free = $2 }
                END { print available, total, free }
            ' /proc/meminfo
        )
        bench_rss="$(ps -e -o comm=,rss= | awk '$1 == "llama-bench" { total += $2 } END { print total + 0 }')"
        npu_memory=0
        [[ -r /sys/class/accel/accel0/device/npu_memory_utilization ]] &&
            read -r npu_memory < /sys/class/accel/accel0/device/npu_memory_utilization
        read -r psi_some psi_full < <(
            awk '
                $1 == "some" { for (i = 2; i <= NF; i++) if ($i ~ /^total=/) { split($i, a, "="); some = a[2] } }
                $1 == "full" { for (i = 2; i <= NF; i++) if ($i ~ /^total=/) { split($i, a, "="); full = a[2] } }
                END { print some + 0, full + 0 }
            ' /proc/pressure/memory
        )
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(awk '{ print $1 }' /proc/uptime)" "$bench_rss" "$mem_available" \
            "$((swap_total - swap_free))" "$npu_memory" "$psi_some" "$psi_full"
        sleep "${MONITOR_INTERVAL:-0.2}"
    done
) > "$OUT/resources.tsv" &
monitor_pid=$!

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
[[ -n "$guard_pid" ]] && kill "$guard_pid" 2>/dev/null || true
[[ -n "$monitor_pid" ]] && kill "$monitor_pid" 2>/dev/null || true
[[ -n "$guard_pid" ]] && wait "$guard_pid" 2>/dev/null || true
[[ -n "$monitor_pid" ]] && wait "$monitor_pid" 2>/dev/null || true
guard_pid=""
monitor_pid=""
{
    echo "load_average_end=$(< /proc/loadavg)"
    echo "memory_pressure_end=$(tr '\n' ';' < /proc/pressure/memory 2>/dev/null || echo unavailable)"
} >> "$OUT/environment.txt"
awk -F '\t' '
    NR > 1 {
        if ($2 > rss) rss = $2
        if (min_available == 0 || $3 < min_available) min_available = $3
        if ($4 > swap) swap = $4
        if ($5 > npu) npu = $5
        if (first_some == "") {
            first_some = $6
            first_full = $7
        }
        last_some = $6
        last_full = $7
    }
    END {
        printf "{\"peak_llama_bench_rss_kib\":%d,\"min_mem_available_kib\":%d,\"peak_swap_used_kib\":%d,\"peak_npu_memory_bytes\":%d,\"memory_psi_some_delta_us\":%d,\"memory_psi_full_delta_us\":%d}\n",
            rss, min_available, swap, npu, last_some - first_some, last_full - first_full
    }
' "$OUT/resources.tsv" > "$OUT/resources.json"
if [[ -n "$energy_before" && -r "$ENERGY_FILE" && -r "${ENERGY_FILE%/energy_uj}/max_energy_range_uj" ]]; then
    energy_after="$(<"$ENERGY_FILE")"
    energy_range="$(<"${ENERGY_FILE%/energy_uj}/max_energy_range_uj")"
    energy_delta=$((energy_after - energy_before))
    (( energy_delta < 0 )) && energy_delta=$((energy_delta + energy_range))
    printf '{"energy_uj":%s}\n' "$energy_delta" > "$OUT/energy.json"
fi

if [[ "${BENCHMARK_SUMMARY:-1}" == 1 ]]; then
    if ! python3 "$ROOT/benchmarks/summarize.py" "$OUT" \
        --output "$OUT/summary.txt" --json-out "$OUT/summary.json"; then
        echo "warning: benchmark summary generation failed for $OUT" >&2
    fi
fi

echo "$OUT"
exit "$status"
