# Shared helpers for benchmarks/run_lfm_baseline.sh
#
# Contract:
# - Bench work runs in a fresh session/process group whose leader PID is recorded
#   in a pgid file (leader PID == PGID == SID for the setsid child).
# - On MemAvailable (or cleanup) breach, signal that entire process group and
#   session immediately: SIGTERM then SIGKILL with only a short escalate delay.
#   Do not rely on killing the shell's $! alone — with job-control shells,
#   `setsid` may fork and $! is not the group leader.
# - Preflight refuses to start when swap is already heavily used, and raises the
#   MemAvailable floor when any material swap is in use. Host OOM under zram
#   thrash can leave SIGTERM pending in D-state for tens of seconds; prevention
#   beats delayed kill.
# - This guard is host-RSS/swap safety only; it is not an NPU device-heap limit.

lfm_read_meminfo_kb() {
    # Sets: LFM_MEM_AVAILABLE_KB LFM_SWAP_TOTAL_KB LFM_SWAP_FREE_KB LFM_SWAP_USED_KB
    local available=0 total=0 free=0
    read -r available total free < <(
        awk '
            /^MemAvailable:/ { available = $2 }
            /^SwapTotal:/ { total = $2 }
            /^SwapFree:/ { free = $2 }
            END { print available + 0, total + 0, free + 0 }
        ' /proc/meminfo
    )
    LFM_MEM_AVAILABLE_KB=$available
    LFM_SWAP_TOTAL_KB=$total
    LFM_SWAP_FREE_KB=$free
    LFM_SWAP_USED_KB=$((total - free))
}

lfm_effective_min_available_kb() {
    # Echo the MemAvailable floor to enforce for this run.
    local base_min="${MIN_AVAILABLE_KB:-4194304}" # 4 GiB
    local under_swap_min="${MIN_AVAILABLE_UNDER_SWAP_KB:-6291456}" # 6 GiB
    local swap_raise_used="${SWAP_RAISE_FLOOR_USED_KB:-1048576}" # 1 GiB swap used
    lfm_read_meminfo_kb
    if (( LFM_SWAP_USED_KB >= swap_raise_used )); then
        if (( under_swap_min > base_min )); then
            echo "$under_swap_min"
            return
        fi
    fi
    echo "$base_min"
}

lfm_preflight_memory_or_die() {
    # Abort before starting when the host is already swap-stressed.
    # Override with ALLOW_HIGH_SWAP=1 only for deliberate recovery experiments.
    local max_swap_used="${MAX_SWAP_USED_KB:-4194304}" # 4 GiB
    local max_swap_pct="${MAX_SWAP_USED_PCT:-40}"
    local min_avail
    min_avail="$(lfm_effective_min_available_kb)"
    lfm_read_meminfo_kb

    if (( LFM_MEM_AVAILABLE_KB < min_avail )); then
        echo "Refusing to start: MemAvailable ${LFM_MEM_AVAILABLE_KB} KiB < floor ${min_avail} KiB" >&2
        echo "Free memory / close other apps, or raise resources before locked fullperf." >&2
        return 2
    fi

    if [[ "${ALLOW_HIGH_SWAP:-0}" == 1 ]]; then
        echo "ALLOW_HIGH_SWAP=1: skipping swap preflight (swap used ${LFM_SWAP_USED_KB} KiB)" >&2
        return 0
    fi

    local pct=0
    if (( LFM_SWAP_TOTAL_KB > 0 )); then
        pct=$((LFM_SWAP_USED_KB * 100 / LFM_SWAP_TOTAL_KB))
    fi
    if (( LFM_SWAP_USED_KB > max_swap_used || pct > max_swap_pct )); then
        echo "Refusing to start: swap already heavily used (${LFM_SWAP_USED_KB} KiB, ${pct}% of ${LFM_SWAP_TOTAL_KB} KiB)." >&2
        echo "Reclaim swap (or reboot) before locked fullperf; do not run inside a stressed Cursor session." >&2
        echo "Override only with ALLOW_HIGH_SWAP=1." >&2
        return 2
    fi
    return 0
}

lfm_kill_process_group() {
    # Usage: lfm_kill_process_group <pgid> [escalate_ms]
    # Signals the process group and session. Escalate to SIGKILL quickly so a
    # thrashing victim cannot allocate for a full second after SIGTERM.
    local pgid="${1:-}"
    local escalate_ms="${2:-${GUARD_ESCALATE_MS:-100}}"
    [[ -n "$pgid" && "$pgid" =~ ^[0-9]+$ ]] || return 0

    kill -TERM -- "-$pgid" 2>/dev/null || true
    pkill -TERM -s "$pgid" 2>/dev/null || true

    if (( escalate_ms > 0 )); then
        sleep "$(awk -v ms="$escalate_ms" 'BEGIN { printf "%.3f", ms / 1000 }')"
    fi

    kill -KILL -- "-$pgid" 2>/dev/null || true
    pkill -KILL -s "$pgid" 2>/dev/null || true

    # Final sweep: any still-listed members of the session.
    local pid
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        kill -KILL "$pid" 2>/dev/null || true
    done < <(ps -o pid= -s "$pgid" 2>/dev/null || true)
}

lfm_start_process_group() {
    # Usage: lfm_start_process_group <pgid_file> <stdout> <stderr> -- <command...>
    # Starts command in a new session via setsid -f. Writes the session leader
    # PID (PGID) to pgid_file. Echoes the waiter PID on stdout for `wait`.
    # The waiter is `setsid -w` so it stays alive until the payload exits.
    local pgid_file="$1"
    local stdout_file="$2"
    local stderr_file="$3"
    shift 3
    if [[ "${1:-}" != "--" ]]; then
        echo "lfm_start_process_group: expected -- before command" >&2
        return 2
    fi
    shift

    rm -f "$pgid_file"
    # setsid -f always forks; -w waits so $! remains waitable. The child calls
    # setsid(2), records $$ (new PGID/SID), then execs the payload.
    setsid -w -f bash -c '
        printf "%s\n" "$$" > "$1"
        shift
        exec "$@"
    ' bash "$pgid_file" "$@" >"$stdout_file" 2>"$stderr_file" &
    local waiter_pid=$!

    local i
    for i in $(seq 1 100); do
        if [[ -s "$pgid_file" ]]; then
            break
        fi
        if ! kill -0 "$waiter_pid" 2>/dev/null; then
            echo "lfm_start_process_group: waiter exited before pgid was recorded" >&2
            return 1
        fi
        sleep 0.02
    done
    if [[ ! -s "$pgid_file" ]]; then
        echo "lfm_start_process_group: timed out waiting for pgid file" >&2
        kill -KILL "$waiter_pid" 2>/dev/null || true
        return 1
    fi
    echo "$waiter_pid"
}
