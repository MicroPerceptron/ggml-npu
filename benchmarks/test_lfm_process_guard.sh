#!/usr/bin/env bash
# Non-hardware smoke test for benchmarks/lib/lfm_process_guard.sh
# Run: ./benchmarks/test_lfm_process_guard.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=benchmarks/lib/lfm_process_guard.sh
source "$ROOT/benchmarks/lib/lfm_process_guard.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "ok: $*"; }

bash -n "$ROOT/benchmarks/lib/lfm_process_guard.sh"
bash -n "$ROOT/benchmarks/run_lfm_baseline.sh"
pass "bash -n syntax"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"; [[ -n "${waiter:-}" ]] && kill -KILL "$waiter" 2>/dev/null || true; [[ -n "${pgid:-}" ]] && lfm_kill_process_group "$pgid" 0 2>/dev/null || true' EXIT

# Child tree: leader bash spawns sleep grandchildren that share the session.
pgid_file="$tmp/pgid"
waiter="$(lfm_start_process_group "$pgid_file" "$tmp/out" "$tmp/err" -- \
    bash -c 'sleep 60 & sleep 60 & wait')"
pgid="$(tr -d '[:space:]' < "$pgid_file")"
[[ "$pgid" =~ ^[0-9]+$ ]] || fail "invalid pgid '$pgid'"
pass "started process group pgid=$pgid waiter=$waiter"

# Confirm members exist in the session.
members_before="$(ps -o pid= -s "$pgid" 2>/dev/null | wc -l)"
(( members_before >= 1 )) || fail "expected session members, got $members_before"

start_ms="$(date +%s%3N)"
lfm_kill_process_group "$pgid" 50
# Wait briefly for reaping
for _ in 1 2 3 4 5 6 7 8 9 10; do
    left="$(ps -o pid= -s "$pgid" 2>/dev/null | tr -d ' \n' || true)"
    [[ -z "$left" ]] && break
    sleep 0.05
done
end_ms="$(date +%s%3N)"
elapsed=$((end_ms - start_ms))
left="$(ps -o pid= -s "$pgid" 2>/dev/null | tr -d ' \n' || true)"
[[ -z "$left" ]] || fail "session still alive after kill: $left"
(( elapsed < 2000 )) || fail "kill took ${elapsed}ms (>2s)"
pass "killed process group in ${elapsed}ms"

# Waiter should observe child death.
set +e
wait "$waiter" 2>/dev/null
set -e
waiter=""
pgid=""
pass "waiter reaped"

# effective floor rises when swap is in use — exercise via env without requiring swap.
MIN_AVAILABLE_KB=4194304
MIN_AVAILABLE_UNDER_SWAP_KB=6291456
SWAP_RAISE_FLOOR_USED_KB=0 # treat any swap accounting path; force raise by faking via function? 
# Call effective with SWAP_RAISE_FLOOR_USED_KB higher than current used to avoid raise,
# then with 0 to force raise if any swap exists OR just unit-check the max logic inline.
floor_relaxed="$(
    SWAP_RAISE_FLOOR_USED_KB=9999999999
    lfm_effective_min_available_kb
)"
[[ "$floor_relaxed" == "4194304" ]] || fail "expected base floor, got $floor_relaxed"
pass "effective floor respects base MIN_AVAILABLE_KB"

# Preflight refusal path with absurd floor.
set +e
MIN_AVAILABLE_KB=999999999999
ALLOW_HIGH_SWAP=1
out="$(lfm_preflight_memory_or_die 2>&1)"
rc=$?
set -e
(( rc == 2 )) || fail "preflight should refuse tiny MemAvailable vs huge floor, rc=$rc"
[[ "$out" == *Refusing* ]] || fail "preflight message missing: $out"
pass "preflight refuses low MemAvailable"

echo "ALL SMOKE CHECKS PASSED"
