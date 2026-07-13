# Hetero forward timeline

Env-gated busy/idle + memory-load timeline for tensor-parallel / hetero benches
(`-sm tensor` + Vulkan/NPU). Off by default — the hot path pays one relaxed load when unset.

## Enable

```bash
export GGML_NPU_TIMELINE=/path/to/artifacts/hetero-tg128
# then run llama-bench / llama-cli with the NPU backend loaded as usual
```

On process exit the recorder writes:

| Artifact | Contents |
| --- | --- |
| `<prefix>.jsonl` | One event per line (`ts_ns`, `dur_ns`, `domain`, `kind`, `bytes`, `seq`) |
| `<prefix>.trace.json` | Chrome Trace Event Format (`chrome://tracing` / Perfetto) |
| `<prefix>.summary.txt` | Busy% / idle% / copy_ms / load_bytes per CPU · GPU · NPU |

The same summary is printed on stderr.

## Event kinds

| Kind | Domain meaning |
| --- | --- |
| `forward` | Full `ggml_backend_sched_graph_compute` window (CPU lane) |
| `graph_compute` | Backend split compute (CPU / GPU / NPU by device type) |
| `stage_in` / `copy_out` | NPU host staging copies |
| `submit` / `wait` | NPU queue submit vs device wait |
| `sched_copy` | Scheduler tensor copy between backends |
| `synchronize` | `ggml_backend_synchronize` wait on a device |
| `mem_load` | Bytes loaded onto a domain (instant; may also set `bytes` on copy spans) |

## Domains (Chrome `tid`)

- `cpu` (tid 1) — scheduler, CPU graphs, host staging / copy-out
- `gpu` (tid 2) — Vulkan (or other GPU) graph compute + sync
- `npu` (tid 3) — NPU stage / submit / wait / copy-out

## How to interpret the summary

- **window_ms** — span from earliest to latest event timestamp (typically one or more decode steps in a tg run).
- **busy% / idle%** — merged non-overlapping busy intervals (`graph_compute`, `submit`, `wait`, `synchronize`, `forward`) over that window.
- **copy_ms** — sum of `stage_in` + `copy_out` + `sched_copy` durations on that domain.
- **load_bytes** — sum of `bytes` on `mem_load` / copy events attributed to that domain.

Overlapping same-domain busy intervals are merged so busy% never exceeds 100 for a domain.

## Host probes (llama.cpp)

llama.cpp resolves these via NPU `get_proc_address` (no-ops when the symbol or env is absent):

- `ggml_npu_timeline_enabled`
- `ggml_npu_timeline_now_ns`
- `ggml_npu_timeline_span`
- `ggml_npu_timeline_instant`

Thin wrappers around `graph_compute`, `synchronize`, sched copies, and the outer sched `graph_compute` emit CPU/GPU spans so a hetero `-sm tensor` run does not need Vulkan driver timestamps.

## Smoke (tg128 r=1)

```bash
OUT=benchmarks/results/hetero-timeline-smoke
mkdir -p "$OUT"
export GGML_NPU_TIMELINE="$OUT/timeline"
export GGML_BACKEND_PATH=target/release/libggml_npu.so
# plus OVC / VK_ICD / THREADS=6 as usual
llama-bench -m LFM2.5-8B-A1B-Q4_K_M.gguf -p 0 -n 128 -r 1 -fa on -t 6 \
  -ngl 99 -dev Vulkan0/NPU -sm tensor -ts 0.5/0.5
cat "$OUT/timeline.summary.txt"
```

## Relation to PR #71 (telemetry SHM)

Draft PR #71 (`GGML_NPU_TELEMETRY_SHM` live ring for TUI/wasmrt) remains a separate live-feed
substrate. This timeline reuses the same seqlock-ring idea but targets **bench artifacts**
(JSONL / chrome-trace / busy-idle summary) with span semantics. It does not wire or require
the dormant `#71` modules.

## Constraints

- Does not widen `supports_buft` or change claim policy.
- Does not require deleting Level Zero compat.
- Default path stays clean when `GGML_NPU_TIMELINE` is unset.
