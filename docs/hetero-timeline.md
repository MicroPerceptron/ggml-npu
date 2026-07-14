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
| `<prefix>.summary.txt` | Busy% / idle% / copy_ms / load_bytes per CPU · GPU · NPU · Meta |

The same summary is printed on stderr.

## Event kinds

| Kind | Domain meaning |
| --- | --- |
| `forward` | Full `ggml_backend_sched_graph_compute` window (CPU lane; **not** counted as busy%) |
| `graph_compute` | Backend split compute (CPU / GPU / NPU / Meta by device type) |
| `stage_in` / `copy_out` | Host staging memcpy / zero_outputs (CPU lane) |
| `submit` | Host queue kick before device wait (CPU lane on Direct/L0) |
| `wait` | Device BO / queue wait (NPU lane) |
| `sched_copy` | Scheduler tensor copy between backends (dst domain) |
| `synchronize` | `ggml_backend_synchronize` wait on a device (CPU / GPU / NPU / Meta) |
| `mem_load` | Bytes loaded onto a domain (instant; may also set `bytes` on copy spans) |

## Domains (Chrome `tid`)

- `cpu` (tid 1) — real CPU graphs, host stage/zero/submit/copy-out
- `gpu` (tid 2) — Vulkan (or other GPU) graph compute + sync
- `npu` (tid 3) — NPU device wait (and NPU graph_compute / synchronize when probed)
- `meta` (tid 4) — tensor-parallel Meta facade (record/submit + post-token fence reclaim)

`DEVICE_TYPE_META` maps to **meta**, not CPU — Meta→Vulkan fence waits must not inflate
CPU busy%.

## How to interpret the summary

- **window_ms** — span from earliest to latest event timestamp (typically one or more decode steps in a tg run).
- **busy% / idle%** — merged non-overlapping busy intervals over that window. Busy kinds are
  `graph_compute`, `submit`, `wait`, `synchronize`, `stage_in`, `copy_out`, and `sched_copy`
  (matching `is_busy_kind`). `forward` is excluded so the outer sched span cannot drive CPU ≈ 100%.
- **copy_ms** — sum of `stage_in` + `copy_out` + `sched_copy` durations on that domain
  (also included in busy% on the same domain).
- **load_bytes** — sum of `bytes` on `mem_load` / copy events attributed to that domain.

Overlapping same-domain busy intervals are merged so busy% never exceeds 100 for a domain.

### Before / after attribution (host-sync fix)

| Span | Before | After |
| --- | --- | --- |
| Meta `graph_compute` / `synchronize` | CPU | **meta** |
| Direct/L0 `stage_in` (incl. `zero_outputs`) | NPU | **cpu** |
| Direct/L0 `submit` | NPU | **cpu** |
| Direct/L0 `wait` | NPU | **npu** (unchanged) |
| Direct/L0 `copy_out` | NPU | **cpu** |

On TP `-sm tensor`, post-token Meta fence reclaim therefore shows on the **meta** lane (often
near GPU synchronize), while CPU busy reflects real host work / graphs — not Meta waits.
On sched-auto with NPU dispatches, NPU busy ≈ device wait; host memcpy/zero appear as CPU
`copy_ms` / busy.

## Host probes (llama.cpp)

llama.cpp resolves these via NPU `get_proc_address` (no-ops when the symbol or env is absent):

- `ggml_npu_timeline_enabled`
- `ggml_npu_timeline_now_ns`
- `ggml_npu_timeline_span`
- `ggml_npu_timeline_instant`

Domains passed from llama: `0=cpu`, `1=gpu`, `2=npu`, `3=meta`.

Thin wrappers around `graph_compute`, `synchronize`, sched copies, and the outer sched `graph_compute` emit domain spans so a hetero `-sm tensor` run does not need Vulkan driver timestamps.

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

Expect a non-trivial **meta** busy% on TP (fence reclaim), not 97% “CPU” from Meta sync alone.
For NPU wait attribution: `-ngl 0` + `GGML_NPU_SCHED=auto` should show NPU busy ≈ wait, with
host stage/zero on CPU `copy_ms`.

## Relation to PR #71 (telemetry SHM)

Draft PR #71 (`GGML_NPU_TELEMETRY_SHM` live ring for TUI/wasmrt) remains a separate live-feed
substrate. This timeline reuses the same seqlock-ring idea but targets **bench artifacts**
(JSONL / chrome-trace / busy-idle summary) with span semantics. It does not wire or require
the dormant `#71` modules.

## Constraints

- Does not widen `supports_buft` or change claim policy.
- Does not require deleting Level Zero compat.
- Default path stays clean when `GGML_NPU_TIMELINE` is unset.
