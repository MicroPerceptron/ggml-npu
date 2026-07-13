# Scheduler region coalesce (Rust-first)

Prefill-oriented prototype: widen **same-backend feasible intervals** before
`ggml_backend_sched` cuts the graph on backend color changes. Inspired by
axiom/kore bit-row set algebra for DAG intervals — **not** a Q1 tensor port.

Weighted **cut-cost** mode (GitHub #48) prices handoff edges, grows onto
existing NPU neighbors (anti-unitary), and preserves only fat NPU islands —
without inventing claims.

## Division of responsibility

| Layer | Owns |
|-------|------|
| **Rust (`src/sched/`)** | Interval / island analysis, cut-cost scoring, recolor decisions, unit tests |
| **llama.cpp scheduler hook** | Pack POD rows, resolve `ggml_backend_sched_region_coalesce[_v2]` via `get_proc_address`, apply results behind buft-safe guards; env gates |
| **ggml-npu `offload_op` / policy** | Profitability and claim honesty — unchanged |

Default behavior is unchanged when the env flag is off **or** no provider is
loaded. Algorithm stays in Rust (cdylib/`rlib`) so deps and `core::simd` are
free of llama.cpp/ggml C++ ABI constraints; only thin POD crosses the boundary.

## Flags

- `GGML_SCHED_REGION_COALESCE=1` — enable pass (scheduler must resolve the Rust
  ABI from a loaded backend registry, typically `libggml_npu.so`).
- `GGML_SCHED_REGION_COALESCE=2` — enable + verbose flip / reassignment dump.
- `GGML_SCHED_REGION_COALESCE_MAX_ISLAND` — max foreign run length to absorb
  (default `4`).
- `GGML_SCHED_CUT_COST=1` — use **v2** ABI with nbytes + backend kinds; enable
  weighted / NPU-preserving absorb gates (`OPT_WEIGHTED`). Requires coalesce ≥1.
  Falls back to v1 if the v2 symbol is missing.

## Algorithm (v0 / length-only)

After llama.cpp assignment passes 1–4, for non-view nodes in topo order:

1. Treat consecutive equal `backend_id` runs as intervals (views skipped for
   cut detection only).
2. If a short run (≤ max island) sits between two runs of backend `B`, and every
   island node has `feasible_mask` bit `B` set and is not `PINNED`, recolor the
   island to `B`.
3. Repeat to fixpoint (checkerboard collapse).
4. **llama.cpp apply guard** (not Rust): always apply when the target is
   `GGML_BACKEND_DEVICE_TYPE_NPU` or shares a preferred buft with the node’s
   prior backend. Other targets (Vulkan sandwiches) apply only when every
   **already-committed** buffer that would live on the destination after this
   apply is supported by that backend (`ggml_backend_supports_buft`); unallocated
   temps are safe (gallocr assigns the target buft). Refusing incompatible
   committed buffers avoids `ggml_vk_build_graph` SIGSEGV on pp128 while restoring
   buffer-safe island absorbs. Then re-sync view ops to `view_src`.

Does **not** invent NPU claims: feasibility comes from the scheduler’s
`supports_op` packing. Weights / `INPUT` stay pinned. Rust leaves `VIEW` rows
unchanged so the host can honor the view_src invariant.

## Cut-cost mode (v1 / #48 MVP → anti-unitary)

When `GGML_SCHED_CUT_COST=1`, the v2 provider additionally:

1. Scores sandwich edges as `nbytes × handoff_factor(from_kind, to_kind)`
   (heuristic table; NPU↔GPU highest).
2. **Anti-unitary grow-first**: absorb feasible foreign islands onto an
   **existing** NPU neighbor (classic sandwich **or** asymmetric
   `NPU|foreign|other`), then run classic same-backend sandwiches. Never paints
   a new NPU island when neither neighbor is NPU.
3. **Preserve fat NPU islands** (≥3 ops or ≥64 KiB) instead of dissolving them
   into Vulkan/CPU; thin unitaries and small 2-op scraps may still dissolve so
   buft-safe coalesce duty-cycle wins are not undone (blanket preserve-all-NPU
   rejected — it revived ~412 pp16 disp).
4. Boosts effective `max_island` across NPU growth edges (+2 light; up to 2–3×
   on ≥256 KiB / ≥1 MiB boundaries) so fat prefill glue can merge. Larger light
   boosts were rejected (pp16 ~412 disp).
5. Uses `core::simd` (`portable-simd` feature) to batch nbytes×factor products
   when attributing total cut cost along a coloring.
6. Cheap process counters (absorb-into-NPU / dissolve-NPU / preserve-fat /
   absorb-other) via `ggml_backend_sched_cut_cost_stats[_reset]`; verbose
   coalesce (≥2) dumps them from the llama hook.

MoE fail-closed policy is untouched (`is_opt_in_one`; Argsort/MatMulId unclaim).
Host apply stays buft-safe (NPU/same-buft always; other targets only when
committed buffers are supported).

## C ABI

### v1

```c
#define GGML_SCHED_REGION_COALESCE_FLAG_VIEW   (1u << 0)
#define GGML_SCHED_REGION_COALESCE_FLAG_PINNED (1u << 1)

struct ggml_sched_region_coalesce_node {
    int32_t  backend_id;      /* in/out */
    uint32_t feasible_mask;   /* bit i => backend i supports op */
    uint32_t flags;
};

typedef int (*ggml_backend_sched_region_coalesce_t)(
    struct ggml_sched_region_coalesce_node * nodes,
    int n_nodes,
    int n_backends,
    int max_island);
```

Proc name: `ggml_backend_sched_region_coalesce`.

### v2 (cut-cost)

```c
#define GGML_SCHED_CUT_COST_OPT_WEIGHTED (1u << 0)

struct ggml_sched_cut_cost_node {
    int32_t  backend_id;
    uint32_t feasible_mask;
    uint32_t flags;
    uint32_t nbytes;
};

typedef int (*ggml_backend_sched_region_coalesce_v2_t)(
    struct ggml_sched_cut_cost_node * nodes,
    int n_nodes,
    int n_backends,
    int max_island,
    const uint8_t * backend_kinds, /* ggml_backend_dev_type per backend */
    uint32_t options);
```

Proc name: `ggml_backend_sched_region_coalesce_v2`.

```c
struct ggml_sched_cut_cost_stats {
    uint64_t absorbed_into_npu;
    uint64_t dissolved_npu;
    uint64_t preserved_fat_npu;
    uint64_t absorbed_other;
};

typedef void (*ggml_backend_sched_cut_cost_stats_t)(
    struct ggml_sched_cut_cost_stats * out);
typedef void (*ggml_backend_sched_cut_cost_stats_reset_t)(void);
```

Proc names: `ggml_backend_sched_cut_cost_stats`,
`ggml_backend_sched_cut_cost_stats_reset`.

LTO / same-process dynamic load is assumed to keep FFI cost negligible vs split
overhead.

## Future

- Full bit-row reachability for non-linear maximal feasible regions.
- Calibrated handoff factors from short attribution (not a live BW oracle).
- Layer-windowed analysis for multi-thousand-node prefill graphs.

## Tests

```bash
cargo test -p ggml-npu sched::
```
