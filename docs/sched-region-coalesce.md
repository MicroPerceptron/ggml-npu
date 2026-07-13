# Scheduler region coalesce (Rust-first)

Prefill-oriented prototype: widen **same-backend feasible intervals** before
`ggml_backend_sched` cuts the graph on backend color changes. Inspired by
axiom/kore bit-row set algebra for DAG intervals — **not** a Q1 tensor port.

Weighted **cut-cost** mode (GitHub #48) prices handoff edges (**latency**),
scores op-class compute weight (**throughput** — MatMul/FA heavy vs glue light),
grows onto existing NPU neighbors (anti-unitary), and preserves fat / heavy
NPU islands — without inventing claims.

## Division of responsibility

| Layer | Owns |
|-------|------|
| **Rust (`src/sched/`)** | Interval / island analysis, cut-cost scoring, recolor decisions, unit tests |
| **llama.cpp scheduler hook** | Pack POD rows, resolve `ggml_backend_sched_region_coalesce[_v2]` via `get_proc_address`, apply results behind buft-safe guards; env gates |
| **ggml-npu `offload_op` / policy** | Profitability and claim honesty — unchanged |

Default behavior is unchanged when the env flag is off **or** no provider is
loaded. Algorithm stays in Rust (cdylib/`rlib`) so deps and `core::simd` are
free of llama.cpp/ggml C++ ABI constraints; only thin POD crosses the boundary.

## Recommended production combo (Direct)

For Direct IVPU hybrid (narrow `supports_buft`) the measured duty-cycle trio is:

```bash
export GGML_SCHED_REGION_COALESCE=1
export GGML_SCHED_CUT_COST=1
export GGML_SCHED_NPU_CLAIM_STAGE=1
```

Tip locked LFM pp512 @ `-t 6` with this combo: ~247 tok/s / ~470 NPU disp/pass
(`BENCH/lfm-pp512-f662b28-coa1-cut1-cs1`). Do **not** enable claim-stage under
coalesce-only (without cut-cost) — coalesce can re-absorb NPU seeds into Vulkan.

### Master opt-in: `GGML_NPU_SCHED=auto`

One-flag product path (llama.cpp + this crate):

```bash
export GGML_NPU_SCHED=auto
```

When set to exact `auto`, the scheduler enables coalesce=1 + cut-cost=1 +
claim-stage=1 **only if** an NPU backend is present **and** this library reports
Direct-eligible via `ggml_backend_npu_sched_auto_eligible` (Direct narrow buft;
Level Zero / non-Direct builds return 0). Explicit `GGML_SCHED_*` overrides
always win when set. Unset / non-`auto` keeps the historical default-off trio
(safe; no SEGV / BindingMismatch reintroduction from silent enable).

**Decode-aware split (prefill vs gen):** locked LFM tg128 showed the prefill
trio *hurts* token-gen (auto ~1.9 tok/s / ~25k+ NPU disp/token vs sched-off
~2.6; Vulkan ~53). Under `auto`:

| Graph shape | Claim / cut / coalesce | `offload_op` / `should_pull` |
|-------------|------------------------|------------------------------|
| Prefill-sized (any `MUL_MAT` with `ne[1] > 1`) | Auto-applied trio (unless explicit `GGML_SCHED_*`) | Existing profitability |
| Decode / M=1 (all `MUL_MAT` have `ne[1] <= 1`) | Auto-applied trio **suspended** for that graph | Prefer **unclaim** (Vulkan/host-first) |

Explicit `GGML_SCHED_REGION_COALESCE` / `CUT_COST` / `NPU_CLAIM_STAGE` still win
for decode if set (experiments). `GGML_NPU_FORCE_GRAPH` still forces pulls.
Capability (`supports_op`) is unchanged — decode unclaim is profitability /
scheduler-gate only.

## Flags

- `GGML_NPU_SCHED=auto` — master opt-in: prefill Direct trio + decode
  Vulkan/host-first (skip auto claim/cut/coalesce; prefer unclaim for M=1).
- `GGML_SCHED_REGION_COALESCE=1` — enable pass (scheduler must resolve the Rust
  ABI from a loaded backend registry, typically `libggml_npu.so`).
- `GGML_SCHED_REGION_COALESCE=2` — enable + verbose flip / reassignment dump.
- `GGML_SCHED_REGION_COALESCE_MAX_ISLAND` — max foreign run length to absorb
  (default `4`).
- `GGML_SCHED_CUT_COST=1` — use **v2** ABI with nbytes + backend kinds; enable
  weighted / NPU-preserving absorb gates (`OPT_WEIGHTED`). Requires coalesce ≥1.
  Falls back to v1 if the v2 symbol is missing.
- `GGML_SCHED_NPU_CLAIM_STAGE=1` — Direct claim+stage (llama.cpp); seed via
  `supports_op` and stage foreign inputs into `NPU_HOST`.

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

## Cut-cost mode (v1 / #48 → anti-unitary + compute weight)

When `GGML_SCHED_CUT_COST=1`, the v2 provider additionally:

1. Scores sandwich edges as `nbytes × handoff_factor(from_kind, to_kind)`
   (heuristic table; NPU↔GPU highest) — the **latency** objective.
2. Packs `ggml_op` and scores **throughput** via op-class compute weights
   (MatMul / FA / conv / SSM **heavy**; norms/activations **medium**;
   elementwise/layout **light**). Combined objective prefers keeping heavy
   compute on fat NPU-compatible islands and absorbing light glue onto NPU
   neighbors.
3. **Anti-unitary grow-first**: absorb feasible foreign islands onto an
   **existing** NPU neighbor (classic sandwich **or** asymmetric
   `NPU|foreign|other`), then run classic same-backend sandwiches. Never paints
   a new NPU island when neither neighbor is NPU.
4. **Preserve fat / heavy NPU islands** (≥3 ops or ≥64 KiB for mixed/heavy, **≥4
   ops** or ≥64 KiB for all-light, **or** a heavy-class op ≥16 KiB) instead of
   dissolving them into Vulkan/CPU; thin light unitaries, small 2-op scraps, and
   barely-3 all-light islands may still dissolve so buft-safe coalesce duty-cycle
   wins are not undone (blanket preserve-all-NPU rejected — it revived ~412
   pp16 disp).
5. Boosts effective `max_island` across NPU growth edges: up to 2–3× on
   ≥256 KiB / ≥1 MiB boundaries; **fat-neighbor** preferential growth (+4
   all-light / +3 medium glue when the abutting NPU run already has ≥2 ops);
   unitary NPU neighbors stay at +2. Flat 2× light boosts were rejected
   (pp16 ~412 disp).
6. Uses `core::simd` (`portable-simd` feature) to batch nbytes×factor products
   when attributing total cut cost along a coloring.
7. Cheap process counters (absorb-into-NPU / dissolve-NPU / preserve-fat /
   absorb-other / light-glue-into-NPU / heavy-compute-preserved) via
   `ggml_backend_sched_cut_cost_stats[_reset]`; verbose coalesce (≥2) dumps
   them from the llama hook.

MoE policy remains `is_opt_in_one`; MatMulId stays unclaimed (Vulkan/host experts).
Host apply stays buft-safe (NPU/same-buft always; other targets only when
committed buffers are supported). Claim-stage + ELF pad staging (#62) remain
orthogonal — cut-cost does not widen Direct `supports_buft`.

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
#define GGML_SCHED_CUT_COST_OPT_WEIGHTED        (1u << 0)
#define GGML_SCHED_CUT_COST_OPT_COMPUTE_WEIGHT  (1u << 1)

struct ggml_sched_cut_cost_node {
    int32_t  backend_id;
    uint32_t feasible_mask;
    uint32_t flags;
    uint32_t nbytes;
    uint32_t op;              /* ggml_op; 0 ⇒ medium weight */
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
    uint64_t light_glue_into_npu;
    uint64_t heavy_compute_preserved;
};
```

Proc names: `ggml_backend_sched_cut_cost_stats`,
`ggml_backend_sched_cut_cost_stats_reset`.

### Auto eligibility

```c
typedef int (*ggml_backend_npu_sched_auto_eligible_t)(void);
```

Proc name: `ggml_backend_npu_sched_auto_eligible`. Returns `1` for Direct narrow
buft builds, `0` otherwise. Used by llama.cpp when `GGML_NPU_SCHED=auto`.

llama.cpp keeps the auto trio armed on the sched for **prefill**, and per
`split_graph` **suspends** auto-applied claim/cut/coalesce when every `MUL_MAT`
has `ne[1] <= 1` (decode). Explicit `GGML_SCHED_*` overrides are not suspended.
Rust `policy::should_pull` additionally prefers unclaim for decode-batch
tensors under exact `GGML_NPU_SCHED=auto` (unless `GGML_NPU_FORCE_GRAPH`).

LTO / same-process dynamic load is assumed to keep FFI cost negligible vs split
overhead.

## Future

- Full bit-row reachability for non-linear maximal feasible regions.
- Calibrated handoff / compute factors from short attribution (not a live BW oracle).
- Layer-windowed analysis for multi-thousand-node prefill graphs.

## Tests

```bash
cargo test -p ggml-npu sched::
```
