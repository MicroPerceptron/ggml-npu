# Scheduler region coalesce (Rust-first)

Prefill-oriented prototype: widen **same-backend feasible intervals** before
`ggml_backend_sched` cuts the graph on backend color changes. Inspired by
axiom/kore bit-row set algebra for DAG intervals — **not** a Q1 tensor port.

## Division of responsibility

| Layer | Owns |
|-------|------|
| **Rust (`src/sched/region_coalesce.rs`)** | Interval / island analysis, recolor decisions, unit tests |
| **llama.cpp scheduler hook** | Pack POD rows (`backend_id`, `feasible_mask`, flags), resolve `ggml_backend_sched_region_coalesce` via `get_proc_address`, apply results; env gate |
| **ggml-npu `offload_op` / policy** | Profitability and claim honesty — unchanged |

Default behavior is unchanged when the env flag is off **or** no provider is
loaded.

## Flag

- `GGML_SCHED_REGION_COALESCE=1` — enable pass (scheduler must resolve the Rust
  ABI from a loaded backend registry, typically `libggml_npu.so`).
- `GGML_SCHED_REGION_COALESCE=2` — enable + verbose flip / reassignment dump.
- `GGML_SCHED_REGION_COALESCE_MAX_ISLAND` — max foreign run length to absorb
  (default `4`).

## Algorithm (v0)

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

## C ABI

Symbol (exported from the NPU registry):

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

LTO / same-process dynamic load is assumed to keep FFI cost negligible vs split
overhead.

## Future (stubs)

- Edge-cost weighting: handoff bytes × backend hop instead of island length alone.
- Full bit-row reachability for non-linear “maximal feasible regions.”
- Layer-windowed analysis for multi-thousand-node prefill graphs.

## Tests

```bash
cargo test -p ggml-npu sched::region_coalesce
```
