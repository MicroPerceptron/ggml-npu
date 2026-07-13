# Direct buft residency and attribution

Tracks the Level Zero vs Direct buffer-type gap that drives residual staging and
scheduler affinity (#6; contributes to #4/#5). Does **not** change MoE fail-closed
policy (`is_opt_in_one`).

## Problem

| Runtime | `supports_buft` | Alloc honesty |
| --- | --- | --- |
| Level Zero | Any host buft (`ggml_backend_buft_is_host`) | May fall back to bare `alloc_host` under `NPU_HOST` |
| Direct (PR #22) | Only our `NPU_HOST` buft | SharedAllocationOwner only; null on failure |

Under L0, `ggml_backend_sched_backend_from_buffer` can assign NPU affinity to tensors that
live on CPU host buffers because NPU “supports” those bufts. Under Direct, those probes
return false, so affinity and zero-copy import require real `NPU_HOST` + shared owner.

Blindly widening Direct `supports_buft` back to all host bufts recreates the “owned but
unimportable” staging path (`FINDING/direct-host-buft-forced-staging`).

## Duty-cycle fix: claim + explicit stage (not supports_buft)

NPU is a spillover tier (no `-ngl` weight ownership). L0 faked affinity by accepting every
host buft. Direct keeps that probe **false** and instead:

1. **`GGML_SCHED_NPU_CLAIM_STAGE`** (llama.cpp; default on, `=0` disables) lets
   `offload_op` pull profitable ops whose weights/inputs live on foreign bufts
   (cause `1.npu`). Split creation copies unsupported inputs into the NPU buffer type.
2. Those copies and gallocr temps land on real **`SharedAllocationOwner` `NPU_HOST`**, so
   `backend_from_buffer` can keep subsequent affinity honestly (`our_shared` ↑).
3. `supports_buft` stays narrow — foreign probes still increment `foreign_host_refused`.

This is safe vs PR #22 because we never advertise an unsupported buft as supported; staging
is explicit in the scheduler copy path.

## Attribution

With `GGML_NPU_PROFILE` set, the exit dump includes:

- **buft probes:** `ours_ok`, `foreign_host_ok`, `foreign_host_refused`, `device_refused`,
  `null_refused`
- **residency** (Graph IR / operand capture): `our_shared`, `our_legacy_host`,
  `foreign_host`, `foreign_device`, `unalloc`, `dishonest_npu_host`

`foreign_host_refused` may stay high (still refuse unsafe probes). Success looks like
**`our_shared` ↑**, NPU disp and/or NPU GiB ↑, `dishonest_npu_host=0`.

Counters live in `src/backend/buft.rs` (`buft_attribution()`).

## Honest `NPU_HOST` residency

- Direct `buft_alloc_buffer` still fails closed if `alloc_shared_backend` fails (no bare
  host lie).
- `owns_tensor` under Direct requires `SharedAllocationOwner`, not merely buft pointer
  equality, so Graph IR / operands never claim zero-copy for unimportable storage.
- Intermediates assigned to the NPU backend still go through our buft via gallocr; if
  shared alloc fails, ggml sees OOM for that buft rather than a fake host buffer.

## Selective allowlist (gated, empty)

`GGML_NPU_IMPORT_FOREIGN_HOST=1` is required **and**
`importable_foreign_host_allowlist()` must list a **proven** buft class name. The list is
empty today, so the gate stays closed even if the env is set. Widening all host bufts
remains a non-goal.

### Non-goals

- Accepting every host buft like Level Zero.
- Fighting anti-unitary / cut-cost files for this lane.
- Retiring Level Zero until Direct residency + duty cycle close the measured gap.

## Tests

`cargo test --lib backend::buft` locks:

- allowlist empty + gate off (env alone insufficient);
- Direct narrow policy (ours only);
- broad host policy under non-Direct / L0 feature sets.
