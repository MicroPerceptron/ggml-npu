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

## Attribution (this change)

With `GGML_NPU_PROFILE` set, the exit dump includes:

- **buft probes:** `ours_ok`, `foreign_host_ok`, `foreign_host_refused`, `device_refused`,
  `null_refused`
- **residency** (Graph IR / operand capture): `our_shared`, `our_legacy_host`,
  `foreign_host`, `foreign_device`, `unalloc`, `dishonest_npu_host`

`foreign_host_refused` is the Direct-vs-L0 signal: probes L0 would accept that Direct
refuses. `dishonest_npu_host` must stay **0** on Direct (legacy `alloc_host` branded as
`NPU_HOST` without a shared owner).

Counters live in `src/backend/buft.rs` (`buft_attribution()`).

## Honest `NPU_HOST` residency

- Direct `buft_alloc_buffer` still fails closed if `alloc_shared_backend` fails (no bare
  host lie).
- `owns_tensor` under Direct requires `SharedAllocationOwner`, not merely buft pointer
  equality, so Graph IR / operands never claim zero-copy for unimportable storage.
- Intermediates assigned to the NPU backend still go through our buft via gallocr; if
  shared alloc fails, ggml sees OOM for that buft rather than a fake host buffer.

## Selective allowlist (next step — gated, not enabled)

`importable_foreign_host_allowlist_enabled()` is hard-`false`. A future spike may admit
**named** foreign host bufts only when:

1. Hermetic tests prove import/export (or DMA-BUF handoff) for that buft class;
2. Profile shows a material `foreign_host_refused` → staged binding reduction;
3. MoE / fail-closed contracts stay untouched.

### Non-goals

- Accepting every host buft like Level Zero.
- Changing cut-cost / region-coalesce / llama.cpp coalesce apply (owned elsewhere).
- Retiring Level Zero until Direct residency + duty cycle close the measured gap.

## Tests

`cargo test --lib backend::buft` locks:

- allowlist off;
- Direct narrow policy (ours only);
- broad host policy under non-Direct / L0 feature sets.
