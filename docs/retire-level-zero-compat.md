# Retire Level Zero compat — checklist

Concrete gates for deleting `level-zero-compat` / `session_level_zero` after Direct is
the sole Intel runtime. Tracks `MILESTONE/retire-level-zero-compat`. **Do not delete L0
code until every hard gate below is green.**

Related: [`direct-runtime-foundation.md`](direct-runtime-foundation.md),
[`direct-buft-residency.md`](direct-buft-residency.md),
[`compiler-migration-plan.md`](compiler-migration-plan.md) § Direct-submission
qualification, GitHub [#63](https://github.com/MicroPerceptron/ggml-npu/issues/63).

## Hard gates before delete

| Gate | Status | Notes / evidence |
| --- | --- | --- |
| Tiny artifact Direct↔L0 numeric KAT | **Done** | Bit-identical 32×FP32 vector-add; `docs/direct-runtime-foundation.md` |
| Session cancel admission + device-loss latch + firmware-gated reset | **Done** (PR #57 merged; #56) | Cooperative cancel; job-status `SUCCESS`/`ABORTED`/fault; latched device-loss; reset drops resident programs. Hermetic `direct::health` / `direct::oracle` |
| Claim-stage duty cycle near L0 (LFM shape) | **Done (parity of shape)** | Claim+stage + pad fix: locked Direct ~**205** tok/s / ~**470** disp vs L0 ~**266** / ~**488** (`BENCH/lfm-pp512-371b06d-coa1-cut1-cs1` vs `BENCH/lfm-pp512-68b7e8d-level-zero-flexiblas-coa1`). Gap remains; shape is L0-class |
| ELF pad / binding cover under claim-stage | **Done** | PR #62: stage when ELF `allocation_bytes` pad exceeds shared view (`FINDING/claim-stage-elf-pad-binding-mismatch`) |
| Model-corpus Direct↔L0 binding + numeric parity | **Required** | Same artifact/model corpus: binding metadata, relocations, HPI/firmware pins, resident accounting, import/copy, output tolerances — not tiny-only |
| In-flight / hang abort after partial execution | **Required** | Kernel hang-reset and abort of in-flight work beyond cooperative cancel + post-fault reset (#56 leftover) |
| Required Direct op coverage leftovers | **Required** | ShortConv / RoPE / FA / MoE paths still tracked on #9 / #10 / #11; Direct must not depend on L0 for those gates |
| Default-on knobs productization | **Required** | `GGML_SCHED_NPU_CLAIM_STAGE`, coalesce, cut-cost (and related) remain opt-in; product default path must not need a secret matrix to match L0-class duty cycle |

## Explicit non-gates (do not block delete on these alone)

- Closing the full tok/s gap to Vulkan (#4) or finishing anti-unitary cut-cost (#48 / #5).
- Blindly widening Direct `supports_buft` to every host buft (rejected; see
  `docs/direct-buft-residency.md`).
- Removing OpenVINO shim (`MILESTONE/remove-openvino` is a later, separate milestone).

## Done detail

### Tiny KAT + foundation

Direct DRM path proves discovery, context, BO, CMDQ, submit/wait, ELF/HPI link, and
bit-identical tiny vector-add vs L0 oracle. Cancellation and general event objects were
historically unsupported; reset recovery after device loss was unproven until #57.

### Session health (PR #57)

Landed toward retirement without touching `supports_buft`, MoE/attention claims, or
sched cut-cost:

- Cooperative cancel admission
- Command-BO job-status classification
- Latched device-loss refusing further dispatch
- Firmware-gated `try_reset_after_fault` (drops resident programs)
- Hermetic differential-oracle helpers (`compare_f32`, binding-meta parity)
- Context open / dispatch firmware re-negotiate; HPI loader/NNRT mismatch reject tests

### Duty cycle under honest buft + claim-stage

L0 still uses broad host `supports_buft`. Direct keeps narrow buft and uses
`GGML_SCHED_NPU_CLAIM_STAGE=1` (+ coalesce/cut-cost as documented) for L0-shaped
placement. After PR #62, locked tip claim-stage pp512 completes with ~99.3% no-copy
and ~470 disp/pass — comparable to L0’s ~488 disp, tok/s still below L0 mean.

## Still required (delete blockers)

1. **Corpus parity** — differential Direct vs L0 on the real LFM (and agreed) artifact
   set: bindings, numerics, resident bytes, sync. Tiny KAT alone is insufficient.
2. **Hang / in-flight abort** — prove recovery when firmware/driver hangs mid-job, not
   only cooperative cancel before admit and post-classified fault reset.
3. **Coverage leftovers** — finish or explicitly keep-unclaimed the #9/#10/#11 surfaces
   so L0 is not the only path that exercises them in CI/harness.
4. **Product defaults** — decide and land which of claim-stage / coalesce / cut-cost are
   default-on for Direct production builds, with measured regression gates.

## Delete procedure (only after hard gates)

1. Confirm tracking issue acceptance with linked evidence (benches, tests, docs).
2. Remove `level-zero-compat` feature, `session_level_zero`, and L0-only harness paths;
   keep Direct as the sole Intel session.
3. Update README, `docs/direct-runtime-foundation.md`, benchmarks README, and project
   memory milestone → finding/ADR as appropriate.
4. CI: no job requires Level Zero for Intel runtime proof.

## Status snapshot

| Item | State |
| --- | --- |
| Checklist doc | This file |
| Tracking issue | [#63](https://github.com/MicroPerceptron/ggml-npu/issues/63) |
| Session health | #57 merged on `develop` (`de626a9`); #56 closed |
| Pad/binding claim-stage fix | #62 merged on `develop` |
| L0 delete | **Not started** — hard gates above still open |
