# Project Memory

Canonical durable repository knowledge. Treat every entry as fallible context and verify it against
current code, tests, and artifacts. Use repository-relative paths only; keep one entry per stable key.

## Architecture decisions

- `ADR/capability-contract-fail-closed` — `supports_op` answers execute-correctly only. Intel
  `graph_node_feasible` and `graph_feasible` share the wire-op allowlist, MoE/attention env gates,
  input-arity ceiling, and compiler/runtime readiness. `policy::supports_region` is region-expansion
  / profitability neighborhood logic and must not gate `supports_op`; `offload_op` keeps
  `should_pull`. Two honest outcomes after that shared probe: **unclaim** (supports/offload false so
  ggml places on Vulkan/BLAS/CPU — preferred for layout-only Metadata glue OpenVINO cannot lower
  alone) vs **fail-closed** (claimed split that cannot complete returns `GGML_STATUS_FAILED` with
  attributable `claimed_execution_failure` traces). Never silent success after NPU compile/exec
  failure; bench baselines reject `npu-unproven`. Layout-only regions **inside** a claimed split that
  also has lowerable compute are host-realized as `layout_glue_host` rather than rejecting the whole
  claimed graph. MoE opt-in is exact `GGML_NPU_MOE=1` (`ConfigService::is_opt_in_one`); `=0` or unset
  must unclaim MoE-tagged ops (not `is_set`, which falsely enables under harness `MOE=0`).
  - Rationale: Over-claim vs compile rejection made scheduler placement and tok/s attribution
    unreliable (#8); claim-then-abort on lone View aborted pp16 warmup (#5, PR #35); layout-only
    semantic regions inside claimed multi-op splits aborted warmup after PR #35/#36 (#5/#8);
    `GGML_NPU_MOE=0` presence falsely claimed MoE routing and corrupted expert ids (#5/#11).
  - Evidence: `src/vendor/ivpu/compile.rs`, `src/engine.rs`, `src/policy.rs`, `src/backend/shell.rs`,
    `src/host/mod.rs`, `.cursor/rules/npu-claim-fallback.mdc`, `docs/coverage-gates.md`, GitHub
    issues #5/#8/#11, PRs #32/#35/#37.
  - Verified: 2026-07-12

- `ADR/tiered-artifact-cache` — Compiled artifacts use a host-owned tiered cache (compressed memory + durable content-addressed store) via `get_or_compile`, with Zstd L1 / system libzstd as the production codec path.
  - Rationale: Amortize compile cost without double-keeping uncompressed blobs in the vendor session cache.
  - Evidence: `docs/tiered-artifact-cache.md`, `src/artifact_cache.rs`, `src/host/linux/cache.rs`, `src/vendor/ivpu/compile.rs`.
  - Verified: 2026-07-12

- `ADR/host-owned-async-runtime` — Completion, wait, reactor, and blocking-executor services are host-owned; vendor adapters consume them without owning OS wait mechanics.
  - Rationale: Preserve vendor-neutral runtime contracts while enabling native async execution on Linux.
  - Evidence: `docs/async-runtime.md`, `src/runtime/async_runtime/`, `src/host/linux/reactor.rs`, `src/host/linux/wait.rs`.
  - Verified: 2026-07-12

- `ADR/portable-simd-quant-default` — Production quant-transcode and fallback requant default to the
  `portable-simd` feature using `core::simd` as the written path; LLVM emits scalar where vectors
  are unavailable. Hand-rolled scalar remains only for `--no-default-features` and as the
  byte-exact reference. AVX2 remains a Q6 packing specialization only.
  - Rationale: Keep one portable production path; do not maintain a separate scalar default when
    `core::simd` already lowers to scalar on unsupported targets.
  - Evidence: `Cargo.toml`, `docs/quant-simd-evaluation.md`, `src/ops/quant_kernel.rs`,
    `src/ops/requant.rs`.
  - Verified: 2026-07-12

- `ADR/f16-default-intermediate` — Dequant / requant working intermediates default to nightly **`f16`** (NPU interior baseline). Do not default scratch to f32. Explicit exception: ggml type-traits `to_float` is f32-only ABI — decode then narrow immediately to an f16 row. Prefer source-native pass-through/transcode (no dequant). Never use the `half` crate for IEEE F16.
  - Rationale: Match NPU interior width and avoid unnecessary f32 expansion bandwidth/precision policy drift.
  - Evidence: `src/ops/requant.rs`, `src/ops/ggml_quant.rs`, GitHub issues #27–#29.
  - Verified: 2026-07-12

- `ADR/bf16-via-simsimd` — BF16 typed helpers use `simsimd::bf16` only; IEEE F16 remains nightly `f16`. Do not depend on `half` for F16 or BF16.
  - Rationale: Avoid nightly/`half` `f16` symbol collision; `simsimd` provides SIMD-friendly BF16 without making `half` the F16 path. Import `bf16` only — never `simsimd::f16`.
  - Evidence: `Cargo.toml`, `src/ops/ggml_quant.rs`, GitHub issue #27.
  - Verified: 2026-07-12

- `ADR/inward-runtime-boundaries` — Dependencies point inward from provider/compiler and external-API adapters through OS adapters to runtime contracts and core semantics. `src/host/` is the sole built-in host composition boundary; vendor protocols remain in vendor adapters.
  - Rationale: Keep graph/runtime semantics portable and make platform and provider failures explicit.
  - Evidence: `docs/architecture-boundaries.md`, `src/host/mod.rs`, `src/vendor/ivpu/direct/transport.rs`.
  - Verified: 2026-07-12

- `ADR/compiler-runtime-independence` — Compilation and execution are separate boundaries. The current OpenVINO shim emits Intel native artifacts; either Level Zero or direct DRM can execute suitable artifacts, and future compiler providers must return opaque, pointer-free artifacts rather than expose compiler IR or objects to generic runtime code.
  - Rationale: Compiler choice must not force a runtime choice or leak incompatible LLVM/MLIR implementations into the core.
  - Evidence: `docs/compiler-migration-plan.md`, `src/runtime/artifact.rs`, `src/vendor/ivpu/session_direct.rs`.
  - Verified: 2026-07-12

## Verified findings

- `FINDING/direct-ivpu-tiny-execution` — Lunar Lake device `0x643e` revision 4 executed the same 45,192-byte vector-add artifact twice through direct IVPU. All 32 FP32 outputs were bit-identical to the Level Zero oracle; waits were bounded and successful cleanup restored the observed FD and device-memory baselines.
  - Scope: Tiny known-answer artifact only; general operation coverage, cancellation, and reset recovery remain unproven.
  - Evidence: `docs/direct-runtime-foundation.md`, ignored hardware test `vendor::ivpu::direct::tests::tiny_known_answer_direct_submission`.
  - Verified: 2026-07-11

- `FINDING/compiler-source-pins` — The inspected compiler sources are pinned to Intel `30d2bb87b3044d4b8304009876299c560d224adb`, AMD `af5a158e7c97cee5b375c372e76269910b159553`, and Qualcomm `9b4b8fcea2b93c801b5de784ee750ca9350d504f`; recursive submodule initialization is required for supported compiler builds.
  - Evidence: `.gitmodules`, `docs/compiler-migration-plan.md`, submodule metadata.
  - Verified: 2026-07-12

- `FINDING/lfm-pp512-cursor-host-oom` — A locked full LFM pp512 run co-located with Cursor failed from host OOM pressure (llama-bench RSS growth / host kill), not NPU device-heap exhaustion; Cursor received SIGBUS in the same cgroup pressure event while NPU utilization stayed small.
  - Scope: Host/cgroup memory failure mode for fullperf under a stressed IDE session; not an NPU allocator or device-heap baseline.
  - Evidence: `benchmarks/results/pr15-8e3ab44-pp512-fullperf-locked-memguard-1gib/`, GitHub issue #17.
  - Verified: 2026-07-12

- `FINDING/lfm-memguard-session-kill-gap` — The LFM MemAvailable guard could report a kill yet leave the bench process group alive long enough under swap/zram thrash for host OOM to win. The durable harness contract is a recorded `setsid` session/PGID kill (SIGTERM→SIGKILL), swap preflight refusal, and a raised MemAvailable floor when swap is already in use (issue #17 / PR #18 merged to develop as `e8fdcd4`).
  - Scope: Host-RSS/swap safety only; not an NPU device-heap limit.
  - Evidence: `benchmarks/lib/lfm_process_guard.sh`, `benchmarks/test_lfm_process_guard.sh`, `docs/benchmarking.md`, `benchmarks/README.md`, GitHub issue #17, PR #18.
  - Verified: 2026-07-12

- `FINDING/lfm-shared-bindings-dispatch-merge-loss` — Develop fullperf at `8cc351f` reported `0 imported / 1296 staged (0.0% no-copy)` vs remediation `880 imported / 416 staged (67.9%)` while still showing the same `shared allocations: 1 unique, 46.00 MiB` and similar stage-in/copy-out wall times. Root cause: tiered-cache merges kept `prepare_inference_shared` / `binding_key` but dropped the `a3b4af0` dispatch contract (`direct_binding_count` attribution, skip copy-out for direct outputs, `StagingAllocation` only for non-direct). PR #19 restores that contract on develop; short attribution (not full pp512) is the next validation gate for #6.
  - Scope: Direct-IVPU binding attribution and direct-output skip; residual staged bounce after restore still tracked by #6.
  - Evidence: `src/vendor/ivpu/session_direct.rs`, `benchmarks/results/develop-8cc351f-pp512-fullperf-locked/`, `benchmarks/results/remediation-pp512-on-stable-locked/`, GitHub issue #6, PR #19.
  - Verified: 2026-07-12

- `FINDING/direct-host-buft-forced-staging` — Under Direct IVPU, `supports_buft` historically accepted every host buffer type (Level Zero–era “shared RAM ⇒ bind without copy”). Direct can only import `SharedAllocationOwner` allocations from the NPU buft, so CPU-host tensors stayed foreign and were staged at dispatch. Remediation sharing-on still showed 416 staged / 880 imported under that contract. PR #22 narrows Direct-only `supports_buft` to our buft and refuses unimportable `alloc_host` fallbacks when shared alloc fails.
  - Scope: Placement/buffer-type policy for Direct zero-copy; not a claim that staged volume is already at remediation or better without a short attribution re-measure.
  - Evidence: `src/backend/shell.rs`, `src/backend/buffer.rs`, `benchmarks/results/remediation-pp512-on-stable-locked/`, GitHub issue #6, PR #22.
  - Verified: 2026-07-12

- `FINDING/offload-skipped-ffn-glue` — `offload_op` historically pulled only `MUL_MAT` / `SSM_CONV` (plus opt-in ShortConv candidates). FFN glue claimed via `supports_op`/`claims_glue` stayed on foreign buffers unless already placed, so matmul anchors and their silu/mul/residual often split into unitary NPU programs. Policy now dependency-pulls claimed FFN glue and layout under an FFN fusible ancestor, refuses non-claimable runtime-weight `MUL_MAT` pulls, and aligns `FORCE_GRAPH` fusible ancestry depth with `supports_region` (12).
  - Scope: Scheduler placement/`should_pull` only; not a measured split/unitary delta until short attribution or locked harness evidence.
  - Evidence: `src/policy.rs`, `src/ops/subgraph.rs`, `src/backend/shell.rs`, GitHub issue #5.
  - Verified: 2026-07-12

- `FINDING/layout-only-overclaim-warmup-abort` — After fail-closed (#32), develop `856b43d` pp16 warmup aborted when a disconnected 1-node `View` partition was claimed then OpenVINO returned `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` (`GGML_STATUS_FAILED`); an adjacent 7-node partition had already run direct-IVPU at 100% no-copy. Fix: reject Metadata/`traits.metadata` in `graph_node_feasible` and layout-only graphs in `graph_feasible` while keeping layout ops on the fused wire allowlist (PR #35).
  - Scope: Standalone layout-only partitions (View/Reshape/Cont/Transpose/Permute); not a claim that fused layout+compute is proven on every LFM split.
  - Evidence: `src/vendor/ivpu/compile.rs`, `benchmarks/results/develop-856b43d-targeted-pp16-debug/`, GitHub issues #5/#8, PR #35.
  - Verified: 2026-07-12

- `FINDING/layout-only-semantic-region-failclosed` — After PR #35/#36, LFM targeted pp16/pp128 r=1 on develop `dfbc35a` aborted warmup (`res=-3`) when a claimed multi-op split partitioned into layout-only semantic regions; `graph_feasible` rejected them and fail-closed returned. Fix (PR #37 @ `5ccffe4`): classify layout-only only when **all** nodes are layout metadata; inside a claimed split that also has lowerable compute, schedule those regions as `PartitionReason::LayoutGlueHost` and host-realize (aliasing View/Reshape/Transpose/Permute no-op; Cont copies) instead of asking OpenVINO to compile them alone. Standalone layout-only splits still refuse (prefer unclaim). Post-fix pp16 smoke shows `layout_glue_host` working; warmup then hits a separate MoE expert-id assert (`FINDING/moe-expert-id-assert-after-layout-glue`).
  - Scope: Execute/partition path for claimed graphs; does not reopen soft-fallback for failed NPU compute compile/exec.
  - Evidence: `src/engine.rs`, `src/graph.rs`, `src/vendor/ivpu/compile.rs`, `.cursor/rules/npu-claim-fallback.mdc`, `benchmarks/results/develop-dfbc35a-targeted-pp16-sharing-on/`, `benchmarks/results/develop-dfbc35a-targeted-pp128-sharing-on/`, `benchmarks/results/develop-5ccffe4-targeted-pp16-sharing-on/`, issues #5/#8, PR #37.
  - Verified: 2026-07-12

- `FINDING/moe-expert-id-assert-after-layout-glue` — After PR #37 cleared layout-only region rejects, develop `5ccffe4` pp16 r=1 sharing-on smoke aborted warmup (exit 134) with `GGML_ASSERT(id >= 0 && id < n_expert)` in llama.cpp `ggml-backend.cpp` after NPU claimed MoE routing (Argsort/GetRows/SumRows/Sigmoid) despite `npu_moe=0`. Root cause: `moe_enabled()` used `ConfigService::is_set`, and `_run_targeted_smoke.sh` always exports `GGML_NPU_MOE=0`, so presence alone falsely enabled MoE; bad expert ids then reached `MUL_MAT_ID` on another backend. Fix: MoE (and shared `is_opt_in_one`) requires exact value `1`; `=0`/unset unclaims MoE-tagged ops.
  - Scope: Env-gate honesty for MoE claim policy under harness `GGML_NPU_MOE=0`; not a claim that `GGML_NPU_MOE=1` routing is correct on NPU.
  - Evidence: `src/host/mod.rs`, `src/vendor/ivpu/compile.rs`, `src/policy.rs`, `benchmarks/results/_run_targeted_smoke.sh`, `benchmarks/results/develop-5ccffe4-targeted-pp16-sharing-on/` (pre-fix abort), `benchmarks/results/develop-0a804c7-targeted-pp16-sharing-on/` (post-fix exit 0; zero Argsort/Sigmoid/SumRows claims), issues #5/#11, PR #38.
  - Verified: 2026-07-12

- `FINDING/shortconv-setrows-attention-gate` — `WireOp::SetRows` was incorrectly tagged
  `traits.attention`, so `graph_feasible` rejected ShortConv/mutable-state write regions unless
  `GGML_NPU_ATTENTION=full`. SetRows is stateful scatter (also used by LFM `cache_r*` recurrent
  writes via CPY/GET_ROWS adjacency), not attention SoftMax/FA/Rope. Under
  `GGML_NPU_LFM_SHORTCONV`, placement now pulls SSM_CONV/SET_ROWS/GET_ROWS/CPY when they sit under
  `.conv.` ancestry or touch `cache_r*` (not `cache_k`/`cache_v`).
  - Scope: Capability/placement for ShortConv state edges (#9); not a claim that one OpenVINO
    program can fuse SSM_CONV+SET_ROWS (Intel NPU compile currently rejects that fusion).
  - Evidence: `src/graph.rs`, `src/policy.rs`, `tests/graph_ir_shortconv_check.c`, GitHub issue #9.
  - Verified: 2026-07-12

- `FINDING/direct-warm-binding-reuse` — Direct resident programs historically keyed the full shared-binding identity into one cache entry, so weightless/glue programs re-staged ELF+HPI+command for every distinct IO set. PR #25 caches artifacts ≤512 KiB by program semantic key (L0 `SG_GRAPHS` analogue), retains a pre-JIT section snapshot, and re-JITs IO bindings in place; larger baked-weight programs keep the binding-set key to avoid host double-keep of weight blobs.
  - Scope: Direct session resident cache / prepare path only; not a measured handoff_begin or tok/s delta until short attribution or locked harness evidence. Device/host async overlap remains intentionally synchronous.
  - Evidence: `src/vendor/ivpu/session_direct.rs`, `src/vendor/ivpu/direct/prepare.rs`, `src/vendor/ivpu/direct/load.rs`, GitHub issue #5, PR #25.
  - Verified: 2026-07-12

## Constraints

- `CONSTRAINT/no-default-full-pp512` — Full LFM pp512 and similarly long multi-sample locked llama-bench runs must not be started by default; prefer tests → Criterion → short attribution/profile → optional `r=1` smoke, and reserve full locked pp512 for explicit requests, acceptance criteria, or tok/s-moving changes after cheaper gates. Do not run locked fullperf inside a memory-/swap-stressed Cursor session; prefer a separate terminal/session and the cheaper ladder first. Respect the global flock; keep shim/`compile_graph_v3` current before NPU full benches; do not re-run only to “confirm.”
  - Evidence: `.cursor/rules/benchmark-invocation.mdc`, `docs/benchmarking.md`, `benchmarks/README.md`, `benchmarks/lib/lfm_process_guard.sh`.
  - Verified: 2026-07-12

- `CONSTRAINT/upstream-ggml-npu-device-type` — Stock `ggml-org/llama.cpp` lacks `GGML_BACKEND_DEVICE_TYPE_NPU` and the external-memory registry ABI. `build.rs` feature-detects both: without NPU the backend defaults to `ACCEL` (and maps `GGML_NPU_DEVICE_CLASS=npu` to `ACCEL`) rather than inventing an enum value; without external memory the registry extension is omitted. Native NPU typing + external memory require MicroPerceptron/llama.cpp. CI uses public `ggml-org` (the private fork is not clonable from Actions without credentials). Issue #13.
  - Evidence: `.github/workflows/ci.yml`, `build.rs`, `src/backend/shell.rs`, `src/backend/mod.rs`, `tests/registration_check.c`, `tests/run_registration_check.sh`, `README.md`.
  - Verified: 2026-07-12

- `CONSTRAINT/shared-memory-promotion` — Generic interop currently proves only allocation identity, access intent, and an owner-controlled host mapping. External handles, visibility, ownership transfer, synchronization, cache maintenance, and queue-family rules stay adapter-owned until at least two implementations share their invariants and failure semantics without convention-based translation.
  - Evidence: `docs/architecture-boundaries.md`, `src/interop.rs`.
  - Verified: 2026-07-12

- `CONSTRAINT/direct-ivpu-release-pins` — Direct IVPU depends on Linux stable accel UAPI plus separately versioned Intel firmware and ELF-loader ABIs. Firmware/loader versions must remain release-pinned and mismatches must be rejected.
  - Evidence: `docs/direct-runtime-foundation.md`, `src/vendor/ivpu/direct/`.
  - Verified: 2026-07-12

## Rejected alternatives

- `REJECTED/moe-enabled-via-is-set` — Do not treat `ConfigService::is_set("GGML_NPU_MOE")` as MoE
  enable. Targeted smokes and docs export `GGML_NPU_MOE=0` to mean off; presence-only gating falsely
  claimed Argsort/Sigmoid/SumRows/MatMulId, corrupted expert ids, and aborted on
  `GGML_ASSERT(id >= 0 && id < n_expert)`.
  - Rationale: Opt-in flags that document `=1` must use exact-value checks (`is_opt_in_one`), matching
    `GGML_NPU_ATTENTION=full` style, not mere env presence.
  - Evidence: `src/host/mod.rs`, `src/vendor/ivpu/compile.rs`, `benchmarks/results/_run_targeted_smoke.sh`,
    `FINDING/moe-expert-id-assert-after-layout-glue`, issues #5/#11.
  - Verified: 2026-07-12

- `REJECTED/invent-npu-device-type-vs-upstream` — Do not define a local `GGML_BACKEND_DEVICE_TYPE_NPU` (or fake external-memory ABI) against stock `ggml-org/llama.cpp` headers to green CI.
  - Rationale: A backend-local enum value the host scheduler does not understand would mis-report device class; use ACCEL fallback + omit external memory when headers lack the ABI, and MicroPerceptron/llama.cpp when native NPU typing is required (issue #13). Pinning CI to the private MicroPerceptron fork was also rejected for Actions: anonymous HTTPS clone fails without credentials.
  - Evidence: `.github/workflows/ci.yml`, `build.rs`, `src/backend/shell.rs`.
  - Verified: 2026-07-12

- `REJECTED/half-crate-for-f16` — Do not switch IEEE F16 constants or intermediates to the `half` crate (or any crate `f16` type). Keep nightly `f16`.
  - Rationale: Nightly `f16` is the objective baseline; `half::f16` collides in naming/API surface and would pull F16 off the NPU interior type.
  - Evidence: `rust-toolchain.toml` / `#![feature(f16)]` in `src/lib.rs`, `src/ops/requant.rs`, `src/ops/ggml_quant.rs`, issue #27.
  - Verified: 2026-07-12

- `REJECTED/avx2-q4-quant-transcode` — Do not ship an AVX2 specialization for Q4_K quant-transcode; production keeps the portable-SIMD baseline (with scalar fallback).
  - Rationale: Authoritative sequential `perf stat` showed explicit AVX2 worse than portable on cycles/element and throughput for Q4_K, so the ISA path is not selected.
  - Evidence: `docs/quant-simd-evaluation.md`, `src/ops/quant_kernel.rs`.
  - Verified: 2026-07-12

- `REJECTED/vnni-quant-transcode` — Do not add AVX-VNNI or AVX-512 VNNI paths for Q4_K/Q6_K quant-transcode.
  - Rationale: Transcoding is bit extraction, shifts, masks, rearrangement, and metadata copies—not integer multiply-dot-accumulate—so VNNI is inapplicable.
  - Evidence: `docs/quant-simd-evaluation.md`.
  - Verified: 2026-07-12

- `REJECTED/project-wide-memory-addr` — Do not represent all memory as a generic integer `memory_addr`. Core uses checked typed address domains; shared host mappings expose `NonNull<u8>` plus length without claiming cross-device visibility.
  - Rationale: Process pointers, device virtual addresses, and foreign handles have different provenance, visibility, and failure rules.
  - Evidence: `docs/architecture-boundaries.md`, `src/address.rs`, `src/interop.rs`.
  - Verified: 2026-07-12

- `REJECTED/project-wide-vfio-contract` — Do not couple generic runtime, compiler, or artifact contracts to VFIO (or any OS handle/API). Direct IVPU resource operations are transport-neutral and the current Linux implementation uses the `intel_vpu` DRM accel UAPI behind an adapter.
  - Rationale: OS transport mechanics do not define compiler artifacts, firmware commands, or portable runtime semantics.
  - Evidence: `docs/direct-runtime-foundation.md`, `src/vendor/ivpu/direct/transport.rs`, `src/vendor/ivpu/direct/linux.rs`.
  - Verified: 2026-07-12

## Benchmark baselines

- `BENCH/lfm-pp512-direct-ivpu-aace217` — LFM2.5 8B.A1B Q4_K Medium pp512 on Lunar Lake, OpenVINO native-artifact compilation plus direct-IVPU submission, full attention placement, flash attention on, direct MoE disabled, scheduler fallback allowed, 8 threads and 3 repetitions: `8.292 ± 0.147 tok/s`, `61.758 ± 1.097 s`; 632 direct dispatches, 6.21 GiB peak RSS, and 3.19 GiB NPU high-water usage.
  - Limitation: All 2,976 bindings bounced, so this is an execution proof/baseline, not zero-copy performance.
  - Evidence: `benchmarks/results/20260712T043711Z-npu/summary.txt`, `benchmarks/results/20260712T043711Z-npu/stderr.txt`.
  - Reproduce: `RUN_NPU=1 MODEL=<model.gguf> MODE=npu PROMPT_TOKENS=512 GEN_TOKENS=0 REPETITIONS=3 THREADS=8 ATTENTION_PLACEMENT=full FLASH_ATTENTION=on GGML_NPU_MOE=0 ./benchmarks/run_lfm_baseline.sh`
  - Verified: 2026-07-12

- `BENCH/lfm-pp512-remediation-sharing-on` — Same LFM pp512 locked remediation with sharing+fusion on: `119.427 ± 6.430 tok/s`, `4.295 ± 0.230 s`; 492 dispatches; scheduler **336 splits/pass**; ~23 unitary NPU programs/pass (mostly 1-node `MUL_MAT`); handoff_begin ~15.1 s and compute/enqueue ~42.9 s across 4 passes; 880 direct / 416 staged bindings. Vulkan full offload remains ~843–849 tok/s (`benchmarks/results/20260711T232859Z-vulkan`).
  - Limitation: Hybrid/NPU proof still ~7× behind Vulkan; staged copies remain material (772 staged_copies / ~17.3 GiB).
  - Evidence: `benchmarks/results/remediation-pp512-on-stable-locked/summary.txt`, sibling `remediation-pp512-off-stable-locked` at `85.164 ± 5.988 tok/s`.
  - Reproduce: locked LFM harness per `docs/benchmarking.md` with sharing/fusion enabled as in the remediation environment.
  - Verified: 2026-07-12

## Pending milestones

- `MILESTONE/capability-contract-complete` — Finish GitHub #8 beyond fail-closed + layout-only
  unclaim: harness summaries that distinguish proven direct-IVPU vs scheduler/Vulkan fallback;
  broader mismatch regression tests (tensor-level supports_op vs compile); quant canonical-fallback
  attribution clarity. Landed slices: shared wire/env gates + fail-loud claimed execution (PR #32);
  Metadata layout-only unclaim (PR #35); agent rule + hermetic contract tests (PR #36,
  `ADR/capability-contract-fail-closed`, `.cursor/rules/npu-claim-fallback.mdc`); host-realize
  layout-only semantic regions inside claimed multi-op splits
  (`FINDING/layout-only-semantic-region-failclosed`).
  - Acceptance: Bench summaries cannot present fallback tok/s as NPU proof; automated tests cover
    supports_op/compile disagreement cases; issue #8 acceptance criteria met with evidence.
  - Evidence: `benchmarks/run_lfm_baseline.sh`, `src/engine.rs`, `src/vendor/ivpu/compile.rs`,
    `docs/coverage-gates.md`, issue #8.

- `MILESTONE/shared-memory-interop` — Promote a shared-memory interop contract only after at least two adapters prove matching ownership, visibility, ordering, cache-maintenance, synchronization, import/export, and failure semantics.
  - Acceptance: Cross-adapter known-answer tests cover zero/copy/shared bindings, lifetime and aliasing, synchronization, unsupported negotiation, and deterministic cleanup without generic foreign-handle types.
  - Evidence: `docs/architecture-boundaries.md`, `docs/compiler-migration-plan.md`.

- `MILESTONE/scheduler-handoff-reduction` — Reduce llama.cpp scheduler splits and uneconomic unitary NPU programs so handoff is no longer first-order on LFM pp512 (GitHub #5; parent #4). First slice (FFN-shaped residual glue + `npu_program` attribution, plus FFN activation-on-`src[1]` fix) landed on `main` via PR #15 and is on `develop` via PR #20 (`95f4baa` / `a4c8530` / merge `b2ab6ad`). PR #22 raises the default unitary offload floor to `4× MIN_OFFLOAD_WORK` and deepens ShortConv ancestry (6→12) when `GGML_NPU_LFM_SHORTCONV` is on. PR #23 dependency-pulls claimed FFN glue / FFN-span layout, refuses non-claimable matmul pulls, and sets `FORCE_GRAPH` fusible depth 12 (`FINDING/offload-skipped-ffn-glue`). PR #25 adds Direct warm binding reuse for small similar programs (`FINDING/direct-warm-binding-reuse`). PR #35/#36 lock prefer-unclaim for standalone layout-only Metadata; layout-only regions inside claimed splits are host-realized as glue (`FINDING/layout-only-semantic-region-failclosed`). Next: short attribution once swap is calm; safe async overlap only after per-submission completion objects.
  - Acceptance: Measurable drop in splits/pass and/or unitary NPU programs vs `BENCH/lfm-pp512-remediation-sharing-on`, with attribution hooks and tok/s movement toward Vulkan under the locked harness.
  - Evidence: `src/policy.rs`, `src/ops/subgraph.rs`, `src/engine.rs`, `src/metrics.rs`, `src/vendor/ivpu/session_direct.rs`, issue #5, PR #15, PR #20, PR #22, PR #23, PR #25.

- `MILESTONE/residual-shared-binding-bounce` — Restore accurate no-copy binding attribution and reduce residual staged copies on LFM (GitHub #6). Develop `8cc351f` fullperf showed a false 0% no-copy line from merge-lost dispatch accounting; PR #19 restores the contract. PR #22 addresses a real residual cause: Direct no longer advertises foreign host bufts as bindable without copy, and refuses unimportable host alloc fallbacks.
  - Acceptance: With shared bindings enabled, profile imported/staged counters and bounce allocation events match direct-vs-staged reality; residual staged volume moves toward remediation or better without requiring a locked fullperf as the first gate (binding counters / short attribution suffice).
  - Evidence: `src/vendor/ivpu/session_direct.rs`, `src/vendor/ivpu/direct/prepare.rs`, `src/backend/shell.rs`, `src/backend/buffer.rs`, issue #6, PR #19, PR #22.

- `MILESTONE/retire-level-zero-compat` — Keep Level Zero as the compatibility runtime and differential oracle until direct submission is qualified beyond the tiny artifact.
  - Acceptance: The same artifact/model corpus passes direct and Level Zero with matching binding metadata and numerics; direct relocation/HPI/firmware checks, resident accounting, import/copy behavior, synchronization, device-loss handling, reset recovery, and required operation coverage are proven.
  - Evidence: `docs/direct-runtime-foundation.md`, `docs/compiler-migration-plan.md`.

- `MILESTONE/neutral-compiler-providers` — Replace compiler-specific frontend coupling with dynamically loaded, versioned, neutral providers, starting with Intel and later adding AMD and Qualcomm adapters behind the same semantic request/artifact envelope.
  - Acceptance: Capability/compile agreement, stable diagnostics, ownership/error ABI tests, deterministic pinned builds, artifact validation, and target-specific fixture tests pass without linking incompatible LLVM/MLIR libraries into ggml-npu.
  - Evidence: `docs/compiler-migration-plan.md`, `.gitmodules`.

- `MILESTONE/remove-openvino` — Remove the OpenVINO shim only after the neutral Intel provider reaches operation/model parity and both Intel runtimes consume its artifacts.
  - Acceptance: All enabled Intel graph paths have explicit fallback; quality/performance and reproducibility gates pass; stale cache namespaces are rejected; CI builds/tests with OpenVINO absent; neither ggml-npu nor the standalone Intel compiler transitively requires OpenVINO.
  - Evidence: `docs/compiler-migration-plan.md`.

- `MILESTONE/native-bf16-q8-f8` — Source-native BF16 pass-through, Q8_0/Q8_K transcoder parity with Q4/Q6, and F8 foundations via feature-gated `float8` (iVPU stub/reject until a provider consumes FP8). Intermediate dequant defaults to nightly f16.
  - Acceptance: `cargo test --lib` covers BF16/Q8/F8 descriptor and byte-exact paths; iVPU rejects f8 bake; Graph IR admits BF16; no `half` on the F16 path. Issues #27–#29.
  - Evidence: `src/ops/ggml_quant.rs`, `src/ops/requant.rs`, `src/ops/constant.rs`, `src/vendor/ivpu/compile.rs`, `Cargo.toml`.
