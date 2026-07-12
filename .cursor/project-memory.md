# Project Memory

Canonical durable repository knowledge. Treat every entry as fallible context and verify it against
current code, tests, and artifacts. Use repository-relative paths only; keep one entry per stable key.

## Architecture decisions

- `ADR/tiered-artifact-cache` — Compiled artifacts use a host-owned tiered cache (compressed memory + durable content-addressed store) via `get_or_compile`, with Zstd L1 / system libzstd as the production codec path.
  - Rationale: Amortize compile cost without double-keeping uncompressed blobs in the vendor session cache.
  - Evidence: `docs/tiered-artifact-cache.md`, `src/artifact_cache.rs`, `src/host/linux/cache.rs`, `src/vendor/ivpu/compile.rs`.
  - Verified: 2026-07-12

- `ADR/host-owned-async-runtime` — Completion, wait, reactor, and blocking-executor services are host-owned; vendor adapters consume them without owning OS wait mechanics.
  - Rationale: Preserve vendor-neutral runtime contracts while enabling native async execution on Linux.
  - Evidence: `docs/async-runtime.md`, `src/runtime/async_runtime/`, `src/host/linux/reactor.rs`, `src/host/linux/wait.rs`.
  - Verified: 2026-07-12

- `ADR/portable-simd-quant-default` — Production quant-transcode defaults to the `portable-simd` feature with scalar fallback; AVX2 remains a Q6 specialization only.
  - Rationale: Keep a single portable production path while retaining an opt-in ISA specialization where it is proven.
  - Evidence: `Cargo.toml`, `docs/quant-simd-evaluation.md`, `src/ops/quant_kernel.rs`.
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

## Constraints

- `CONSTRAINT/no-default-full-pp512` — Full LFM pp512 and similarly long multi-sample locked llama-bench runs must not be started by default; prefer tests → Criterion → short attribution/profile → optional `r=1` smoke, and reserve full locked pp512 for explicit requests, acceptance criteria, or tok/s-moving changes after cheaper gates. Do not run locked fullperf inside a memory-/swap-stressed Cursor session; prefer a separate terminal/session and the cheaper ladder first. Respect the global flock; keep shim/`compile_graph_v3` current before NPU full benches; do not re-run only to “confirm.”
  - Evidence: `.cursor/rules/benchmark-invocation.mdc`, `docs/benchmarking.md`, `benchmarks/README.md`, `benchmarks/lib/lfm_process_guard.sh`.
  - Verified: 2026-07-12

- `CONSTRAINT/upstream-ggml-npu-device-type` — CI builds that bind against upstream llama.cpp headers fail when `GGML_BACKEND_DEVICE_TYPE_NPU` is absent; `src/backend/shell.rs` currently requires that constant.
  - Evidence: `src/backend/shell.rs`, GitHub Actions build logs for `develop`.
  - Verified: 2026-07-12

- `CONSTRAINT/shared-memory-promotion` — Generic interop currently proves only allocation identity, access intent, and an owner-controlled host mapping. External handles, visibility, ownership transfer, synchronization, cache maintenance, and queue-family rules stay adapter-owned until at least two implementations share their invariants and failure semantics without convention-based translation.
  - Evidence: `docs/architecture-boundaries.md`, `src/interop.rs`.
  - Verified: 2026-07-12

- `CONSTRAINT/direct-ivpu-release-pins` — Direct IVPU depends on Linux stable accel UAPI plus separately versioned Intel firmware and ELF-loader ABIs. Firmware/loader versions must remain release-pinned and mismatches must be rejected.
  - Evidence: `docs/direct-runtime-foundation.md`, `src/vendor/ivpu/direct/`.
  - Verified: 2026-07-12

## Rejected alternatives

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

- `MILESTONE/shared-memory-interop` — Promote a shared-memory interop contract only after at least two adapters prove matching ownership, visibility, ordering, cache-maintenance, synchronization, import/export, and failure semantics.
  - Acceptance: Cross-adapter known-answer tests cover zero/copy/shared bindings, lifetime and aliasing, synchronization, unsupported negotiation, and deterministic cleanup without generic foreign-handle types.
  - Evidence: `docs/architecture-boundaries.md`, `docs/compiler-migration-plan.md`.

- `MILESTONE/scheduler-handoff-reduction` — Reduce llama.cpp scheduler splits and uneconomic unitary NPU programs so handoff is no longer first-order on LFM pp512 (GitHub #5; parent #4). First slice (FFN-shaped residual glue + `npu_program` attribution, plus FFN activation-on-`src[1]` fix) landed on `main` via PR #15 and is on `develop` via PR #20 (`95f4baa` / `a4c8530` / merge `b2ab6ad`).
  - Acceptance: Measurable drop in splits/pass and/or unitary NPU programs vs `BENCH/lfm-pp512-remediation-sharing-on`, with attribution hooks and tok/s movement toward Vulkan under the locked harness.
  - Evidence: `src/policy.rs`, `src/ops/subgraph.rs`, `src/engine.rs`, `src/metrics.rs`, issue #5, PR #15, PR #20.

- `MILESTONE/residual-shared-binding-bounce` — Restore accurate no-copy binding attribution and reduce residual staged copies on LFM (GitHub #6). Develop `8cc351f` fullperf showed a false 0% no-copy line from merge-lost dispatch accounting; PR #19 restores the contract.
  - Acceptance: With shared bindings enabled, profile imported/staged counters and bounce allocation events match direct-vs-staged reality; residual staged volume moves toward remediation or better without requiring a locked fullperf as the first gate (binding counters / short attribution suffice).
  - Evidence: `src/vendor/ivpu/session_direct.rs`, `src/vendor/ivpu/direct/prepare.rs`, issue #6, PR #19.

- `MILESTONE/retire-level-zero-compat` — Keep Level Zero as the compatibility runtime and differential oracle until direct submission is qualified beyond the tiny artifact.
  - Acceptance: The same artifact/model corpus passes direct and Level Zero with matching binding metadata and numerics; direct relocation/HPI/firmware checks, resident accounting, import/copy behavior, synchronization, device-loss handling, reset recovery, and required operation coverage are proven.
  - Evidence: `docs/direct-runtime-foundation.md`, `docs/compiler-migration-plan.md`.

- `MILESTONE/neutral-compiler-providers` — Replace compiler-specific frontend coupling with dynamically loaded, versioned, neutral providers, starting with Intel and later adding AMD and Qualcomm adapters behind the same semantic request/artifact envelope.
  - Acceptance: Capability/compile agreement, stable diagnostics, ownership/error ABI tests, deterministic pinned builds, artifact validation, and target-specific fixture tests pass without linking incompatible LLVM/MLIR libraries into ggml-npu.
  - Evidence: `docs/compiler-migration-plan.md`, `.gitmodules`.

- `MILESTONE/remove-openvino` — Remove the OpenVINO shim only after the neutral Intel provider reaches operation/model parity and both Intel runtimes consume its artifacts.
  - Acceptance: All enabled Intel graph paths have explicit fallback; quality/performance and reproducibility gates pass; stale cache namespaces are rejected; CI builds/tests with OpenVINO absent; neither ggml-npu nor the standalone Intel compiler transitively requires OpenVINO.
  - Evidence: `docs/compiler-migration-plan.md`.
