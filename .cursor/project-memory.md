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

- `ADR/sched-region-coalesce-rust-first` — Prefill same-backend region coalesce (island
  absorption / uneconomic-cut heuristics) lives in `no_std` Rust
  (`src/sched/region_coalesce.rs`) and is exported as
  `ggml_backend_sched_region_coalesce` via registry `get_proc_address`. The llama.cpp
  scheduler packs POD rows and applies results behind `GGML_SCHED_REGION_COALESCE`
  (default off), applying NPU/same-buft targets always and other targets only
  when committed buffers are supported by the destination (buffer-safe Vulkan
  sandwiches), then re-syncing views to `view_src`. Claim profitability stays in
  `offload_op` / policy; the pass must not invent NPU claims; Rust leaves `VIEW` rows
  untouched for the host sync.
  - Rationale: Safety and bitset/interval analysis belong in Rust; LTO/same-process load
    mitigates FFI cost vs split overhead. Avoids growing opaque C++ scheduler logic.
    Host apply guards prevent Vulkan buffer inconsistency (pp128 SEGV).
  - Evidence: `docs/sched-region-coalesce.md`, `src/sched/region_coalesce.rs`,
    `src/backend/shell.rs`, PR #41/#44/#50; companion llama.cpp hook PRs
    MicroPerceptron/llama.cpp#10/#11/#12.
  - Verified: 2026-07-13

- `ADR/sched-weighted-cut-cost` — Weighted cut-cost coalesce (#48) extends the Rust
  scheduler surface: `src/sched/cut_cost.rs` + `ggml_backend_sched_region_coalesce_v2`
  packs `nbytes` and `ggml_backend_dev_type` kinds; gated by `GGML_SCHED_CUT_COST=1`
  (requires coalesce ≥1). Heuristic edge cost = nbytes × backend-pair handoff factor.
  **Anti-unitary grow-first**: absorb feasible foreign islands onto an *existing* NPU
  neighbor (classic sandwich or asymmetric `NPU|foreign|other`), never invent NPU when
  neither neighbor is NPU; then classic sandwiches. Preserve **fat** NPU islands
  (≥3 ops or ≥64 KiB); dissolve thin unitaries / 2-op scraps; +2 light / 2–3× heavy
  `max_island` on NPU growth edges. Process counters via
  `ggml_backend_sched_cut_cost_stats[_reset]`. `core::simd` batches nbytes×factor under
  `portable-simd`. llama.cpp stays thin packing + buft-safe apply + optional stats dump
  (llama.cpp PR #13 / anti-unitary companion). Does not invent NPU claims or change MoE
  `is_opt_in_one` / Argsort unclaim; does not widen Direct `supports_buft`.
  - Rationale: Length-only coalesce can dissolve feasible NPU sandwiches into Vulkan;
    cut costs + selective absorb-into-NPU move toward fat NPU_HOST islands under Direct’s
    honest buft contract without recreating preserve-all ~412 pp16 disp or PR #22 failure.
  - Evidence: `docs/sched-region-coalesce.md`, `src/sched/cut_cost.rs`,
    `src/sched/region_coalesce.rs`, PR #51, issue #48, `FINDING/sched-anti-unitary-npu-absorb`.
  - Verified: 2026-07-13

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

- `ADR/rust-first-implementation` — Prefer new logic in Rust (including graph/scheduler analysis and
  policy). Interop with ggml/llama.cpp is a thin C ABI only; do not grow C/C++ for algorithms that
  can live in Rust. Fat LTO (`Cargo.toml` `[profile.release]`) amortizes FFI cost; Rust safety
  remains the priority. OpenVINO/shim C++ stays only where the vendor compiler API forces it.
  - Rationale: Keep algorithms and ownership in the `no_std` Rust cdylib; confine C++ to the
    OpenVINO compile shim rather than duplicating policy across the FFI boundary.
  - Evidence: `Cargo.toml`, `docs/restructure-decision.md`, `docs/architecture-boundaries.md`,
    `docs/compiler-migration-plan.md`, `.cursor/rules/rust-first-implementation.mdc`, `shim/`.
  - Verified: 2026-07-12

- `ADR/direct-npu-host-residency` — Direct IVPU `supports_buft` accepts only our `NPU_HOST` buft;
  `buft_alloc_buffer` fails closed without `SharedAllocationOwner`; `owns_tensor` requires a shared
  owner. Duty cycle grows via llama.cpp `GGML_SCHED_NPU_CLAIM_STAGE=1` (default off): claim with
  `supports_op`, stage foreign inputs into NPU_HOST — never widen foreign `supports_buft`. Selective
  allowlist stays empty/gated (`GGML_NPU_IMPORT_FOREIGN_HOST=1` insufficient alone). MoE untouched.
  - Rationale: L0’s broad host contract inflated affinity while Direct cannot import foreign host
    pointers; claim+stage restores L0-shaped seeding without PR #22’s lie.
  - Evidence: `docs/direct-buft-residency.md`, `src/backend/buft.rs`, companion llama.cpp#15,
    issue #6, PR #22, PR #52.
  - Verified: 2026-07-13

## Verified findings


- `FINDING/sched-cut-cost-fat-npu-preserve` — On develop tip with PR #51 + llama.cpp#13,
  LFM sharing-on ShortConv=1 MoE=0 ATTENTION=full FlexiBLAS hybrid, `COALESCE=1`
  `CUT_COST=1` (fat-NPU preserve) vs coalesce-only: pp128 keeps **10** disp, unitary
  60%→**40%**, avg nodes 3.20→**3.60**, no-copy 93.8%→**100%**, soft tok/s ~25.9→**~38.3**;
  pp16 raises NPU program count 108→**324** while unitary 55.6%→**21.0%**, avg nodes
  1.85→**2.82**, no-copy 82.7%→**100%** (soft tok/s ~36.5→~32.9). MoE=1 pp16 + cut-cost
  exit 0. Blanket preserve-all-NPU (pre-fat gate) regressed pp16 to ~412 disp.
  - Scope: Targeted r=1 attribution; not locked pp512; gitignored result dirs under
    `benchmarks/results/develop-68b7e8d-targeted-pp*-coa1-cut*`.
  - Evidence: `src/sched/cut_cost.rs`, `docs/sched-region-coalesce.md`, PR #51,
    llama.cpp#13, issue #48, `BENCH/lfm-cut-cost-coa1-targeted`.
  - Verified: 2026-07-13

- `FINDING/sched-anti-unitary-npu-absorb` — Anti-unitary cut-cost follow-on (asymmetric
  absorb onto existing NPU neighbors, fat gate ≥3 ops or ≥64 KiB, +2 light max_island,
  `cut_cost_stats` ABI) on `feat/sched-anti-unitary-npu-absorb` matches MVP fat-preserve
  LFM shape under Direct honest buft: pp16 **324**/21.0%/2.82/100%; pp128 **10**/40%/3.60/100%;
  MoE=1 pp16 exit 0. Does not close the pp16 108→324 disp residual vs coalesce-only.
  Aggressive light `max_island` 2× regressed pp16 to **412**/27.2% (same failure class as
  preserve-all). L0 vs Direct pp512 north-star (L0 ~266 tok/s / 488 disp / 5.12 avg vs
  Direct ~101 / 20 / 3.20) remains a buft/duty-cycle gap — not addressed by widening
  `supports_buft`.
  - Scope: Targeted r=1; not locked pp512; policy + unit tests + short smoke only.
  - Evidence: `src/sched/cut_cost.rs`, `src/sched/region_coalesce.rs`,
    `docs/sched-region-coalesce.md`, `BENCH/lfm-anti-unitary-coa1-targeted`,
    `REJECTED/sched-cut-cost-aggressive-light-island-boost`, issues #48/#5.
  - Verified: 2026-07-13

- `FINDING/coverage-prs-gated-surface-not-duty-cycle` — After merges #45/#46/#47 on develop
  `8ec283e`, targeted LFM sharing-on ShortConv=1 MoE=0 `GGML_SCHED_REGION_COALESCE=1` with
  llama.cpp master PR#11 guard did not raise NPU duty cycle vs prior develop: pp128 stays
  **12** dispatches / 50% unitary / avg nodes ~3 / 100% no-copy (direct-ivpu); pp16 stays
  ~400–412 disp (not the unguarded `ab3eceb` 66-disp absorb). Coverage expanded gated surface
  (hermetic ShortConv, SoftMax/FA behind `ATTENTION=full`, MoE=1 claim path); separate MoE=1
  pp16 still aborts on `GGML_ASSERT(id >= 0 && id < n_expert)` after Argsort/GetRows claim.
  - Scope: Targeted r=1 attribution only; not locked pp512 tok/s; soft tok/s not comparable to
    `5db8513` pp512 (~307 tok/s, 24 disp).
  - Evidence: `benchmarks/results/develop-8ec283e-targeted-pp16-sharing-on-coalesce-attn-default/`,
    `benchmarks/results/develop-8ec283e-targeted-pp128-sharing-on-coalesce-attn-default/`,
    `benchmarks/results/develop-8ec283e-targeted-pp16-sharing-on-coalesce-moe1-attn-default/`,
    issues #4/#5/#11, PRs #45/#46/#47.
  - Verified: 2026-07-13

- `FINDING/sched-region-coalesce-pp128-vulkan-segv` — Unguarded `GGML_SCHED_REGION_COALESCE=1`
  island absorb into Vulkan left incompatible buffers inside Vulkan splits; LFM pp128
  SIGSEGV in `ggml_vk_build_graph` (pp16 survived). Interim fix: llama.cpp apply only
  NPU/same-buft + `view_src` sync (PR MicroPerceptron/llama.cpp#11); Rust leaves views
  untouched (ggml-npu PR #44). Nearest-neighbor view paint rejected (broke weight views,
  still crashed). Superseded for duty-cycle by buffer-safe apply
  (`FINDING/sched-coalesce-buft-safe-duty-cycle`).
  - Scope: Prefill coalesce enablement safety under the NPU-only guard era; does not
    describe post–llama.cpp#12 behavior.
  - Evidence: `docs/sched-region-coalesce.md`, `src/sched/region_coalesce.rs`,
    `benchmarks/results/develop-ab3eceb-targeted-pp128-sharing-on-coalesce/`,
    `benchmarks/results/fix-coalesce-view-pp128-sharing-on-coalesce/`, GitHub #5, PRs #44 /
    llama.cpp#11.
  - Verified: 2026-07-13

- `FINDING/sched-coalesce-buft-safe-duty-cycle` — Buffer-safe coalesce apply (llama.cpp#12;
  ggml-npu docs PR #50) allows non-NPU island absorbs only when every already-committed
  buffer that would share the destination is `ggml_backend_supports_buft`-OK (unallocated
  temps remain gallocr-safe). On develop tip with that apply, LFM sharing-on ShortConv=1
  MoE=0 ATTENTION unset `COALESCE=1` raises duty cycle vs coalesce-off / prior NPU-only
  guard without pp128 SEGV: pp16 **108** disp (vs 400 off) soft ~38 tok/s; pp128 **8**
  disp (vs 12 off) soft ~71 tok/s. Does not fully restore unguarded `ab3eceb` pp16 66-disp
  class; some staging returns (pp16 82.7% no-copy).
  - Scope: Targeted r=1 attribution; not locked pp512; MoE `is_opt_in_one` unchanged.
  - Evidence: `benchmarks/results/develop-8ec283e-targeted-pp16-sharing-on-coa1-buftsafe/`,
    `benchmarks/results/develop-8ec283e-targeted-pp16-sharing-on-coa0-buftsafe/`,
    `benchmarks/results/develop-8ec283e-targeted-pp128-sharing-on-coa1-buftsafe/`,
    `benchmarks/results/develop-8ec283e-targeted-pp128-sharing-on-coa0-buftsafe/`,
    `docs/sched-region-coalesce.md`, GitHub #5, ggml-npu PR #50, llama.cpp#12.
  - Verified: 2026-07-13

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

- `FINDING/lfm-memguard-session-kill-gap` — The LFM MemAvailable guard could report a kill yet leave the bench process group alive long enough under swap/zram thrash for host OOM to win. The durable harness contract is a recorded `setsid` session/PGID kill (SIGTERM→SIGKILL), swap preflight refusal, and a raised MemAvailable floor when swap is already in use (PR #18 / `e8fdcd4`). GitHub #17 closed: written acceptance met via harness + `benchmarks/test_lfm_process_guard.sh` (no full pp512 required).
  - Scope: Host-RSS/swap safety only; not an NPU device-heap limit.
  - Evidence: `benchmarks/lib/lfm_process_guard.sh`, `benchmarks/test_lfm_process_guard.sh`, `docs/benchmarking.md`, `benchmarks/README.md`, GitHub issue #17, PR #18.
  - Verified: 2026-07-13

- `FINDING/lfm-shared-bindings-dispatch-merge-loss` — Develop fullperf at `8cc351f` reported `0 imported / 1296 staged (0.0% no-copy)` vs remediation `880 imported / 416 staged (67.9%)` while still showing the same `shared allocations: 1 unique, 46.00 MiB` and similar stage-in/copy-out wall times. Root cause: tiered-cache merges kept `prepare_inference_shared` / `binding_key` but dropped the `a3b4af0` dispatch contract (`direct_binding_count` attribution, skip copy-out for direct outputs, `StagingAllocation` only for non-direct). PR #19 restores that contract on develop; short attribution (not full pp512) is the next validation gate for #6.
  - Scope: Direct-IVPU binding attribution and direct-output skip; residual staged bounce after restore still tracked by #6.
  - Evidence: `src/vendor/ivpu/session_direct.rs`, `benchmarks/results/develop-8cc351f-pp512-fullperf-locked/`, `benchmarks/results/remediation-pp512-on-stable-locked/`, GitHub issue #6, PR #19.
  - Verified: 2026-07-12

- `FINDING/direct-host-buft-forced-staging` — Under Direct IVPU, `supports_buft` historically accepted every host buffer type (Level Zero–era “shared RAM ⇒ bind without copy”). Direct can only import `SharedAllocationOwner` allocations from the NPU buft, so CPU-host tensors stayed foreign and were staged at dispatch. Remediation sharing-on still showed 416 staged / 880 imported under that contract. PR #22 narrows Direct-only `supports_buft` to our buft and refuses unimportable `alloc_host` fallbacks when shared alloc fails. Follow-on `ADR/direct-npu-host-residency` adds profile probe/residency counters and hermetic policy tests (`foreign_host_refused` vs L0 broad host).
  - Scope: Placement/buffer-type policy for Direct zero-copy; not a claim that staged volume is already at remediation or better without a short attribution re-measure.
  - Evidence: `src/backend/shell.rs`, `src/backend/buffer.rs`, `src/backend/buft.rs`,
    `docs/direct-buft-residency.md`, `benchmarks/results/remediation-pp512-on-stable-locked/`,
    GitHub issue #6, PR #22.
  - Verified: 2026-07-13

- `FINDING/direct-buft-claim-stage-duty-cycle` — `GGML_SCHED_NPU_CLAIM_STAGE=1` (llama.cpp#15)
  seeds NPU from host-weight ops without widening Direct `supports_buft`. LFM pp16 r=1 Direct:
  coalesce=0 claim=0→1 raises disp **412→482** and `our_shared` **2242→2514** (`dishonest_npu_host=0`);
  coalesce=1 alone regresses 108→94 (re-absorb into Vulkan); coalesce=1+cut-cost raises **348→394**.
  - Scope: Targeted r=1; not locked pp512; cut-cost measured on anti-unitary binary.
  - Evidence: `docs/direct-buft-residency.md`, `BENCH/lfm-claim-stage-pp16-targeted`, PR #52,
    llama.cpp#15, issues #5/#6.
  - Verified: 2026-07-13

- `FINDING/offload-skipped-ffn-glue` — `offload_op` historically pulled only `MUL_MAT` / `SSM_CONV` (plus opt-in ShortConv candidates). FFN glue claimed via `supports_op`/`claims_glue` stayed on foreign buffers unless already placed, so matmul anchors and their silu/mul/residual often split into unitary NPU programs. Policy now dependency-pulls claimed FFN glue and layout under an FFN fusible ancestor, refuses non-claimable runtime-weight `MUL_MAT` pulls, and aligns `FORCE_GRAPH` fusible ancestry depth with `supports_region` (12).
  - Scope: Scheduler placement/`should_pull` only; not a measured split/unitary delta until short attribution or locked harness evidence.
  - Evidence: `src/policy.rs`, `src/ops/subgraph.rs`, `src/backend/shell.rs`, GitHub issue #5.
  - Verified: 2026-07-12

- `FINDING/prefill-const-weight-match-node-overrefuse` — Post-unclaim/MoE-off pp128/pp512 on `5db8513` kept proven Direct but only tiny Mul/Add/GetRows/Clamp programs (avg ~3 nodes, ~0.09 GiB NPU) while pp16 DEBUG still showed ShortConv `MatMul`+`SsmConv` spans. Root cause in profitability: `should_pull` refused all `!is_claimable_matmul` (`match_node`), which also drops **const-weight** prefill matmuls whose activations are non-contiguous after Views. Prefill coalesce fixes pull on `is_const_weight_matmul`, lower the unitary floor for prefill-sized static projections (`M>=32`), enlarge FusibleOnly neighborhoods under ShortConv/static spans (not attention-named orphans), and claim FFN-named prefill RMSNorm glue — without reopening MoE `is_opt_in_one` or layout-only `supports_op`.
  - Scope: Rust `policy`/`subgraph` placement only; NPU dispatch/bytes recovery still needs short attribution or locked evidence after merge.
  - Evidence: `src/policy.rs`, `src/ops/subgraph.rs`, `benchmarks/results/develop-5db8513-pp512-fullperf-locked/`, `benchmarks/results/develop-5db8513-targeted-pp128-sharing-on/`, `benchmarks/results/develop-0a804c7-targeted-pp16-sharing-on/`, GitHub #5.
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
  - Scope: Env-gate honesty for MoE claim policy under harness `GGML_NPU_MOE=0`. Follow-on #11 coverage for `MOE=1` is `FINDING/moe-routed-bank-gather-contract`.
  - Evidence: `src/host/mod.rs`, `src/vendor/ivpu/compile.rs`, `src/policy.rs`, `benchmarks/results/_run_targeted_smoke.sh`, `benchmarks/results/develop-5ccffe4-targeted-pp16-sharing-on/` (pre-fix abort), `benchmarks/results/develop-0a804c7-targeted-pp16-sharing-on/` (post-fix exit 0; zero Argsort/Sigmoid/SumRows claims), issues #5/#11, PR #38.
  - Verified: 2026-07-12

- `FINDING/moe1-argsort-intermediate-id-corruption` — With exact `GGML_NPU_MOE=1` on develop `8ec283e`, NPU claimed fused MoE routing including Argsort; TopK ids stayed device intermediates and were not copied out before host View aliases fed `MUL_MAT_ID` on another backend → exit 134 `GGML_ASSERT(id >= 0 && id < n_expert)`. Sigmoid alone was fine; GetRows/SumRows with host-produced ids were fine. Interim fix (PR #49): unclaim Argsort and MatMulId under `MOE=1`. Superseded for Argsort by `FINDING/moe-argsort-id-writeback-reclaim` (PR #60); MatMulId remains unclaimed.
  - Scope: Historical root cause for MOE=1 expert-id abort; claim policy before writeback.
  - Evidence: `src/vendor/ivpu/compile.rs`, `docs/coverage-gates.md`, `benchmarks/results/develop-8ec283e-targeted-pp16-sharing-on-coalesce-moe1-attn-default/` (pre-fix abort), `benchmarks/results/develop-8ec283e-targeted-pp16-sharing-on-moe1-failclosed/` (PR #49 exit 0), issue #11, PR #49 (`4403a82`).
  - Verified: 2026-07-13

- `FINDING/moe-argsort-id-writeback-reclaim` — Under `GGML_NPU_MOE=1`, Argsort is reclaimable when i32 TopK ids are forced as ggml-visible graph outputs: `capture` and `partition_range` mark Argsort outputs even if View/GetRows consume them in-partition, so copy-out materializes host buffers before foreign `MUL_MAT_ID`. MatMulId stays **supports_op-unclaimed** (Vulkan/host experts). Still exact `is_opt_in_one`. pp16 `MOE=1` t=6 smoke exit 0 with Argsort claimed and MatMulId zero claims.
  - Scope: MoE id writeback + claim policy; not model-scale MatMulId / packed-bank tok/s.
  - Evidence: `src/graph.rs`, `src/vendor/ivpu/compile.rs`, `docs/coverage-gates.md`,
    `benchmarks/results/develop-8fa9bf9-targeted-pp16-sharing-on-moe1-argsort-writeback/`,
    issue #11, PR #60.
  - Verified: 2026-07-13

- `FINDING/moe-routed-bank-gather-contract` — Under exact `GGML_NPU_MOE=1`, Sigmoid/SumRows and Argsort (with ggml-visible id writeback) are claimable when compile-ready. MatMulId stays **supports_op-unclaimed** — Vulkan/host owns routed experts until model-scale evidence (shim may still lower fused Argsort→MatMulId with ids internal). Float f16/bf16 expert-bank packing and packed Q4/Q6/Q8 requant contracts remain for when MatMulId is re-enabled; packed source-native gather stays refused. Output-head-scale dense MatMul stays unclaimed via `expanded_constant_budget` unless `GGML_NPU_ALLOW_GIANT_CONSTANTS`.
  - Scope: Claim policy + shim e2e; not a locked LFM full-model routed-expert tok/s proof.
  - Evidence: `src/ops/ggml_quant.rs`, `src/vendor/ivpu/compile.rs`, `src/graph.rs`,
    `shim/ov_compile.cpp`, `tests/coverage_matrix.csv`, `docs/coverage-gates.md`, issue #11,
    PRs #49/#60.
  - Verified: 2026-07-13

- `FINDING/shortconv-setrows-attention-gate` — `WireOp::SetRows` was incorrectly tagged
  `traits.attention`, so `graph_feasible` rejected ShortConv/mutable-state write regions unless
  `GGML_NPU_ATTENTION=full`. SetRows is stateful scatter (KV / generic), not SoftMax/FA. LFM
  ShortConv recurrent writes use **`ggml_cpy` into a view of `cache_r*`** (not SetRows; see
  `llama.cpp` `models/lfm2.cpp`). Under `GGML_NPU_LFM_SHORTCONV`, placement pulls
  SSM_CONV/SET_ROWS/GET_ROWS/CPY under `.conv.` ancestry or `cache_r*` (including `view_src`
  chains when the CPY node name omits `cache_r`). Connected gather→SSM_CONV→SetRows **and**
  LFM gather→CPY(`cache_r`)→SSM_CONV stay one semantic region; when Intel refuses
  SSM_CONV+state-write fusion, the engine peels `feasibility_limit` adjacent NPU programs and
  never host-realizes those stateful edges as `layout_glue_host`.
  - Scope: Capability/placement/partition contract for ShortConv state edges (#9); not a claim
    that one OpenVINO program can fuse SSM_CONV+SET_ROWS/CPY, nor that Direct places LFM CPY
    writeback on NPU without `cache_r` NPU_HOST residency (view_src follows host buft).
  - Evidence: `src/graph.rs`, `src/policy.rs`, `src/engine.rs`, `tests/coverage_matrix.csv`,
    `docs/coverage-gates.md`, `tests/graph_ir_shortconv_check.c`, GitHub issue #9, PRs #34/#45/#58.
  - Verified: 2026-07-13

- `FINDING/shortconv-lfm-cpy-direct-buft-gap` — On Direct with honest `supports_buft` (NPU_HOST
  only), LFM pp16 ShortConv smokes show GetRows+SsmConv on NPU but **zero** `Copy`/`SetRows`
  claims: `ggml_cpy` into `cache_r` is a view whose backend follows the CPU recurrent buffer, so
  the scheduler keeps the write off NPU unless claim+stage / residency moves `cache_r` onto
  NPU_HOST (#6/#52). Hermetic peel for claimed CPY+SSM_CONV remains preferred over
  `layout_glue_host`; do not invent OV fusion.
  - Scope: Direct placement of LFM state writeback under buft-safe coalesce; not a locked pp512.
  - Evidence: `benchmarks/results/develop-125b9f8-targeted-pp16-sharing-on-shortconv-cpy/`,
    `FINDING/shortconv-setrows-attention-gate`, issue #9, PR #58.
  - Verified: 2026-07-13

- `FINDING/prefill-qkv-rope-not-attention-gate` — `WireOp::Rope` and `WireOp::Permute` were tagged
  `traits.attention`, so default env rejected fused prefill QKV+RoPE regions unless
  `GGML_NPU_ATTENTION=full` (which also admits SoftMax/FA). RoPE modes 0|2 (LFM NEOX=2) are
  QKV-adjacent; Permute is layout metadata. PR #39 untags both; SoftMax/FlashAttention and
  runtime-weight matmul stay behind `ATTENTION=full`. Policy pulls QKV-adjacent RoPE, qk RMSNorm,
  and `cache_k`/`cache_v` SetRows/CPY/GET_ROWS for region continuity (distinct from ShortConv
  `cache_r*`).
  - Scope: Capability + `should_pull` for prefill QKV continuity (#10/#5); not a claim that
    SoftMax/FA or full attention-core fusion is default-on or measured for tok/s.
  - Evidence: `src/graph.rs`, `src/policy.rs`, `src/vendor/ivpu/compile.rs`,
    `tests/coverage_matrix.csv`, GitHub issue #10, PR #39.
  - Verified: 2026-07-12


- `FINDING/attention-runtime-core-opt-in` — SoftMax, FlashAttention, and runtime-weight MatMul
  (KQ/KQV) stay behind exact `GGML_NPU_ATTENTION=full` (not `=1`/presence). Supported SoftMax:
  `max_bias==0`, arity 1–2; FlashAttention: `max_bias==0` and `logit_softcap==0`, arity 3–4, GQA
  when query heads divisible by KV heads. Const-weight MatMul / RoPE / KV continuity remain
  default-claimable (PR #39). Under `full`, policy pulls FA as Anchor and SoftMax under an
  anchor, and admits runtime-weight MatMul past the const-weight-only pull refuse (general work
  floor). Capability uses shared `env_policy_admits_gated`; fail-closed unclaim when not `full`.
  - Scope: Contract + policy + shim SoftMax numerical in `graph_ir_shortconv_check`; not a claim
    that ATTENTION=full reduces LFM splits or moves tok/s without measured attribution.
  - Evidence: `src/policy.rs`, `src/vendor/ivpu/compile.rs`, `src/graph.rs`, `docs/coverage-gates.md`,
    `tests/coverage_matrix.csv`, `tests/graph_ir_shortconv_check.c`, GitHub #10.
  - Verified: 2026-07-13

- `FINDING/direct-warm-binding-reuse` — Direct resident programs historically keyed the full shared-binding identity into one cache entry, so weightless/glue programs re-staged ELF+HPI+command for every distinct IO set. PR #25 caches artifacts ≤512 KiB by program semantic key (L0 `SG_GRAPHS` analogue), retains a pre-JIT section snapshot, and re-JITs IO bindings in place; larger baked-weight programs keep the binding-set key to avoid host double-keep of weight blobs.
  - Scope: Direct session resident cache / prepare path only; not a measured handoff_begin or tok/s delta until short attribution or locked harness evidence. Device/host async overlap remains intentionally synchronous.
  - Evidence: `src/vendor/ivpu/session_direct.rs`, `src/vendor/ivpu/direct/prepare.rs`, `src/vendor/ivpu/direct/load.rs`, GitHub issue #5, PR #25.
  - Verified: 2026-07-12

## Constraints

- `CONSTRAINT/no-default-full-pp512` — Full LFM pp512 and similarly long multi-sample locked llama-bench runs must not be started by default; prefer tests → Criterion → short attribution/profile → optional `r=1` smoke, and reserve full locked pp512 for explicit requests, acceptance criteria, or tok/s-moving changes after cheaper gates. Do not run locked fullperf inside a memory-/swap-stressed Cursor session; prefer a separate terminal/session and the cheaper ladder first. Respect the global flock; keep shim/`compile_graph_v3` current before NPU full benches; do not re-run only to “confirm.”
  - Evidence: `.cursor/rules/benchmark-invocation.mdc`, `docs/benchmarking.md`, `benchmarks/README.md`, `benchmarks/lib/lfm_process_guard.sh`.
  - Verified: 2026-07-12

- `CONSTRAINT/bench-default-threads-6` — LFM harness default is `THREADS=6` (llama-bench `-t`, and on CPU modes `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `GOTO_NUM_THREADS` / `MKL_NUM_THREADS`). Do not default to `nproc`; full logical-core counts saturate shared Lunar Lake hosts and inflate variance. Override explicitly when reproducing historical 8-thread citations.
  - Evidence: `benchmarks/run_lfm_baseline.sh`, `docs/benchmarking.md`, `.cursor/rules/benchmark-invocation.mdc`.
  - Verified: 2026-07-13

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

- `REJECTED/sched-cut-cost-preserve-all-npu-islands` — Do not refuse every NPU→non-NPU
  island absorb under cut-cost. Preserving tiny unitary NPU sandwiches undid buft-safe
  coalesce duty-cycle (pp16 ~108→~412 disp). Keep fat islands only (≥3 ops or ≥64 KiB).
  - Rationale: NPU-aware scheduling must not recreate the coalesce-off unitary explosion.
  - Evidence: `FINDING/sched-cut-cost-fat-npu-preserve`, `src/sched/cut_cost.rs`, PR #51.
  - Verified: 2026-07-13

- `REJECTED/sched-cut-cost-aggressive-light-island-boost` — Do not 2× light-edge
  `effective_max_island` for NPU growth under cut-cost. On LFM pp16 that raised disp
  324→**412** and unitary 21%→27.2% (preserve-all failure class). Keep modest +2 on
  light edges; reserve 2–3× for ≥256 KiB / ≥1 MiB boundaries only.
  - Rationale: Anti-unitary absorb must not recreate the coalesce-off / preserve-all
    dispatch explosion while chasing avg-nodes.
  - Evidence: `FINDING/sched-anti-unitary-npu-absorb`,
    `benchmarks/results/develop-7461e97-targeted-pp16-sharing-on-coa1-cut1-antiu2/`,
    `src/sched/cut_cost.rs`.
  - Verified: 2026-07-13

- `REJECTED/moe1-claim-argsort-before-id-writeback` — Do not claim Argsort under `GGML_NPU_MOE=1`
  until i32 expert ids are ggml-visible outputs (not device-only intermediates). Claim-then-View
  corrupted `MUL_MAT_ID` on another backend.
  - Rationale: Prefer unclaim over claim-then-abort / garbage ids (#11); shim fused Argsort→MatMulId
    with internal ids remains allowed.
  - Evidence: `FINDING/moe1-argsort-intermediate-id-corruption`, `src/vendor/ivpu/compile.rs`, PR #49.
  - Verified: 2026-07-13

- `REJECTED/sched-coalesce-nearest-view-paint` — Do not paint coalesce `VIEW` rows to the nearest
  preceding non-view backend in Rust. That broke weight views (buffers often unset at pin time)
  and still SIGSEGV’d on pp128; host must sync views to `view_src` after guarded apply.
  - Rationale: ggml invariant is view↔view_src backend identity, not split-neighbor color.
  - Evidence: `FINDING/sched-region-coalesce-pp128-vulkan-segv`, `docs/sched-region-coalesce.md`,
    PR #44 / llama.cpp#11.
  - Verified: 2026-07-13

- `REJECTED/sched-coalesce-core-in-ggml-backend-cpp` — Do not implement bitmat/interval
  same-backend region coalesce inside llama.cpp `ggml-backend.cpp`. Keep a thin C hook
  only; algorithm stays in ggml-npu Rust.
  - Rationale: User/product constraint prefers Rust for analysis safety; C++ scheduler
    remains packing + split cutting.
  - Evidence: `ADR/sched-region-coalesce-rust-first`, `docs/sched-region-coalesce.md`, PR #41.
  - Verified: 2026-07-12

- `REJECTED/blind-widen-direct-supports-buft-all-host` — Do not re-accept every host buft under
  Direct to match Level Zero placement. That recreates owned-but-unimportable tensors and forced
  staging (`FINDING/direct-host-buft-forced-staging`). Any widen must be a selective, hermetically
  proven importable-buft allowlist (`importable_foreign_host_allowlist_enabled` stays false).
  - Rationale: Scheduler affinity must match bindable storage; L0 duty-cycle is not a license to lie.
  - Evidence: `ADR/direct-npu-host-residency`, `docs/direct-buft-residency.md`, `src/backend/buft.rs`,
    issue #6, PR #22.
  - Verified: 2026-07-13

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


- `BENCH/lfm-claim-stage-pp16-targeted` — Claim+stage smokes with llama.cpp#15; LFM Q4_K_M pp16
  r=1 sharing-on ShortConv=1 MoE=0 ATTENTION=full FlexiBLAS, flock + `ALLOW_HIGH_SWAP=1`,
  `GGML_NPU_PROFILE=1`: coalesce=0 claim=0 — 412/27.2%/2242 our_shared; claim=1 —
  **482**/32.4%/**2514**, dishonest=0. coalesce=1+cut=1 claim=0 — 348/32.2%; claim=1 —
  **394**/39.6% (anti-unitary .so).
  - Limitation: Soft r=1; not locked pp512; coalesce-only+claim regresses NPU disp.
  - Evidence: `benchmarks/results/develop-7870cf7-targeted-pp16-sharing-on-coa0-cs*/`,
    `docs/direct-buft-residency.md`, llama.cpp#15, PR #52.
  - Reproduce: `GGML_SCHED_NPU_CLAIM_STAGE=1` ± `GGML_SCHED_REGION_COALESCE` / `CUT_COST`
    with `_run_targeted_smoke.sh` PROMPT_TOKENS=16.
  - Verified: 2026-07-13

- `BENCH/lfm-cut-cost-coa1-targeted` — Branch `feat/sched-weighted-cut-cost` (`6f65a61`) +
  llama.cpp `d110b3736` (PR #13), LFM Q4_K_M targeted r=1 sharing-on ShortConv=1 MoE=0
  ATTENTION=full FlexiBLAS hybrid, flock + `ALLOW_HIGH_SWAP=1`:
  coalesce=1 cut=0 — pp16 108/55.6%/1.85/82.7%/~36.5; pp128 10/60%/3.20/93.8%/~25.9.
  coalesce=1 cut=1 (fat preserve) — pp16 324/21.0%/2.82/100%/~32.9; pp128 10/40%/3.60/100%/~38.3.
  Separate MoE=1 pp16 cut=1: exit 0.
  - Limitation: Soft r=1 only; not locked pp512; pp16 dispatch count rises vs coalesce-only
    while unitary%/no-copy/avg-nodes improve.
  - Evidence: `benchmarks/results/develop-68b7e8d-targeted-pp16-sharing-on-coa1-cut0/`,
    `benchmarks/results/develop-68b7e8d-targeted-pp16-sharing-on-coa1-cut1b/`,
    `benchmarks/results/develop-68b7e8d-targeted-pp128-sharing-on-coa1-cut0/`,
    `benchmarks/results/develop-68b7e8d-targeted-pp128-sharing-on-coa1-cut1b/`,
    `benchmarks/results/develop-68b7e8d-targeted-pp16-sharing-on-coa1-cut1b-moe1/`,
    issue #48, PR #51, llama.cpp#13.
  - Reproduce: `GGML_SCHED_REGION_COALESCE=1` ± `GGML_SCHED_CUT_COST=1` with
    `benchmarks/results/_run_targeted_smoke.sh` (PROMPT_TOKENS=16|128).
  - Verified: 2026-07-13

- `BENCH/lfm-anti-unitary-coa1-targeted` — Branch `feat/sched-anti-unitary-npu-absorb`
  (anti-unitary absorb + fat≥3 + stats) + llama.cpp cut-cost/stats hook, LFM Q4_K_M
  targeted r=1 sharing-on ShortConv=1 MoE=0 ATTENTION=full FlexiBLAS hybrid,
  flock + `ALLOW_HIGH_SWAP=1`:
  coalesce=1 cut=0 — pp16 108/55.6%/1.85/82.7%/~29.7; pp128 10/60%/3.20/93.8%/~31.5.
  coalesce=1 cut=1 — pp16 324/21.0%/2.82/100%/~31.1; pp128 10/40%/3.60/100%/~33.0.
  MoE=1 pp16 cut=1: exit 0 (412/27.2%/2.97/100%). Matches MVP cut1b dispatch shape;
  soft tok/s not comparable across runs. Aggressive light max_island 2× (antiu2) rejected
  at pp16 412/27.2%.
  - Limitation: Soft r=1 only; not locked pp512; pp16 108→324 residual remains.
  - Evidence: `benchmarks/results/develop-7461e97-targeted-pp*-sharing-on-coa1-cut*-antiu/`,
    `benchmarks/results/develop-7461e97-targeted-pp16-sharing-on-coa1-cut1-antiu-moe1/`,
    `FINDING/sched-anti-unitary-npu-absorb`, issue #48.
  - Reproduce: same as `BENCH/lfm-cut-cost-coa1-targeted` on the anti-unitary branch.
  - Verified: 2026-07-13

- `BENCH/lfm-pp512-68b7e8d-flexiblas-coa1` — Develop `68b7e8d` locked LFM Q4_K_M pp512 r=3
  t=8 sharing-on ShortConv=1 MoE=0 ATTENTION=full FA on `GGML_SCHED_REGION_COALESCE=1`,
  llama.cpp `2a365b784` (buft-safe #12) `build-interop` with **Vulkan + FlexiBLAS**
  (`GGML_BLAS=ON`, FlexiBLAS → OPENBLAS-OPENMP): **100.814 ± 1.905 tok/s**,
  5.080 ± 0.095 s; exit 0; backends **BLAS,Vulkan,NPU**; NPU profile 20 disp /
  60% unitary / avg nodes 3.20 / 93.8% no-copy / ~0.09 GiB NPU; direct-ivpu
  baseline-eligible. Vs memory: Vulkan ~843–849; remediation ~119; post-unclaim
  `5db8513` ~307 (24 disp).
  - Limitation: First locked combo of buft-safe coalesce + FlexiBLAS in the
    hybrid binary; no BLAS-off control on the same tip; `ALLOW_HIGH_SWAP=1`.
  - Evidence: `benchmarks/results/develop-68b7e8d-pp512-fullperf-locked-sharing-on-coa1-flexiblas/`,
    `benchmarks/results/summary-develop-68b7e8d-pp512-flexiblas.txt`, issues #4/#5.
  - Reproduce: rebuild `llama.cpp/build-interop` with `-DGGML_VULKAN=ON -DGGML_BLAS=ON
    -DGGML_BLAS_VENDOR=FlexiBLAS`; flock + `benchmarks/results/_run_pp512_fullperf.sh`
    with `GGML_SCHED_REGION_COALESCE=1` and FlexiBLAS env as in the artifact
    `blas-provenance.txt`.
  - Verified: 2026-07-13

- `BENCH/lfm-pp512-68b7e8d-level-zero-flexiblas-coa1` — Locked LFM Q4_K_M pp512 r=3 t=8
  Level Zero compat session vs Direct tip above: same model/FA/ATTENTION=full/
  ShortConv=1/MoE=0/`COALESCE=1`/FlexiBLAS+Vulkan hybrid `llama-bench` binary
  (`2a365b784`). L0 cdylib built from clean `68b7e8d` with
  `--no-default-features --features level-zero-compat,portable-simd` (no
  `direct-ivpu`; `session_level_zero` aliased as `session`). Result:
  **266.083 ± 89.573 tok/s**, 2.123 ± 0.887 s; exit 0; `execution: level-zero`
  baseline-eligible; profile **488** disp / 28.7% unitary / avg nodes 5.12 /
  52.2% no-copy / ~1.20 GiB NPU; samples 162.7 / 317.7 / 317.9 (cold first
  rep). ~2.64× Direct FlexiBLAS tip tok/s; much higher NPU duty cycle than
  Direct’s 20 disp / 0.09 GiB (L0 still uses broad host `supports_buft`).
  - Limitation: Artifact dir named `develop-7461e97-…-level-zero` from repo
    HEAD at run; `.so` source is `68b7e8d`. Cold L0 cache (~766 s compile in
    profile). Placement/binding contract differs from Direct (not an apples-
    to-apples duty-cycle match). Does **not** retire L0.
  - Evidence: `benchmarks/results/develop-7461e97-pp512-fullperf-locked-sharing-on-coa1-flexiblas-level-zero/`,
    `benchmarks/results/summary-develop-7461e97-pp512-level-zero.txt`,
    `benchmarks/results/_run_pp512_level_zero.sh`, `target-level-zero/BUILD.txt`,
    `MILESTONE/retire-level-zero-compat`.
  - Reproduce: flock + `FLEXIBLAS=OPENBLAS-OPENMP`
    `./benchmarks/results/_run_pp512_level_zero.sh` after building
    `target-level-zero/release/libggml_npu.so` as above.
  - Verified: 2026-07-13

- `BENCH/lfm-post-coverage-8ec283e-targeted` — Develop `8ec283e` (post #45/#46/#47) LFM Q4_K_M
  targeted r=1 sharing-on ShortConv=1 coalesce=1, llama.cpp `9e777bc77` (PR#11 guard), direct-ivpu:
  (1) ATTENTION unset pp16: 400 disp, 28.0% unitary, 100% no-copy, soft 26.1 tok/s;
  pp128: 12 disp, 50% unitary, 100% no-copy, soft 64.6 tok/s.
  (2) Separate ATTENTION=full: pp16 412/27.2%/100%/19.3; pp128 12/50%/100%/72.3.
  (3) Separate MoE=1 pp16: exit 134 expert-id assert.
  Compare: `0a804c7` pp16 ~410/26.8%/30.1; `5db8513` pp128 12/50%/101; `5db8513` pp512 24 disp
  ~307 tok/s; unguarded `ab3eceb` coalesce pp16 66 disp not restored under PR#11.
  - Limitation: Soft r=1 only; OpenVINO stderr can pollute jsonl; not locked fullperf.
  - Evidence: `benchmarks/results/develop-8ec283e-targeted-*/`, GitHub #5 comment on tip `8ec283e`.
  - Verified: 2026-07-13

- `BENCH/lfm-pp128-coalesce-guarded-fix` — After ggml-npu PR #44 + llama.cpp PR #11, LFM Q4_K_M
  pp128 r=1 sharing-on ShortConv=1 MoE=0 `GGML_SCHED_REGION_COALESCE=1`: exit 0, 12 NPU
  dispatches, 50% unitary, avg nodes 3.00, soft ~83.2 tok/s (no SEGV). Matches coalesce-off
  dispatch shape on this prompt; prior unguarded coalesce crashed.
  - Limitation: Soft r=1 only; not locked pp512; NPU-only apply does not restore pp16’s prior
    66-disp Vulkan-absorb win. Superseded for duty-cycle by
    `BENCH/lfm-buft-safe-coalesce-targeted`.
  - Evidence: `benchmarks/results/fix-coalesce-view-pp128-sharing-on-coalesce/`, GitHub #5.
  - Verified: 2026-07-13

- `BENCH/lfm-buft-safe-coalesce-targeted` — LFM Q4_K_M targeted r=1 sharing-on ShortConv=1
  MoE=0 ATTENTION unset, buffer-safe coalesce apply (llama.cpp#12 built atop `9e777bc77`),
  direct-ivpu: coalesce=1 pp16 **108** disp / 55.6% unitary / avg nodes 1.85 / 82.7% no-copy /
  soft 38.05 tok/s; pp128 **8** / 75% / 3.25 / 91.7% / 71.34. coalesce=0 controls: pp16 400 /
  28% / 2.42 / 100% / 28.04; pp128 12 / 50% / 3.00 / 100% / 74.33. No SEGV.
  - Limitation: Soft r=1 only; not locked pp512; pp16 does not fully match unguarded
    `ab3eceb` 66-disp; pp128 soft tok/s slightly below coalesce-off despite fewer dispatches.
  - Evidence: `benchmarks/results/develop-8ec283e-targeted-pp{16,128}-sharing-on-coa{0,1}-buftsafe/`,
    GitHub #5, llama.cpp#12, ggml-npu PR #50.
  - Verified: 2026-07-13

- `BENCH/lfm-shortconv-cpy-pp16-targeted` — PR #58 tip, LFM Q4_K_M pp16 r=1 sharing-on
  ShortConv=1 MoE=0 ATTENTION=full `COALESCE=1`, Direct, `-t 6`, flock + `ALLOW_HIGH_SWAP=1`:
  exit 0; **108** disp / 55.6% unitary / avg nodes 1.85; soft ~59.5 tok/s;
  `mutable_state_boundary=0`; SsmConv/GetRows claimed; **Copy=0 / SetRows=0** on NPU
  (`FINDING/shortconv-lfm-cpy-direct-buft-gap`).
  - Limitation: Soft r=1 only; not locked pp512; does not place LFM CPY writeback on Direct.
  - Evidence: `benchmarks/results/develop-125b9f8-targeted-pp16-sharing-on-shortconv-cpy/`,
    issue #9, PR #58.
  - Reproduce: `THREADS=6 PROMPT_TOKENS=16 GGML_SCHED_REGION_COALESCE=1` targeted smoke with
    ShortConv=1 MoE=0 ATTENTION=full.
  - Verified: 2026-07-13

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

- `BENCH/lfm-moe1-argsort-writeback-pp16` — PR #60 tip `8fa9bf9`, LFM Q4_K_M pp16 r=1
  sharing-on ShortConv=1 `MOE=1` ATTENTION=full FlexiBLAS hybrid, `-t 6` /
  `OMP_NUM_THREADS=6`, flock + `ALLOW_HIGH_SWAP=1`: **exit 0**; backends BLAS,Vulkan,NPU;
  Argsort claimed (DEBUG wire hits); MatMulId **0**; Sigmoid claimed; no expert-id assert;
  soft ~32 tok/s.
  - Limitation: Soft r=1 only; not locked pp512; not a MatMulId reclaim proof.
  - Evidence: `benchmarks/results/develop-8fa9bf9-targeted-pp16-sharing-on-moe1-argsort-writeback/`,
    issue #11, PR #60.
  - Reproduce: `GGML_NPU_MOE=1 THREADS=6 OMP_NUM_THREADS=6` targeted pp16 with current
    `libggml_npu.so` after PR #60.
  - Verified: 2026-07-13

- `BENCH/lfm-tip-5c1acb3-coa-cut-claim-matrix` — Merged ggml-npu `5c1acb3` + llama.cpp
  `a95d49e3c`, LFM Q4_K_M targeted r=1 sharing-on ShortConv=1 MoE=0 ATTENTION=full
  FlexiBLAS+Vulkan+Direct, `-t 6` / `OMP_NUM_THREADS=6`, flock + `ALLOW_HIGH_SWAP=1`,
  `backend_git_head` from loaded `.so` BUILD.txt. All eight short cells exit 0:
  pp16 coa1/cut0/cs0 **108**/55.6%/1.85/82.7%/~33; coa0/cs1 **482**/32.4%/2.29/100%/~19;
  coa1/cut1/cs1 **394**/39.6%/2.43/96%/~22; coa1/cut1/cs0 **348**/32.2%/2.61/95.7%/~24.
  pp128 coa1/cut0/cs0 **10**/60%/3.20/93.8%/~40; coa0/cs1 **516**/30.6%/2.41/100%/~65;
  coa1/cut1/cs1 **428**/27.1%/2.59/99.8%/~89; coa1/cut1/cs0 **20**/40%/3.20/96.8%/~44.
  Pre-fix locked pp512 r=3 coa1+cut1+cs1 warmup failed (`res = -3` /
  `BindingMismatch`); see `FINDING/claim-stage-elf-pad-binding-mismatch`.
  - Limitation: Soft r=1 only for greens; pre-fix pp512 not baseline-eligible.
  - Evidence: `benchmarks/results/develop-5c1acb3-targeted-pp{16,128}-sharing-on-coa*-cut*-cs*/`,
    `benchmarks/results/develop-5c1acb3-pp512-fullperf-locked-sharing-on-coa1-cut1-cs1-t6/`,
    issues #5/#6.
  - Reproduce: `THREADS=6` ± `GGML_SCHED_REGION_COALESCE` / `CUT_COST` / `NPU_CLAIM_STAGE`
    via `_run_targeted_smoke.sh`; pp512 via `_run_pp512_fullperf.sh` with coa1 cut1 claim1.
  - Verified: 2026-07-13

- `FINDING/claim-stage-elf-pad-binding-mismatch` — With `GGML_SCHED_NPU_CLAIM_STAGE=1`,
  Direct shared-import of exact-sized NPU_HOST tensors aborted pp512 warmup as
  `PrepareError::BindingMismatch` → `GGML_STATUS_FAILED` (`res = -3`) on the fused
  MatMul+Swiglu subgraph (`M=512`). OpenVINO ELF `allocation_bytes` can pad past
  logical `nbytes`; `offset + allocation_bytes` then overruns the owner (observed
  logical 14680064 vs alloc 14705872 near the end of a 60686336-byte shared buffer).
  Trigger is **claim-stage**, not coalesce/cut-cost: pp512 r=1 passes with cs0
  (coa1 cut0/cut1) and fails with cs1 (coa0 or coa1). Fix: stage-copy into a fresh
  BO when the shared view cannot cover the padded section (keep no-copy when it can);
  unit tests lock the cover predicate.
  - Scope: Direct prepare path; llama.cpp claim-stage host unchanged.
  - Evidence: `src/vendor/ivpu/direct/prepare.rs`,
    `benchmarks/results/develop-5c1acb3-targeted-pp512-coa*-bisect/`,
    `benchmarks/results/develop-5c1acb3-targeted-pp512-coa1-cut1-cs1-warmup-fix/`
    (exit 0, 470 disp), issues #5/#6.
  - Verified: 2026-07-13

## Pending milestones

- `MILESTONE/moe-coverage-model-scale` — Finish GitHub #11 beyond Argsort writeback reclaim
  (PR #60): model-scale MatMulId / packed expert banks (or keep Vulkan/host experts as the
  durable contract with acceptance met); Sigmoid/SumRows + Argsort-with-writeback already
  opt-in under `MOE=1`. Keep default harness `MOE=0` unclaimed; never reintroduce `is_set`
  MoE enable; never reclaim Argsort without ggml-visible id writeback.
  - Acceptance: Documented MoE strategy with passing routed-expert evidence when MatMulId is
    reclaimed (or explicit permanent Vulkan/host experts); no silent packed source-native
    gather; output-head remains explicitly unclaimed or validated; issue #11 acceptance met
    without requiring full locked pp512 as the first gate.
  - Evidence: `docs/coverage-gates.md`, `FINDING/moe-argsort-id-writeback-reclaim`,
    `FINDING/moe-routed-bank-gather-contract`, issue #11, PR #60.

- `MILESTONE/capability-contract-complete` — Finish GitHub #8 beyond fail-closed + layout-only
  unclaim: broader mismatch regression tests (tensor-level supports_op vs compile); quant
  canonical-fallback attribution clarity; more placement paths under attributable proof.
  Landed slices: shared wire/env gates + fail-loud claimed execution (PR #32); Metadata
  layout-only unclaim (PR #35); agent rule + hermetic contract tests (PR #36,
  `ADR/capability-contract-fail-closed`, `.cursor/rules/npu-claim-fallback.mdc`); host-realize
  layout-only semantic regions inside claimed multi-op splits
  (`FINDING/layout-only-semantic-region-failclosed`); hermetic WireOp claim ⊆ coverage matrix +
  `--require-baseline-proof` (PR #33; GitHub #12 closed).
  - Acceptance: Bench summaries cannot present fallback tok/s as NPU proof; automated tests cover
    supports_op/compile disagreement cases; issue #8 acceptance criteria met with evidence.
  - Evidence: `benchmarks/run_lfm_baseline.sh`, `src/engine.rs`, `src/vendor/ivpu/compile.rs`,
    `docs/coverage-gates.md`, `tests/coverage_matrix.csv`, issues #8/#12, PRs #32/#33/#35/#36/#37.

- `MILESTONE/shared-memory-interop` — Promote a shared-memory interop contract only after at least two adapters prove matching ownership, visibility, ordering, cache-maintenance, synchronization, import/export, and failure semantics.
  - Acceptance: Cross-adapter known-answer tests cover zero/copy/shared bindings, lifetime and aliasing, synchronization, unsupported negotiation, and deterministic cleanup without generic foreign-handle types.
  - Evidence: `docs/architecture-boundaries.md`, `docs/compiler-migration-plan.md`.

- `MILESTONE/scheduler-handoff-reduction` — Reduce llama.cpp scheduler splits and uneconomic unitary NPU programs so handoff is no longer first-order on LFM pp512 (GitHub #5; parent #4). First slice (FFN-shaped residual glue + `npu_program` attribution, plus FFN activation-on-`src[1]` fix) landed on `main` via PR #15 and is on `develop` via PR #20 (`95f4baa` / `a4c8530` / merge `b2ab6ad`). PR #22 raises the default unitary offload floor to `4× MIN_OFFLOAD_WORK` and deepens ShortConv ancestry (6→12) when `GGML_NPU_LFM_SHORTCONV` is on. PR #23 dependency-pulls claimed FFN glue / FFN-span layout, refuses non-claimable matmul pulls, and sets `FORCE_GRAPH` fusible depth 12 (`FINDING/offload-skipped-ffn-glue`). PR #25 adds Direct warm binding reuse for small similar programs (`FINDING/direct-warm-binding-reuse`). PR #35/#36 lock prefer-unclaim for standalone layout-only Metadata; layout-only regions inside claimed splits are host-realized as glue (`FINDING/layout-only-semantic-region-failclosed`). PR #39 untags RoPE/Permute from the SoftMax/FA attention gate and pulls QKV-adjacent RoPE/qk-norm/KV writes for prefill continuity (`FINDING/prefill-qkv-rope-not-attention-gate`). SoftMax/FA/runtime-weight KQ opt-in under exact `ATTENTION=full` is contracted (`FINDING/attention-runtime-core-opt-in`). PR #41 adds Rust same-backend region coalesce ABI (`ADR/rust-first-sched-region-coalesce`). Prefill-region coalesce (`FINDING/prefill-const-weight-match-node-overrefuse`) restores const-weight / ShortConv / FFN-named prefill pulls after the post-#38 surface collapse.   PR #44 + llama.cpp#11 guard coalesce apply (`FINDING/sched-region-coalesce-pp128-vulkan-segv`). Post-#45/#46/#47 attribution on `8ec283e` (`FINDING/coverage-prs-gated-surface-not-duty-cycle`, `BENCH/lfm-post-coverage-8ec283e-targeted`) confirmed gated surface only under NPU-only apply.   Buffer-safe Vulkan-sandwich absorbs (llama.cpp#12 / ggml-npu PR #50; `FINDING/sched-coalesce-buft-safe-duty-cycle`, `BENCH/lfm-buft-safe-coalesce-targeted`) restore short-smoke duty cycle (pp16 400→108 disp; pp128 12→8; no SEGV) without full unguarded 66-disp restore. Locked tip `68b7e8d` with FlexiBLAS+coa1 (`BENCH/lfm-pp512-68b7e8d-flexiblas-coa1`) is ~101 tok/s / 20 disp / 0.09 GiB — near remediation tok/s, not a restore of `5db8513` ~307. Next: close remaining gap toward `ab3eceb` 66-disp / locked pp512 handoff vs remediation and toward Vulkan; placement that grows proven NPU bytes vs `5db8513` 24/0.09 GiB; safe async overlap only after per-submission completion objects. Weighted cut-cost follow-on is `MILESTONE/sched-weighted-cut-cost` (#48).
  - Acceptance: Measurable drop in splits/pass and/or unitary NPU programs vs `BENCH/lfm-pp512-remediation-sharing-on`, **or** (post-unclaim) a clearly larger proven NPU duty cycle (dispatches/bytes/avg nodes) vs `5db8513` 24 / 0.09 GiB without soft-success; attribution hooks and tok/s movement toward Vulkan under the locked harness.
  - Evidence: `src/policy.rs`, `src/ops/subgraph.rs`, `src/engine.rs`, `src/metrics.rs`, `src/vendor/ivpu/session_direct.rs`, `src/sched/region_coalesce.rs`, `src/sched/cut_cost.rs`, issue #5, PR #15, PR #20, PR #22, PR #23, PR #25, PR #39, PR #41, PR #50, PR #51, llama.cpp#12/#13.

- `MILESTONE/sched-weighted-cut-cost` — Weighted DAG / bitmat-inspired cut-cost scheduler on top of landed same-backend region coalesce (GitHub #48; follow-on to #5 coalesce, contributes toward #4). **MVP gated** in PR #51 + llama.cpp#13 (`ADR/sched-weighted-cut-cost`, `FINDING/sched-cut-cost-fat-npu-preserve`, `BENCH/lfm-cut-cost-coa1-targeted`). **Anti-unitary absorb** follow-on (`FINDING/sched-anti-unitary-npu-absorb`, `BENCH/lfm-anti-unitary-coa1-targeted`): grow onto existing NPU neighbors (incl. asymmetric), fat≥3 preserve, stats ABI; LFM shape matches MVP without beating pp16 108→324 residual. Remaining vs full #48: bit-row reachability / non-linear maximal regions; calibrated factors; pp16 dispatch-count reduction without regressing unitary%/no-copy or recreating ~412 preserve-all; optional default-on after more evidence. Do not widen Direct `supports_buft` to chase L0 duty cycle.
  - Acceptance: Measurable drop in handoff bytes and/or splits on short attribution/profile vs coalesce-only; higher NPU duty cycle without inventing unsupported claims; Rust-first in `src/sched/` with thin `get_proc_address` POD export; attribution counters for cut cost / handoff / split shape. Full locked pp512 only when explicitly warranted.
  - Evidence: GitHub issue #48, PR #51, llama.cpp#13, `src/sched/cut_cost.rs`, `src/sched/region_coalesce.rs`, `docs/sched-region-coalesce.md`, issue #5.

- `MILESTONE/residual-shared-binding-bounce` — Restore accurate no-copy binding attribution and reduce residual staged copies on LFM (GitHub #6). Develop `8cc351f` fullperf showed a false 0% no-copy line from merge-lost dispatch accounting; PR #19 restores the contract. PR #22 addresses a real residual cause: Direct no longer advertises foreign host bufts as bindable without copy, and refuses unimportable host alloc fallbacks. PR #52 + llama.cpp#15 add residency attribution and claim+stage (`GGML_SCHED_NPU_CLAIM_STAGE=1`) for L0-shaped duty cycle under coalesce=0 / cut-cost; selective allowlist remains gated off.
  - Acceptance: With shared bindings enabled, profile imported/staged counters and bounce allocation events match direct-vs-staged reality; residual staged volume moves toward remediation or better without requiring a locked fullperf as the first gate (binding counters / short attribution suffice). Material reduction of `foreign_host_refused` → staged only via proven importable allowlist, never blind widen.
  - Evidence: `src/vendor/ivpu/session_direct.rs`, `src/vendor/ivpu/direct/prepare.rs`,
    `src/backend/shell.rs`, `src/backend/buffer.rs`, `src/backend/buft.rs`,
    `docs/direct-buft-residency.md`, issue #6, PR #19, PR #22, PR #52, llama.cpp#15.

- `MILESTONE/retire-level-zero-compat` — Keep Level Zero as the compatibility runtime and differential oracle until direct submission is qualified beyond the tiny artifact. One locked LFM L0 pp512 baseline now exists (`BENCH/lfm-pp512-68b7e8d-level-zero-flexiblas-coa1`); that is evidence for parity comparison only — **not** acceptance to delete L0.
  - Acceptance: The same artifact/model corpus passes direct and Level Zero with matching binding metadata and numerics; direct relocation/HPI/firmware checks, resident accounting, import/copy behavior, synchronization, device-loss handling, reset recovery, and required operation coverage are proven.
  - Evidence: `docs/direct-runtime-foundation.md`, `docs/compiler-migration-plan.md`,
    `BENCH/lfm-pp512-68b7e8d-level-zero-flexiblas-coa1`.

- `MILESTONE/neutral-compiler-providers` — Replace compiler-specific frontend coupling with dynamically loaded, versioned, neutral providers, starting with Intel and later adding AMD and Qualcomm adapters behind the same semantic request/artifact envelope.
  - Acceptance: Capability/compile agreement, stable diagnostics, ownership/error ABI tests, deterministic pinned builds, artifact validation, and target-specific fixture tests pass without linking incompatible LLVM/MLIR libraries into ggml-npu.
  - Evidence: `docs/compiler-migration-plan.md`, `.gitmodules`.

- `MILESTONE/remove-openvino` — Remove the OpenVINO shim only after the neutral Intel provider reaches operation/model parity and both Intel runtimes consume its artifacts.
  - Acceptance: All enabled Intel graph paths have explicit fallback; quality/performance and reproducibility gates pass; stale cache namespaces are rejected; CI builds/tests with OpenVINO absent; neither ggml-npu nor the standalone Intel compiler transitively requires OpenVINO.
  - Evidence: `docs/compiler-migration-plan.md`.

- `MILESTONE/native-bf16-q8-f8` — Source-native BF16 pass-through, Q8_0/Q8_K transcoder parity with Q4/Q6, and F8 foundations via feature-gated `float8` (iVPU stub/reject until a provider consumes FP8). Intermediate dequant defaults to nightly f16.
  - Acceptance: `cargo test --lib` covers BF16/Q8/F8 descriptor and byte-exact paths; iVPU rejects f8 bake; Graph IR admits BF16; no `half` on the F16 path. Issues #27–#29.
  - Evidence: `src/ops/ggml_quant.rs`, `src/ops/requant.rs`, `src/ops/constant.rs`, `src/vendor/ivpu/compile.rs`, `Cargo.toml`.
