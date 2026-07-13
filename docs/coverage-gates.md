# Coverage gates for NPU op / subgraph claims

Issue [#12](https://github.com/MicroPerceptron/ggml-npu/issues/12): capability and compile claims
must not outrun documented proofs.

## Matrix

`tests/coverage_matrix.csv` is the authoritative claim ⊆ coverage table. Columns:

| column | meaning |
| --- | --- |
| `wire_op` | [`WireOp`](../src/graph.rs) name (`WireOp::ALL`) |
| `compile_claimed` | `yes` iff Intel `wire_op_compile_supported` claims a lowering |
| `gate` | `hermetic` (CI), `shim` (OpenVINO shim + backend selftest), `opt_in_hw` (device/MODEL), or `none` |
| `proof` | `;`-separated tags: `contract:…`, `test:module::name`, `script:path`, `issue:#N` |
| `notes` | human context |

## Hermetic gate (CI / local)

```sh
./tests/run_claim_coverage_check.sh
# or
cargo test --lib coverage::
```

This fails when:

1. a `WireOp` is missing from the matrix;
2. compile claims disagree with `compile_claimed`;
3. a compile-claimed row has empty proofs or lacks a `contract:` tag;
4. a `script:` proof path is missing on disk.

It does **not** run full LFM pp512 and does not require NPU hardware.

## Extending coverage when adding an op

1. Add the variant to `WireOp` and `WireOp::ALL` / `as_str`.
2. Wire decode in `OpRegistry` (`src/graph.rs`) — do not silently claim via policy alone.
3. If the Intel compiler can lower it, add it to `wire_op_compile_supported` in
   `src/vendor/ivpu/compile.rs` (issue #8 owns broader capability unification; do not invent a
   second policy path here).
4. Add a matrix row with honest `gate` + proof tags **in the same change**.
5. Prefer a hermetic `contract:` / `test:` proof first; add `script:` or `issue:#…` for shim/HW
   follow-ups (#9–#11). ShortConv gather/write (#9): hermetic proofs cover connected
   gather→SSM_CONV→SetRows partitioning and FeasibilityLimit adjacent peels when Intel refuses
   SSM_CONV+SET_ROWS fusion; shim script remains the numerical gate.
6. Re-run `./tests/run_claim_coverage_check.sh`.

## Full local coverage gate

`./tests/run_coverage_gate.sh` runs the hermetic claim check, unit tests, registration, and the
shim numerical suite (`graph_ir_shortconv_check`). Set `MODEL=…` only for an optional short
prefill/decode smoke — never as a substitute for the matrix gate, and never a full locked pp512
unless explicitly requested.

## Bench baselines

`benchmarks/summarize.py` labels NPU runs without submission proof as `npu-unproven`. Use
`--require-baseline-proof` when aggregating results that will be cited as baselines: CPU/Vulkan
modes and proven direct / Level Zero NPU runs pass; `npu-unproven` fails.

## Claim / unclaim / fail-closed

Capability placement must not reintroduce soft fallback or layout-only over-claim (issues #5/#8,
PR #32/#35). Contract:

1. **Unclaim** — `supports_op` / `offload_op` false (or refuse the split) so ggml places elsewhere.
   Prefer this for layout-only Metadata ops (`View`/`Reshape`/`Cont`/`Transpose`/`Permute`) that
   OpenVINO cannot lower alone; keep them on the fused wire allowlist only.
2. **Fail-closed after claim** — once NPU claimed a split, compile/exec failure returns
   `GGML_STATUS_FAILED` with `claimed_execution_failure` attribution — never silent success.
3. **Proof-gated benches** — do not cite `npu-unproven` as an NPU baseline.

Agent rule: `.cursor/rules/npu-claim-fallback.mdc`. ADR: `ADR/capability-contract-fail-closed` in
`.cursor/project-memory.md`.

## Attention gates (`GGML_NPU_ATTENTION`)

Issue [#10](https://github.com/MicroPerceptron/ggml-npu/issues/10). Exact value `full` opts into the
attention core; `=1`, presence-only, or unset keeps SoftMax / FlashAttention / runtime-weight
matmuls unclaimed (fail-closed; no vanity claim). Prefill RoPE (modes 0|2), Permute, and KV-cache
continuity pulls do **not** require this gate (PR #39).

| Surface | Default | `ATTENTION=full` | Supported variants |
| --- | --- | --- | --- |
| SoftMax | unclaim | capability + fusible pull under an anchor | `max_bias == 0`; arity 1–2 (optional mask); scale in `op_params[0]` |
| FlashAttention | unclaim | capability + anchor pull | `max_bias == 0` and `logit_softcap == 0`; arity 3–4; GQA when query heads divisible by KV heads |
| Runtime-weight MatMul (KQ/KQV) | unclaim + no `offload_op` pull | capability + work-floor pull | same MatMul lowering as const-weight; weight is a non-`NONE` producer |
| Attention-core fusible glue (Scale/Mul/Add/…) | unclaim (attention-side refused) | `offload_op` pull under SoftMax/FA/runtime-KQ or attention-side QKV/RoPE span | Non-Metadata only; layout View/Reshape/Permute stay supports_op-unclaimed (PR #35) |
| Const-weight MatMul / RoPE / KV SetRows | claimable without `full` | unchanged | see prefill QKV row / ShortConv docs |

Shim numerical proofs for SoftMax and FlashAttention live in `tests/graph_ir_shortconv_check.c`
(gate=`shim`). Hermetic contract tests cover decode variants and the env gate without NPU hardware.
Do not cite tok/s movement from enabling `ATTENTION=full` without a measured attribution run.

**Duty-cycle note (Direct):** `ATTENTION=full` admits SoftMax/FA/runtime KQ and pulls their fusible
neighborhood. LFM short smokes with `-fa on` still often show FA as thin NPU islands (FA+Reshape)
because intervening layout Metadata cannot be `supports_op`-claimed. Shared tensor GiB may stay flat
while program-resident ELF grows; coalesce/cut-cost may re-absorb thin FA into Vulkan. Growing fused
NPU work further is a scheduler/layout-continuity problem, not a missing SoftMax/FA capability gate.

## Sparse MoE + output head (#11)

| Path | Default (`MOE` unset/`=0`) | Opt-in (`GGML_NPU_MOE=1`) |
| --- | --- | --- |
| Argsort (expert ids) | **Unclaim** | Claim when compile-ready **with ggml-visible i32 writeback** (`capture` / `partition_range` force Argsort outputs even if View/GetRows consume them in-graph) |
| MatMulId (routed experts) | **Unclaim** | **Unclaim** — Vulkan/host owns routed expert banks until model-scale evidence; shim may still lower fused Argsort→MatMulId with ids internal |
| Sigmoid / SumRows | **Unclaim** (`traits.moe` + `is_opt_in_one`) | Claim when compile-ready |
| Packed Q4/Q6/Q8 expert banks | n/a | Canonical **requant** Fallback (unpacked) when MatMulId is later re-enabled; never packed source-native gather |
| Float f16/bf16 expert banks | n/a | SourceBytes pass-through + Gather-before-MatMul (shim today; claim gated with MatMulId) |
| Output-head-scale dense MatMul | **Unclaim** via `expanded_constant_budget` (GPU-class vocab×hidden); `GGML_NPU_ALLOW_GIANT_CONSTANTS` is the explicit escape hatch — do not claim then corrupt |

Presence of `GGML_NPU_MOE=0` must not enable MoE (PR #38). Claiming Argsort **without** id
writeback corrupted expert ids for host/Vulkan `MUL_MAT_ID` (device-intermediate TopK never
reached ggml tensors — PR #49 unclaimed; writeback reclaim restores Argsort only). MatMulId remains
supports_op-unclaimed at model scale (documented Vulkan/host experts). Shim numerical coverage
lives in `tests/graph_ir_shortconv_check.c` (Argsort+MatMulId fused, Sigmoid, SumRows).
