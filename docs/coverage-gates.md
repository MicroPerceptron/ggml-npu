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
   follow-ups (#9–#11).
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
