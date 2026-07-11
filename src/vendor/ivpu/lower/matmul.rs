//! MatMul lowering for the Intel NPU, **Constant path**: the weight is re-quantized and baked
//! into a compile-time `Constant` in the NPU blob (the fused decompress-matmul), so at dispatch
//! only the activation is an input and the output an output. `inputs` are `[activation, weight]`;
//! `output` is dst. The built graph is cached per weight (and M) in [`super::super::session`]; the
//! [`super::super::compile`] blob it's built from is transient (dropped after the graph is built).

use super::super::{compile, session};
use crate::ops::Operand;

/// Can this MatMul run on the NPU? A cheap **feasibility** check (shim present + weight
/// re-quantizable) — it does **not** compile the blob. The blob compiles lazily in [`run`] (or,
/// when the matmul fuses into an FFN, only the fused subgraph blob is built). This avoids a wasted
/// standalone compile for every FFN matmul at scheduling time; the fused-subgraph path likewise
/// compiles lazily at `graph_compute`, so the scheduler contract is the same.
pub fn can_lower(m: i64, k: i64, n: i64, inputs: &[Operand]) -> bool {
    let Some(weight) = inputs.get(1) else {
        return false;
    };
    compile::const_feasible(weight, m, k, n)
}

pub fn run(m: i64, k: i64, n: i64, inputs: &[Operand], output: &Operand) -> bool {
    let (Some(activation), Some(weight)) = (inputs.first(), inputs.get(1)) else {
        return false;
    };
    // Look up (or, on a cold miss, compile → build graph → drop the blob) this weight's program,
    // keyed by `(weight ptr, M)`. The native blob is a transient compile artifact — only the built
    // graph is kept resident, so a weight is never held twice (as blob bytes and in the graph).
    let ok = session::run_const(weight.ptr as usize, m, &[activation], &[output], || {
        compile::compile_const_blob(weight, m, k, n)
    });
    if !ok {
        super::super::profile::note(&alloc::format!(
            "[npu] matmul run FAIL (m={m} k={k} n={n})\n"
        ));
    }
    ok
}
