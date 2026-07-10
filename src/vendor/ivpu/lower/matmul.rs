//! MatMul lowering for the Intel NPU, **Constant path**: the weight is re-quantized and baked
//! into a compile-time `Constant` in the NPU blob (the fused decompress-matmul), so at dispatch
//! only the activation is an input and the output an output. `inputs` are `[activation, weight]`;
//! `output` is dst. Compilation is cached per weight (and M) — see [`super::super::compile`].

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
    let Some(blob) = compile::get_or_compile_const(weight, m, k, n) else {
        super::super::profile::note(&alloc::format!("[npu] matmul run FAIL: compile None (m={m} k={k} n={n})\n"));
        return false;
    };
    let ok = session::run(blob, &[activation], &[output]);
    if !ok {
        super::super::profile::note(&alloc::format!("[npu] matmul run FAIL: session::run false (m={m} k={k} n={n})\n"));
    }
    ok
}
