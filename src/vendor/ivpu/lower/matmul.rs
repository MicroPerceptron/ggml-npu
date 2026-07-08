//! MatMul lowering for the Intel NPU, **Constant path**: the weight is re-quantized and baked
//! into a compile-time `Constant` in the NPU blob (the fused decompress-matmul), so at dispatch
//! only the activation is an input and the output an output. `inputs` are `[activation, weight]`;
//! `output` is dst. Compilation is cached per weight (and M) — see [`super::super::compile`].

use super::super::{compile, session};
use crate::ops::Operand;

/// Can this MatMul run on the NPU? True iff we can re-quantize the weight and compile (or
/// disk-load) its Constant blob. Compiling here upholds the scheduler contract: a `true` here
/// means the blob is cached, so [`run`] cannot then fail to find it.
pub fn can_lower(m: i64, k: i64, n: i64, inputs: &[Operand]) -> bool {
    let Some(weight) = inputs.get(1) else {
        return false;
    };
    compile::get_or_compile_const(weight, m, k, n).is_some()
}

pub fn run(m: i64, k: i64, n: i64, inputs: &[Operand], output: &Operand) -> bool {
    let (Some(activation), Some(weight)) = (inputs.first(), inputs.get(1)) else {
        return false;
    };
    let Some(blob) = compile::get_or_compile_const(weight, m, k, n) else {
        return false;
    };
    session::run(blob, &[activation], output)
}
