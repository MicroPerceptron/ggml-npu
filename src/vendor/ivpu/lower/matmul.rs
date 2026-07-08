//! MatMul lowering for the Intel NPU: dispatch through the session cache, which JIT-compiles the
//! per-shape graph on first sight and records a per-node execute list bound directly to the
//! operands' NPU memory (zero copy) thereafter. `inputs` are `[activation, weight]` (blob arg
//! order); `output` is dst. The device MatMul takes an **f16 weight**, so the weight is first
//! materialized as an f16 NPU shadow (dequantized once); the f32 activation/output bind directly.

use super::super::{session, weights};
use crate::ops::Operand;

pub fn run(m: i64, k: i64, n: i64, inputs: &[Operand], output: &Operand) -> bool {
    let (Some(activation), Some(weight)) = (inputs.first(), inputs.get(1)) else {
        return false;
    };
    let Some(weight_f16) = weights::shadow(weight, (k * n) as usize) else {
        return false;
    };
    session::run_matmul(m, k, n, activation, &weight_f16, output)
}
