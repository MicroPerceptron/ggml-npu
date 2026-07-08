//! MatMul lowering for the Intel NPU: dispatch through the session cache, which JIT-compiles the
//! per-shape graph on first sight and records a per-node execute list bound directly to the
//! operands' NPU memory (zero copy) thereafter. `inputs` are `[activation, weight]` (blob arg
//! order); `output` is dst.

use super::super::session;
use crate::ops::Operand;

pub fn run(m: i64, k: i64, n: i64, inputs: &[Operand], output: &Operand) -> bool {
    let (Some(activation), Some(weight)) = (inputs.first(), inputs.get(1)) else {
        return false;
    };
    session::run_matmul(m, k, n, activation, weight, output)
}
