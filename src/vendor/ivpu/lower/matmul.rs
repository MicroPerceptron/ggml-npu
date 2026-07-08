//! MatMul lowering for the Intel NPU: dispatch through the per-shape session cache (which
//! JIT-compiles + prepares the graph on first sight and reuses it thereafter). `inputs` are
//! `[activation, weight]` (the blob's arg order).

use super::super::session;

pub fn run(m: i64, k: i64, n: i64, inputs: &[&[f32]], output: &mut [f32]) -> bool {
    let (Some(&activation), Some(&weight)) = (inputs.first(), inputs.get(1)) else {
        return false;
    };
    session::run_matmul(m, k, n, activation, weight, output)
}
