//! MatMul lowering for the Intel NPU: fetch (or JIT-compile) the shape's native blob and
//! submit it over Level Zero. `inputs` are `[activation, weight]` (the blob's arg order).

use super::super::{compile, graph, ze};

pub fn run(m: i64, k: i64, n: i64, inputs: &[&[f32]], output: &mut [f32]) -> bool {
    let Some(blob) = compile::get_or_compile_matmul(m, k, n) else {
        return false;
    };
    let devices = ze::probe();
    let Some(dev) = devices.first() else {
        return false;
    };
    graph::run_native_blob(dev, blob, inputs, output).is_ok()
}
