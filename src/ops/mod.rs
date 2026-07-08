//! The op axis: vendor-agnostic recognition of offloadable ggml ops, and mapping their tensors
//! to a vendor-neutral run call. One module per op/pattern; [`Op`] carries the shape key that
//! identifies the compiled artifact. Lowering (op + shape → artifact) lives per-vendor under
//! [`crate::vendor`].

pub mod matmul;

use crate::ffi::ggml_tensor;
use crate::vendor::Vendor;

/// A recognized, offloadable op together with the key identifying its compiled artifact.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// ggml `MUL_MAT`: src0 (weight) `[K,N]`, src1 (activation) `[K,M]` → `[N,M]`.
    MatMul { m: i64, k: i64, n: i64 },
}

/// Recognize a single cgraph node as a supported op, or `None`.
pub unsafe fn match_node(node: *const ggml_tensor) -> Option<Op> {
    matmul::match_node(node)
    // future: .or_else(|| ffn::match_node(node)).or_else(|| attention::match_node(node))
}

/// Extract the op's operands from the node and run it on `vendor`.
pub unsafe fn run(op: Op, node: *mut ggml_tensor, vendor: &dyn Vendor) -> bool {
    match op {
        Op::MatMul { .. } => matmul::run(op, node, vendor),
    }
}
