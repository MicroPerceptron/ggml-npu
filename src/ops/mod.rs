//! The op axis: vendor-agnostic recognition of offloadable ggml ops, and mapping their tensors
//! to a vendor-neutral run call. One module per op/pattern; [`Op`] carries the shape key that
//! identifies the compiled artifact. Lowering (op + shape → artifact) lives per-vendor under
//! [`crate::vendor`].

pub mod matmul;

use core::ffi::c_void;

use crate::ffi::ggml_tensor;
use crate::vendor::Vendor;

/// A recognized, offloadable op together with the key identifying its compiled artifact.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// ggml `MUL_MAT`: src0 (weight) `[K,N]`, src1 (activation) `[K,M]` → `[N,M]`.
    MatMul { m: i64, k: i64, n: i64 },
}

/// One operand of an op, described so a vendor can bind it to the device **without copying**:
/// the raw data pointer (`tensor->data`), its element count, and whether it already lives in
/// NPU-mapped memory (`owned` — bindable directly) or in a foreign buffer (must be staged
/// through a bounce buffer). Elements are f32 — the only device precision we compile for today.
#[derive(Clone, Copy)]
pub struct Operand {
    pub ptr: *mut c_void,
    pub elems: usize,
    pub owned: bool,
}

impl Operand {
    /// Describe a ggml tensor as an operand: data pointer, element count, and whether it's in
    /// our (NPU-mapped) buffer. Caller must ensure `t` is non-null with a valid `data`.
    pub unsafe fn from_tensor(t: *const ggml_tensor) -> Operand {
        let d = &*t;
        Operand {
            ptr: d.data,
            elems: (d.ne[0] * d.ne[1] * d.ne[2] * d.ne[3]) as usize,
            owned: crate::backend::owns_tensor(t),
        }
    }
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
