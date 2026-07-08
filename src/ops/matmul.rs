//! MUL_MAT recognition + tensor mapping (vendor-agnostic). Lowering to a device artifact is
//! per-vendor (`crate::vendor::*::lower::matmul`).

use super::Op;
use crate::ffi::*;
use crate::vendor::Vendor;

fn nelements(t: &ggml_tensor) -> usize {
    (t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]) as usize
}

/// Recognize a 2D f32 `MUL_MAT`: src0 (weight) `[K,N]`, src1 (activation) `[K,M]`, both
/// contiguous, K matching, within sane bounds. Returns the shape key.
pub unsafe fn match_node(node: *const ggml_tensor) -> Option<Op> {
    if node.is_null() {
        return None;
    }
    let t = &*node;
    if t.op != GGML_OP_MUL_MAT || t.type_ != GGML_TYPE_F32 {
        return None;
    }
    let (src0, src1) = (t.src[0], t.src[1]);
    if src0.is_null() || src1.is_null() {
        return None;
    }
    let (s0, s1) = (&*src0, &*src1);
    if s0.type_ != GGML_TYPE_F32 || s1.type_ != GGML_TYPE_F32 {
        return None;
    }
    // 2D only (no batch dims), contraction dim K must match, both contiguous.
    if s0.ne[2] != 1 || s0.ne[3] != 1 || s1.ne[2] != 1 || s1.ne[3] != 1 || s0.ne[0] != s1.ne[0] {
        return None;
    }
    if !ggml_is_contiguous(src0 as *const ggml_tensor) || !ggml_is_contiguous(src1 as *const ggml_tensor) {
        return None;
    }
    let (k, n, m) = (s0.ne[0], s0.ne[1], s1.ne[1]);
    if !(1..=65536).contains(&k) || !(1..=65536).contains(&n) || !(1..=65536).contains(&m) {
        return None;
    }
    Some(Op::MatMul { m, k, n })
}

/// Read the operand host data and run on `vendor`. Contract: the artifact's input args are
/// ordered `[activation, weight]` (src1, then src0); the output maps directly to dst.
pub unsafe fn run(op: Op, node: *mut ggml_tensor, vendor: &dyn Vendor) -> bool {
    let t = &*node;
    let (src0, src1) = (t.src[0], t.src[1]);
    if src0.is_null() || src1.is_null() || t.data.is_null() {
        return false;
    }
    let (s0, s1) = (&*src0, &*src1);
    if s0.data.is_null() || s1.data.is_null() {
        return false;
    }
    let weight = core::slice::from_raw_parts(s0.data as *const f32, nelements(s0));
    let activation = core::slice::from_raw_parts(s1.data as *const f32, nelements(s1));
    let out = core::slice::from_raw_parts_mut(t.data as *mut f32, nelements(t));
    vendor.run(op, &[activation, weight], out)
}
