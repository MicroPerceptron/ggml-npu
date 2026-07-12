//! MUL_MAT recognition + tensor mapping (vendor-agnostic). Lowering to a device artifact is
//! per-vendor (`crate::vendor::*::lower::matmul`).

use super::{Op, Operand};
use crate::ffi::*;
use crate::vendor::Vendor;

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
    // Only a **constant weight** — a loaded model leaf (`op == GGML_OP_NONE`) — can be baked as a
    // compile-time Constant. A runtime tensor (`op != NONE`) changes across tokens, so baking it
    // would be wrong; we never claim those (they stay on CPU). In llama.cpp every offloadable
    // projection matmul has its weight as such a leaf, so this excludes nothing we accelerated.
    if s0.op != GGML_OP_NONE {
        return None;
    }
    // Weight (src0) may be f32 or any quantized type ggml can dequantize; activation (src1)
    // stays f32. For a quantized weight the K dim must be a whole number of quant blocks so a
    // row-major `to_float` over the tensor stays block-aligned.
    if s1.type_ != GGML_TYPE_F32 {
        return None;
    }
    let traits = ggml_get_type_traits(s0.type_);
    if traits.is_null() {
        return None;
    }
    let tr = &*traits;
    if s0.type_ != GGML_TYPE_F32 && !tr.is_quantized {
        return None;
    }
    if tr.is_quantized && (tr.blck_size <= 0 || s0.ne[0] % tr.blck_size != 0) {
        return None;
    }
    // 2D only (no batch dims), contraction dim K must match, both contiguous.
    if s0.ne[2] != 1 || s0.ne[3] != 1 || s1.ne[2] != 1 || s1.ne[3] != 1 || s0.ne[0] != s1.ne[0] {
        return None;
    }
    if !ggml_is_contiguous(src0 as *const ggml_tensor)
        || !ggml_is_contiguous(src1 as *const ggml_tensor)
    {
        return None;
    }
    let (k, n, m) = (s0.ne[0], s0.ne[1], s1.ne[1]);
    if !(1..=65536).contains(&k) || !(1..=65536).contains(&n) || !(1..=65536).contains(&m) {
        return None;
    }
    Some(Op::MatMul { m, k, n })
}

/// Scheduler probe (`supports_op`): can `vendor` lower this node? Builds the `[activation, weight]`
/// operands and asks the vendor, which (for the Constant path) re-quantizes + compiles the weight
/// blob now so a later [`run`] is guaranteed to find it. The weight (src0) must be materialized;
/// src1/dst may still be unallocated at probe time (their pointers aren't dereferenced here).
pub unsafe fn can_lower(op: Op, node: *const ggml_tensor, vendor: &dyn Vendor) -> bool {
    let t = &*node;
    let (src0, src1) = (t.src[0], t.src[1]);
    if src0.is_null() || src1.is_null() || (*src0).data.is_null() {
        return false;
    }
    let weight = Operand::from_tensor(src0);
    let activation = Operand::from_tensor(src1);
    vendor.can_lower(op, &[activation, weight])
}

/// Describe the operands (pointer + element count + NPU-ownership) and run on `vendor`, which
/// binds them directly when they're in NPU memory (no copy) and bounces them otherwise.
/// Contract: the artifact's input args are ordered `[activation, weight]` (src1, then src0);
/// the output maps directly to dst.
pub unsafe fn run(op: Op, node: *mut ggml_tensor, vendor: &dyn Vendor) -> bool {
    let t = &*node;
    let (src0, src1) = (t.src[0], t.src[1]);
    if src0.is_null() || src1.is_null() || t.data.is_null() {
        return false;
    }
    if (*src0).data.is_null() || (*src1).data.is_null() {
        return false;
    }
    let weight = Operand::from_tensor(src0);
    let activation = Operand::from_tensor(src1);
    let out = Operand::from_tensor(node as *const ggml_tensor);
    vendor.run(op, &[activation, weight], &out)
}
