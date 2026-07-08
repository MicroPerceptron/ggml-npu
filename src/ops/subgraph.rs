//! Fused-subgraph recognition (vendor-agnostic). Where [`matmul`](super::matmul) recognizes a
//! single node, this recognizes a **connected run of nodes** the vendor can compile into one NPU
//! program — so the device runs it as a single command (intra-graph fusion + pipelining, one
//! submit) instead of one dispatch per node with a CPU↔NPU round-trip between each.
//!
//! The IR ([`Subgraph`]) is a small op-DAG: N boundary inputs (f32 activations crossing from CPU),
//! some baked constant weights, nodes in topological order, and one output. It's deliberately
//! general (adding an op = one arm in [`recognize`] + one in the vendor lowering); the recognizer
//! starts with the **FFN GLU** pattern (`down · (silu(gate) * up)`), the largest self-contained
//! block with no attention/RoPE/softmax to handle yet.

use alloc::vec;
use alloc::vec::Vec;

use super::Operand;
use crate::ffi::*;

/// A value inside a subgraph: an external boundary input, or a prior node's output.
#[derive(Clone, Copy)]
pub enum Ref {
    Input(u32),
    Node(u32),
}

/// A fusible op. Extend this (and the recognizer + vendor lowering) to grow coverage.
#[derive(Clone, Copy)]
pub enum SgOp {
    /// `out[M,N] = a @ weight^T`, weight baked from `consts[weight]`. `n` is the output width.
    MatMul { n: i64 },
    /// Elementwise `out = a * b`.
    Mul,
    /// `out = silu(a)`.
    Silu,
}

/// One node: its op, up to two value inputs, and (for MatMul) the baked-weight index.
pub struct SgNode {
    pub op: SgOp,
    pub a: Ref,
    pub b: Option<Ref>,
    pub weight: Option<u32>,
}

/// A baked constant weight and its logical `[N, K]` shape (needed to re-quantize it).
pub struct SgConst {
    pub weight: Operand,
    pub k: i64,
    pub n: i64,
}

/// A recognized, offloadable subgraph the vendor compiles into one program. Vendor-agnostic, like
/// [`Operand`]: `inputs` are boundary activations (`[M, K_i]`), `consts` the weights to bake,
/// `nodes` a topologically-ordered DAG, `output` the node whose result leaves the subgraph.
pub struct Subgraph {
    pub m: i64,
    pub inputs: Vec<Operand>,
    pub consts: Vec<SgConst>,
    pub nodes: Vec<SgNode>,
    pub output: u32,
    pub out: Operand,
}

unsafe fn sg_const(w: *mut ggml_tensor) -> SgConst {
    SgConst { weight: Operand::from_tensor(w), k: (*w).ne[0], n: (*w).ne[1] }
}

/// Is `w` a bakeable constant weight (a loaded model leaf with data)?
unsafe fn is_const_weight(w: *mut ggml_tensor) -> bool {
    !w.is_null() && (*w).op == GGML_OP_NONE && !(*w).data.is_null()
}

/// Recognize the split as a fusible subgraph, or `None` (caller then dispatches per node).
///
/// **FFN GLU**: `gate = X@Wg`, `up = X@Wu` (shared activation `X`), `down = Wd @ (silu(gate) * up)`
/// — the five computed nodes `{matmul, matmul, silu, mul, matmul}`. Requiring the split to be
/// exactly these five keeps the boundary unambiguous: the sole external input is `X` and the sole
/// output is `down` (safe outlining without cross-split consumer analysis).
pub unsafe fn recognize(cgraph: *mut ggml_cgraph) -> Option<Subgraph> {
    if ggml_graph_n_nodes(cgraph) != 5 {
        return None;
    }

    // Classify the five nodes: expect three MUL_MAT, one MUL, one UNARY(silu).
    let mut matmuls: Vec<*mut ggml_tensor> = Vec::new();
    let mut mul: *mut ggml_tensor = core::ptr::null_mut();
    let mut silu: *mut ggml_tensor = core::ptr::null_mut();
    for i in 0..5 {
        let t = ggml_graph_node(cgraph, i);
        if t.is_null() {
            return None;
        }
        match (*t).op {
            GGML_OP_MUL_MAT => matmuls.push(t),
            GGML_OP_MUL => {
                if !mul.is_null() {
                    return None;
                }
                mul = t;
            }
            GGML_OP_UNARY => {
                if (*t).op_params[0] != GGML_UNARY_OP_SILU as i32 || !silu.is_null() {
                    return None;
                }
                silu = t;
            }
            _ => return None,
        }
    }
    if matmuls.len() != 3 || mul.is_null() || silu.is_null() {
        return None;
    }

    // gate = silu's input matmul; up = mul's other input; down = the remaining matmul (fed by mul).
    let gate = (*silu).src[0];
    let (ms0, ms1) = ((*mul).src[0], (*mul).src[1]);
    let up = if ms0 == silu {
        ms1
    } else if ms1 == silu {
        ms0
    } else {
        return None;
    };
    if !matmuls.contains(&gate) || !matmuls.contains(&up) || gate == up {
        return None;
    }
    let down = *matmuls.iter().find(|&&t| t != gate && t != up)?;
    if (*down).src[1] != mul {
        return None;
    }

    // gate and up share the activation X; all three weights are constant leaves.
    let x = (*gate).src[1];
    if x.is_null() || (*up).src[1] != x || (*x).data.is_null() || (*down).data.is_null() {
        return None;
    }
    let (gate_w, up_w, down_w) = ((*gate).src[0], (*up).src[0], (*down).src[0]);
    if !is_const_weight(gate_w) || !is_const_weight(up_w) || !is_const_weight(down_w) {
        return None;
    }

    // Build the IR in canonical topological order: 0=gate, 1=up, 2=silu, 3=mul, 4=down.
    let m = (*x).ne[1];
    let consts = vec![sg_const(gate_w), sg_const(up_w), sg_const(down_w)];
    let nodes = vec![
        SgNode { op: SgOp::MatMul { n: (*gate_w).ne[1] }, a: Ref::Input(0), b: None, weight: Some(0) },
        SgNode { op: SgOp::MatMul { n: (*up_w).ne[1] }, a: Ref::Input(0), b: None, weight: Some(1) },
        SgNode { op: SgOp::Silu, a: Ref::Node(0), b: None, weight: None },
        SgNode { op: SgOp::Mul, a: Ref::Node(2), b: Some(Ref::Node(1)), weight: None },
        SgNode { op: SgOp::MatMul { n: (*down_w).ne[1] }, a: Ref::Node(3), b: None, weight: Some(2) },
    ];
    Some(Subgraph {
        m,
        inputs: vec![Operand::from_tensor(x)],
        consts,
        nodes,
        output: 4,
        out: Operand::from_tensor(down),
    })
}

// ---------------------------------------------------------------------------
// FFN-glue claiming + single-node fallback.
//
// `supports_op` is per-node, so to make the scheduler place a whole FFN on us as ONE split we must
// claim its glue ops (silu, mul) too — otherwise they land on CPU and fragment the FFN into
// single-node matmul splits. We gate the claim on the glue's *local* structure (only what a
// per-node view can see) so we grab FFN-shaped glue, not arbitrary elementwise ops. A claimed glue
// op that then doesn't fuse (the split wasn't a full FFN) still runs correctly as a single-node
// subgraph via [`recognize_glue`], preserving the "claimed ⟹ runnable" invariant.
// ---------------------------------------------------------------------------

/// Is `t` a SiLU unary op?
unsafe fn is_silu(t: *const ggml_tensor) -> bool {
    !t.is_null() && (*t).op == GGML_OP_UNARY && (*t).op_params[0] == GGML_UNARY_OP_SILU as i32
}

/// Plain 2D (`[K, M]`), so the single-node subgraph's `[M, K]` mapping is exact.
unsafe fn glue_2d(t: *const ggml_tensor) -> bool {
    !t.is_null() && (*t).ne[2] == 1 && (*t).ne[3] == 1
}

/// Is `t` a MUL_MAT we would claim (a bakeable constant-weight matmul)?
unsafe fn is_claimable_matmul(t: *const ggml_tensor) -> bool {
    !t.is_null() && (*t).op == GGML_OP_MUL_MAT && super::matmul::match_node(t).is_some()
}

/// Is `node` a glue op *type* (silu or mul) — dispatched via [`recognize_glue`], not `match_node`?
pub unsafe fn is_glue_op(node: *const ggml_tensor) -> bool {
    !node.is_null() && (is_silu(node) || (*node).op == GGML_OP_MUL)
}

/// Should `supports_op` claim this glue node as part of a claimable FFN? Upstream structure only
/// (per-node can't see downstream): a silu fed by a claimable matmul; a mul combining such a silu
/// and a claimable matmul. Claiming these + the matmuls groups the FFN into one NPU split.
pub unsafe fn claims_glue(node: *const ggml_tensor) -> bool {
    if node.is_null() || !glue_2d(node) {
        return false;
    }
    if is_silu(node) {
        return is_claimable_matmul((*node).src[0]);
    }
    if (*node).op == GGML_OP_MUL {
        let (a, b) = ((*node).src[0], (*node).src[1]);
        let silu_mm = |s: *const ggml_tensor| is_silu(s) && is_claimable_matmul((*s).src[0]);
        return (silu_mm(a) && is_claimable_matmul(b)) || (silu_mm(b) && is_claimable_matmul(a));
    }
    false
}

/// Build a single-node subgraph for a claimed glue op (silu or mul) — the per-node fallback for a
/// glue op that didn't fuse into a full FFN. `None` if it isn't a 2D silu/mul with present operands
/// (of matching shape for a mul). All operands are plain f32 activations at the boundary.
pub unsafe fn recognize_glue(node: *const ggml_tensor) -> Option<Subgraph> {
    if node.is_null() || (*node).data.is_null() || !glue_2d(node) {
        return None;
    }
    if is_silu(node) {
        let s0 = (*node).src[0];
        if s0.is_null() || (*s0).data.is_null() {
            return None;
        }
        return Some(Subgraph {
            m: (*s0).ne[1],
            inputs: vec![Operand::from_tensor(s0)],
            consts: Vec::new(),
            nodes: vec![SgNode { op: SgOp::Silu, a: Ref::Input(0), b: None, weight: None }],
            output: 0,
            out: Operand::from_tensor(node),
        });
    }
    if (*node).op == GGML_OP_MUL {
        let (a, b) = ((*node).src[0], (*node).src[1]);
        if a.is_null() || b.is_null() || (*a).data.is_null() || (*b).data.is_null() {
            return None;
        }
        // Elementwise: same 2D shape (broadcasting muls aren't claimed, but guard anyway).
        if (*a).ne[0] != (*b).ne[0] || (*a).ne[1] != (*b).ne[1] {
            return None;
        }
        return Some(Subgraph {
            m: (*a).ne[1],
            inputs: vec![Operand::from_tensor(a), Operand::from_tensor(b)],
            consts: Vec::new(),
            nodes: vec![SgNode { op: SgOp::Mul, a: Ref::Input(0), b: Some(Ref::Input(1)), weight: None }],
            output: 0,
            out: Operand::from_tensor(node),
        });
    }
    None
}
