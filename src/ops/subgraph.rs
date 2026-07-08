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
    /// Elementwise `out = a + b`.
    Add,
    /// `out = silu(a)`.
    Silu,
    /// `out = a / sqrt(mean(a²) + eps)` over the last dim (the per-channel weight-multiply is a
    /// separate [`Mul`] node).
    RmsNorm { eps: f32 },
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
/// **FFN SwiGLU**: `gate = X@Wg`, `up = X@Wu` (shared activation `X`), `down = Wd @ (silu(gate)*up)`.
/// ggml emits the gated activation two ways: separate `silu` + `mul` (5-node split) or a fused
/// `GGML_OP_GLU`/SwiGLU node (4-node split — what modern models use). Both are anchored on the
/// `down` matmul (its activation `src1` is a `MUL` or a SwiGLU `GLU`) and lower to the **same** IR
/// (`silu`+`mul`). Requiring the split to be exactly the FFN nodes keeps the boundary unambiguous:
/// sole external input `X`, sole output `down` (safe outlining without cross-split consumer analysis).
pub unsafe fn recognize(cgraph: *mut ggml_cgraph) -> Option<Subgraph> {
    // The FFN core is 4 (fused GLU) or 5 (silu+mul) nodes; a prepended norm adds 2, a trailing
    // residual adds 1 — so 4..=8. The exact `total` check below validates the count precisely.
    let n = ggml_graph_n_nodes(cgraph);
    if !(4..=8).contains(&n) {
        return None;
    }

    // The `down` matmul is the one whose activation is a gated-activation node (MUL or SwiGLU GLU).
    let mut down: *mut ggml_tensor = core::ptr::null_mut();
    for i in 0..n {
        let t = ggml_graph_node(cgraph, i);
        if t.is_null() {
            return None;
        }
        if (*t).op == GGML_OP_MUL_MAT {
            // `down`'s activation is the FFN gated activation: a SwiGLU, or a MUL that contains a
            // silu (silu(gate)*up) — NOT the rms_norm-weight MUL that feeds gate/up.
            let act = (*t).src[1];
            let ffn_act = !act.is_null()
                && (is_swiglu(act)
                    || ((*act).op == GGML_OP_MUL
                        && (is_silu((*act).src[0]) || is_silu((*act).src[1]))));
            if ffn_act {
                if !down.is_null() {
                    return None; // ambiguous — two candidate down matmuls
                }
                down = t;
            }
        }
    }
    if down.is_null() {
        return None;
    }
    let act = (*down).src[1];

    // Extract gate + up from the gated activation. `silu` is tracked only to gate the node count.
    let (gate, up, has_silu_node) = if (*act).op == GGML_OP_MUL {
        let (a, b) = ((*act).src[0], (*act).src[1]);
        let (silu, up) = if is_silu(a) {
            (a, b)
        } else if is_silu(b) {
            (b, a)
        } else {
            return None;
        };
        ((*silu).src[0], up, true)
    } else {
        // SwiGLU GLU(gate, up): src0=gate, src1=up; `swapped` flips which operand gets the silu.
        let swapped = (*act).op_params[1] != 0;
        let (a, b) = ((*act).src[0], (*act).src[1]);
        if swapped {
            (b, a, false)
        } else {
            (a, b, false)
        }
    };

    if gate.is_null() || up.is_null() || gate == up {
        return None;
    }
    if (*gate).op != GGML_OP_MUL_MAT || (*up).op != GGML_OP_MUL_MAT {
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

    let m = (*x).ne[1];

    // Optional prepended norm: if the FFN input `x = mul(rms_norm(x_in), norm_w)` and that pair is
    // in OUR split, fold it — the subgraph input becomes `x_in` (pre-norm) + `norm_w`, and gate/up
    // read the folded norm output. Gated on M==1: there the per-channel weight-multiply is same-
    // shape, so `norm_w` is just another bounced input (no broadcast / const machinery). At M>1 the
    // norm stays on CPU (not claimed) and the FFN fuses without it.
    let mut norm: Option<(*mut ggml_tensor, *mut ggml_tensor, f32)> = None;
    if m == 1 && (*x).op == GGML_OP_MUL && in_split(cgraph, n, x) {
        let (a, b) = ((*x).src[0], (*x).src[1]);
        let (rn, nw) = if is_rms_norm(a) {
            (a, b)
        } else if is_rms_norm(b) {
            (b, a)
        } else {
            (core::ptr::null_mut(), core::ptr::null_mut())
        };
        if !rn.is_null() && in_split(cgraph, n, rn) {
            let x_in = (*rn).src[0];
            if !x_in.is_null() && !nw.is_null() && glue_2d(x_in) && glue_2d(nw)
                && !(*x_in).data.is_null() && !(*nw).data.is_null()
            {
                norm = Some((x_in, nw, f32::from_bits((*rn).op_params[0] as u32)));
            }
        }
    }

    // Optional trailing residual: an `ADD` whose one operand is `down` and the other a distinct 2D
    // runtime tensor (the pre-FFN residual — which, when a norm is folded, is the norm's own x_in).
    let mut residual: *mut ggml_tensor = core::ptr::null_mut();
    let mut residual_add: *mut ggml_tensor = core::ptr::null_mut();
    for i in 0..n {
        let t = ggml_graph_node(cgraph, i);
        if t.is_null() || (*t).op != GGML_OP_ADD {
            continue;
        }
        let (s0, s1) = ((*t).src[0], (*t).src[1]);
        let other = if s0 == down {
            s1
        } else if s1 == down {
            s0
        } else {
            continue;
        };
        if !other.is_null() && other != x && glue_2d(other) && !(*other).data.is_null() {
            residual = other;
            residual_add = t;
        }
        break;
    }

    // Exactly the split's nodes: [rms_norm + mul] (norm) + FFN (+ silu form) (+ residual add).
    let ffn_count = if has_silu_node { 5 } else { 4 };
    let total = ffn_count + if norm.is_some() { 2 } else { 0 } + if residual_add.is_null() { 0 } else { 1 };
    if n != total {
        return None;
    }

    // Build the IR with running indices (the norm, when present, prepends 2 nodes).
    let consts = vec![sg_const(gate_w), sg_const(up_w), sg_const(down_w)];
    let mut nodes: Vec<SgNode> = Vec::new();
    let mut inputs: Vec<Operand> = Vec::new();
    let x_in_tensor: *mut ggml_tensor;
    let ffn_in: Ref = if let Some((x_in, norm_w, eps)) = norm {
        x_in_tensor = x_in;
        inputs.push(Operand::from_tensor(x_in)); // Input(0) = x_in (pre-norm)
        inputs.push(Operand::from_tensor(norm_w)); // Input(1) = norm weight
        nodes.push(SgNode { op: SgOp::RmsNorm { eps }, a: Ref::Input(0), b: None, weight: None });
        nodes.push(SgNode { op: SgOp::Mul, a: Ref::Node(0), b: Some(Ref::Input(1)), weight: None });
        Ref::Node(1)
    } else {
        x_in_tensor = core::ptr::null_mut();
        inputs.push(Operand::from_tensor(x)); // Input(0) = the FFN input
        Ref::Input(0)
    };
    let g = nodes.len() as u32;
    nodes.push(SgNode { op: SgOp::MatMul { n: (*gate_w).ne[1] }, a: ffn_in, b: None, weight: Some(0) });
    nodes.push(SgNode { op: SgOp::MatMul { n: (*up_w).ne[1] }, a: ffn_in, b: None, weight: Some(1) });
    nodes.push(SgNode { op: SgOp::Silu, a: Ref::Node(g), b: None, weight: None });
    nodes.push(SgNode { op: SgOp::Mul, a: Ref::Node(g + 2), b: Some(Ref::Node(g + 1)), weight: None });
    nodes.push(SgNode { op: SgOp::MatMul { n: (*down_w).ne[1] }, a: Ref::Node(g + 3), b: None, weight: Some(2) });
    let mut output = g + 4;
    let mut out = down;
    if !residual_add.is_null() {
        // Reuse Input(0) when the residual is the norm's own x_in (the common real-layer case).
        let residual_ref = if !x_in_tensor.is_null() && residual == x_in_tensor {
            Ref::Input(0)
        } else {
            let ri = inputs.len() as u32;
            inputs.push(Operand::from_tensor(residual));
            Ref::Input(ri)
        };
        nodes.push(SgNode { op: SgOp::Add, a: Ref::Node(g + 4), b: Some(residual_ref), weight: None });
        output = g + 5;
        out = residual_add;
    }
    Some(Subgraph {
        m,
        inputs,
        consts,
        nodes,
        output,
        out: Operand::from_tensor(out),
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

/// Is `t` a fused SwiGLU `GLU` node (`glu = silu(a)*b`, the modern-model FFN activation)?
unsafe fn is_swiglu(t: *const ggml_tensor) -> bool {
    !t.is_null() && (*t).op == GGML_OP_GLU && (*t).op_params[0] == GGML_GLU_OP_SWIGLU as i32
}

/// Is `t` an RMSNorm node?
unsafe fn is_rms_norm(t: *const ggml_tensor) -> bool {
    !t.is_null() && (*t).op == GGML_OP_RMS_NORM
}

/// Is `t` one of the split's computed nodes (vs an external boundary input the split reads)?
unsafe fn in_split(cgraph: *mut ggml_cgraph, n: i32, t: *const ggml_tensor) -> bool {
    for i in 0..n {
        if ggml_graph_node(cgraph, i) as *const ggml_tensor == t {
            return true;
        }
    }
    false
}

/// Plain 2D (`[K, M]`), so the single-node subgraph's `[M, K]` mapping is exact.
unsafe fn glue_2d(t: *const ggml_tensor) -> bool {
    !t.is_null() && (*t).ne[2] == 1 && (*t).ne[3] == 1
}

/// Is `t` a MUL_MAT we would claim (a bakeable constant-weight matmul)?
unsafe fn is_claimable_matmul(t: *const ggml_tensor) -> bool {
    !t.is_null() && (*t).op == GGML_OP_MUL_MAT && super::matmul::match_node(t).is_some()
}

/// Is `node` a glue op *type* (silu, mul, SwiGLU, add, or RMSNorm) — dispatched via
/// [`recognize_glue`], not `match_node`?
pub unsafe fn is_glue_op(node: *const ggml_tensor) -> bool {
    !node.is_null()
        && (is_silu(node)
            || (*node).op == GGML_OP_MUL
            || (*node).op == GGML_OP_ADD
            || is_swiglu(node)
            || is_rms_norm(node))
}

/// Should `supports_op` claim this glue node as part of a claimable FFN? Upstream structure only
/// (per-node can't see downstream): a silu fed by a claimable matmul; a mul combining such a silu
/// and a claimable matmul; a SwiGLU fed by two claimable matmuls (gate + up); a **residual add**
/// combining a claimable matmul (the `down` projection) and a runtime tensor. Claiming these + the
/// matmuls groups the FFN (+ its residual) into one NPU split. (The residual-add claim also grabs
/// e.g. the attention residual, which then runs as a standalone 1-node add — correct, minor cost.)
pub unsafe fn claims_glue(node: *const ggml_tensor) -> bool {
    if node.is_null() || !glue_2d(node) {
        return false;
    }
    if is_silu(node) {
        return is_claimable_matmul((*node).src[0]);
    }
    if is_swiglu(node) {
        return is_claimable_matmul((*node).src[0]) && is_claimable_matmul((*node).src[1]);
    }
    // RMSNorm and its weight-multiply — only at decode (M==1), where the per-channel weight-multiply
    // is same-shape (no broadcast). Over-claims: an attention norm gets claimed too and runs as its
    // own standalone dispatch (correct, minor cost) until attention fuses.
    if is_rms_norm(node) {
        return (*node).ne[1] == 1;
    }
    if (*node).op == GGML_OP_MUL {
        let (a, b) = ((*node).src[0], (*node).src[1]);
        let silu_mm = |s: *const ggml_tensor| is_silu(s) && is_claimable_matmul((*s).src[0]);
        if (silu_mm(a) && is_claimable_matmul(b)) || (silu_mm(b) && is_claimable_matmul(a)) {
            return true;
        }
        // norm-weight mul: rms_norm × per-channel weight, at decode.
        return (*node).ne[1] == 1 && (is_rms_norm(a) || is_rms_norm(b));
    }
    if (*node).op == GGML_OP_ADD {
        let (a, b) = ((*node).src[0], (*node).src[1]);
        // matmul output + a runtime tensor (op != NONE) — a residual, not a const-bias add.
        let residual = |mm: *const ggml_tensor, r: *const ggml_tensor| {
            is_claimable_matmul(mm) && !r.is_null() && (*r).op != GGML_OP_NONE && glue_2d(r)
        };
        return residual(a, b) || residual(b, a);
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
    if (*node).op == GGML_OP_ADD {
        let (a, b) = ((*node).src[0], (*node).src[1]);
        if a.is_null() || b.is_null() || (*a).data.is_null() || (*b).data.is_null() {
            return None;
        }
        if (*a).ne[0] != (*b).ne[0] || (*a).ne[1] != (*b).ne[1] {
            return None;
        }
        return Some(Subgraph {
            m: (*a).ne[1],
            inputs: vec![Operand::from_tensor(a), Operand::from_tensor(b)],
            consts: Vec::new(),
            nodes: vec![SgNode { op: SgOp::Add, a: Ref::Input(0), b: Some(Ref::Input(1)), weight: None }],
            output: 0,
            out: Operand::from_tensor(node),
        });
    }
    if is_swiglu(node) {
        // Standalone SwiGLU → a 2-node silu+mul subgraph. `swapped` picks which operand gets silu.
        let swapped = (*node).op_params[1] != 0;
        let (g, u) = if swapped {
            ((*node).src[1], (*node).src[0])
        } else {
            ((*node).src[0], (*node).src[1])
        };
        if g.is_null() || u.is_null() || (*g).data.is_null() || (*u).data.is_null() {
            return None;
        }
        if (*g).ne[0] != (*u).ne[0] || (*g).ne[1] != (*u).ne[1] {
            return None;
        }
        return Some(Subgraph {
            m: (*g).ne[1],
            inputs: vec![Operand::from_tensor(g), Operand::from_tensor(u)],
            consts: Vec::new(),
            nodes: vec![
                SgNode { op: SgOp::Silu, a: Ref::Input(0), b: None, weight: None },
                SgNode { op: SgOp::Mul, a: Ref::Node(0), b: Some(Ref::Input(1)), weight: None },
            ],
            output: 1,
            out: Operand::from_tensor(node),
        });
    }
    if is_rms_norm(node) {
        let s0 = (*node).src[0];
        if s0.is_null() || (*s0).data.is_null() {
            return None;
        }
        return Some(Subgraph {
            m: (*s0).ne[1],
            inputs: vec![Operand::from_tensor(s0)],
            consts: Vec::new(),
            nodes: vec![SgNode {
                op: SgOp::RmsNorm { eps: f32::from_bits((*node).op_params[0] as u32) },
                a: Ref::Input(0),
                b: None,
                weight: None,
            }],
            output: 0,
            out: Operand::from_tensor(node),
        });
    }
    None
}
