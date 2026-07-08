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

/// Upper bound on split size (keeps the O(n²) operand/consumer scans cheap; a layer is ~30 nodes).
const MAX_SG_NODES: usize = 64;

/// Elementwise ops require identical 2D shapes (the `[M, K]` boundary model has no broadcast).
unsafe fn same_2d(a: *const ggml_tensor, b: *const ggml_tensor) -> bool {
    glue_2d(a) && glue_2d(b) && (*a).ne[0] == (*b).ne[0] && (*a).ne[1] == (*b).ne[1]
}

/// Resolve a ggml src operand to a subgraph value reference: an earlier node in the split
/// (`Ref::Node`) or an external boundary input (`Ref::Input`, deduped by tensor identity and
/// bounced/bound at run time). `None` for a null/dataless src or a non-topological forward reference.
unsafe fn resolve(
    src: *mut ggml_tensor,
    cnodes: &[*mut ggml_tensor],
    ir_out: &[u32],
    ci: usize,
    inputs: &mut Vec<Operand>,
    input_ptrs: &mut Vec<usize>,
) -> Option<Ref> {
    if src.is_null() {
        return None;
    }
    for (j, &cn) in cnodes.iter().enumerate() {
        if cn == src {
            if j >= ci {
                return None; // forward reference — graph not in topological order
            }
            return Some(Ref::Node(ir_out[j]));
        }
    }
    if (*src).data.is_null() {
        return None;
    }
    let p = src as usize;
    if let Some(idx) = input_ptrs.iter().position(|&x| x == p) {
        return Some(Ref::Input(idx as u32));
    }
    let idx = inputs.len() as u32;
    inputs.push(Operand::from_tensor(src));
    input_ptrs.push(p);
    Some(Ref::Input(idx))
}

/// Recognize the split as a fusible subgraph, or `None` (caller then dispatches per node).
///
/// **General op-DAG matcher.** Walks the split's nodes in topological order and lowers each
/// supported op to the [`SgOp`] IR, resolving every operand to an earlier node, a baked constant
/// weight (matmul `src0`), or an external boundary input. Fuses any **single-output** connected DAG
/// of supported ops — an FFN (`silu`+`mul` or fused SwiGLU), a prepended RMSNorm, a trailing
/// residual, a bare `matmul + residual`, and combinations — into one program. Bails (→ per-node) on
/// an unsupported op, a non-bakeable matmul weight, a broadcast elementwise op, or **more than one
/// output** (a branchy split like `{Q,K,V}`, which a single-output artifact can't represent).
///
/// Supported ops: `MUL_MAT` (const weight), `MUL`/`ADD` (same-shape), SiLU, RMSNorm, SwiGLU (→
/// silu·mul). Adding an op = one arm here + one shim `case`. The `[M, K]` boundary model has no
/// broadcast, so per-channel weight-multiplies (RMSNorm's) only fuse at M==1 (same-shape); the
/// scheduler's `claims_glue` gates that so M>1 norms stay on CPU and the FFN still fuses.
pub unsafe fn recognize(cgraph: *mut ggml_cgraph) -> Option<Subgraph> {
    let n = ggml_graph_n_nodes(cgraph) as usize;
    // Single nodes go through the per-node path (matmul Constant path / 1-node glue).
    if n < 2 || n > MAX_SG_NODES {
        return None;
    }
    let mut cnodes: Vec<*mut ggml_tensor> = Vec::with_capacity(n);
    for i in 0..n {
        let t = ggml_graph_node(cgraph, i as i32);
        if t.is_null() {
            return None;
        }
        cnodes.push(t);
    }

    let mut ir_nodes: Vec<SgNode> = Vec::new();
    let mut ir_out: Vec<u32> = vec![0u32; n]; // ir_out[ci] = IR node producing cnodes[ci]'s value
    let mut inputs: Vec<Operand> = Vec::new();
    let mut input_ptrs: Vec<usize> = Vec::new();
    let mut consts: Vec<SgConst> = Vec::new();

    for ci in 0..n {
        let t = cnodes[ci];
        let op = (*t).op;
        if op == GGML_OP_MUL_MAT {
            let w = (*t).src[0];
            if !is_const_weight(w) {
                return None; // runtime-operand matmul (e.g. attention KQ) — not bakeable yet
            }
            let act = resolve((*t).src[1], &cnodes, &ir_out, ci, &mut inputs, &mut input_ptrs)?;
            let ci_c = consts.len() as u32;
            consts.push(sg_const(w));
            ir_nodes.push(SgNode { op: SgOp::MatMul { n: (*w).ne[1] }, a: act, b: None, weight: Some(ci_c) });
        } else if op == GGML_OP_MUL || op == GGML_OP_ADD {
            if !same_2d((*t).src[0], (*t).src[1]) {
                return None;
            }
            let a = resolve((*t).src[0], &cnodes, &ir_out, ci, &mut inputs, &mut input_ptrs)?;
            let b = resolve((*t).src[1], &cnodes, &ir_out, ci, &mut inputs, &mut input_ptrs)?;
            let sgop = if op == GGML_OP_MUL { SgOp::Mul } else { SgOp::Add };
            ir_nodes.push(SgNode { op: sgop, a, b: Some(b), weight: None });
        } else if is_silu(t) {
            if !glue_2d(t) {
                return None;
            }
            let a = resolve((*t).src[0], &cnodes, &ir_out, ci, &mut inputs, &mut input_ptrs)?;
            ir_nodes.push(SgNode { op: SgOp::Silu, a, b: None, weight: None });
        } else if is_rms_norm(t) {
            if !glue_2d(t) {
                return None;
            }
            let a = resolve((*t).src[0], &cnodes, &ir_out, ci, &mut inputs, &mut input_ptrs)?;
            let eps = f32::from_bits((*t).op_params[0] as u32);
            ir_nodes.push(SgNode { op: SgOp::RmsNorm { eps }, a, b: None, weight: None });
        } else if is_swiglu(t) {
            // SwiGLU(g, u) = silu(g)·u → two IR nodes; `swapped` picks which operand gets the silu.
            let swapped = (*t).op_params[1] != 0;
            let (g, u) = if swapped { ((*t).src[1], (*t).src[0]) } else { ((*t).src[0], (*t).src[1]) };
            if !same_2d(g, u) {
                return None;
            }
            let gr = resolve(g, &cnodes, &ir_out, ci, &mut inputs, &mut input_ptrs)?;
            let ur = resolve(u, &cnodes, &ir_out, ci, &mut inputs, &mut input_ptrs)?;
            ir_nodes.push(SgNode { op: SgOp::Silu, a: gr, b: None, weight: None });
            let silu_idx = (ir_nodes.len() - 1) as u32;
            ir_nodes.push(SgNode { op: SgOp::Mul, a: Ref::Node(silu_idx), b: Some(ur), weight: None });
        } else {
            return None; // unsupported op
        }
        ir_out[ci] = (ir_nodes.len() - 1) as u32; // this node's value = the last IR node emitted
    }

    // The split's single output: the one node whose value no other split node consumes.
    let mut consumed = vec![false; n];
    for ci in 0..n {
        for &src in (*cnodes[ci]).src.iter() {
            if src.is_null() {
                continue;
            }
            for (j, &cn) in cnodes.iter().enumerate() {
                if cn == src {
                    consumed[j] = true;
                    break;
                }
            }
        }
    }
    let mut out_ci: Option<usize> = None;
    for ci in 0..n {
        if !consumed[ci] {
            if out_ci.is_some() {
                return None; // multiple outputs — not a single-output subgraph
            }
            out_ci = Some(ci);
        }
    }
    let out_t = cnodes[out_ci?];
    let m = (*out_t).ne[1];
    if m <= 0 {
        return None;
    }
    // Every activation must be `[K, m]`-shaped (so input_k = elems/m is exact at the shim).
    for inp in &inputs {
        if inp.elems % (m as usize) != 0 {
            return None;
        }
    }

    Some(Subgraph {
        m,
        inputs,
        consts,
        nodes: ir_nodes,
        output: ir_out[out_ci?],
        out: Operand::from_tensor(out_t),
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
