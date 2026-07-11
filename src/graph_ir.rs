//! Vendor-neutral graph IR captured from one ggml scheduler split.
//!
//! This is deliberately richer than the original pattern-specific `Subgraph`: values retain rank,
//! strides, dtype and storage role; nodes retain all ggml operands and opaque op parameters; graph
//! outputs and mutable state are explicit. Intel lowering may recognize profitable supernodes while
//! other vendors can consume the same representation without inheriting OpenVINO semantics.

use alloc::vec::Vec;

use crate::ffi::*;

pub type ValueId = u32;
pub const OP_PARAM_BYTES: usize = 64;

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireOp {
    MatMul = 1,
    Mul = 2,
    Add = 3,
    RmsNorm = 4,
    Silu = 5,
    Reshape = 6,
    View = 7,
    Cont = 8,
    Transpose = 9,
    Concat = 10,
    Copy = 11,
    SsmConv = 12,
    Rope = 13,
    SoftMax = 14,
    GetRows = 15,
    Argsort = 16,
    MatMulId = 17,
    Div = 18,
    Scale = 19,
    Clamp = 20,
    Permute = 21,
    FlashAttention = 22,
    Swiglu = 23,
    Sigmoid = 24,
    SumRows = 25,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    Static(i64),
    /// Runtime dimension bounded by `max`; used by bucketed prefill/continuous batching.
    #[allow(dead_code)] // represented on the wire; concrete ggml captures are static today
    Bounded {
        min: i64,
        max: i64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorRole {
    Constant,
    RuntimeInput,
    MutableState,
    Intermediate,
    Output,
}

#[derive(Clone, Debug)]
pub struct TensorDesc {
    pub dtype: ggml_type,
    pub dims: [Dim; 4],
    pub strides: [usize; 4],
    pub role: TensorRole,
    pub data: usize,
    pub owned: bool,
}

#[derive(Clone, Debug)]
pub struct Node {
    /// Raw `ggml_op`. Keeping the source opcode makes the capture forward-compatible; lowerers
    /// advertise the subset and op-parameter encodings they understand.
    pub op: ggml_op,
    pub inputs: Vec<ValueId>,
    pub output: ValueId,
    pub op_params: [u8; OP_PARAM_BYTES],
}

#[derive(Clone, Debug)]
pub struct GraphIr {
    pub generation: u64,
    pub values: Vec<TensorDesc>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

fn has_name_prefix(t: &ggml_tensor, prefix: &[u8]) -> bool {
    if prefix.len() > t.name.len() {
        return false;
    }
    t.name
        .iter()
        .take(prefix.len())
        .zip(prefix)
        .all(|(&a, &b)| a as u8 == b)
}

unsafe fn describe(t: *const ggml_tensor, role: TensorRole) -> TensorDesc {
    let x = &*t;
    TensorDesc {
        dtype: x.type_,
        dims: [
            Dim::Static(x.ne[0]),
            Dim::Static(x.ne[1]),
            Dim::Static(x.ne[2]),
            Dim::Static(x.ne[3]),
        ],
        strides: x.nb,
        role,
        data: x.data as usize,
        owned: crate::backend::owns_tensor(t),
    }
}

fn find_ptr(ptrs: &[usize], p: usize) -> Option<ValueId> {
    ptrs.iter().position(|&x| x == p).map(|i| i as ValueId)
}

/// Capture a concrete scheduler split. Unsupported operations are still represented; capability
/// decisions belong to the vendor lowerer rather than the capture layer.
pub unsafe fn capture(cgraph: *mut ggml_cgraph) -> Option<GraphIr> {
    if cgraph.is_null() {
        return None;
    }
    let n = ggml_graph_n_nodes(cgraph);
    if n <= 0 {
        return None;
    }

    let mut graph = GraphIr {
        generation: crate::backend::buffer_generation(),
        values: Vec::new(),
        nodes: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    let mut value_ptrs: Vec<usize> = Vec::new();
    let mut node_ptrs: Vec<usize> = Vec::with_capacity(n as usize);
    let mut consumed: Vec<bool> = Vec::with_capacity(n as usize);

    for i in 0..n {
        let t = ggml_graph_node(cgraph, i);
        if t.is_null() {
            return None;
        }
        // Empty state-selection views carry no bytes and require no device work. ggml may attach
        // them to a non-empty scheduler split even when `supports_op` rejected the zero shape.
        // Omitting them keeps the static NPU graph valid; any non-empty consumer would reintroduce
        // the tensor as an external input and fail capability validation conservatively.
        if (*t).ne.iter().any(|&dim| dim <= 0) {
            continue;
        }
        node_ptrs.push(t as usize);
        consumed.push(false);

        let mut inputs = Vec::new();
        for (src_index, src) in (*t).src.into_iter().enumerate() {
            if src.is_null() {
                continue;
            }
            let sp = src as usize;
            let id = if let Some(id) = find_ptr(&value_ptrs, sp) {
                if let Some(pos) = node_ptrs.iter().position(|&p| p == sp) {
                    consumed[pos] = true;
                }
                id
            } else {
                let sx = &*src;
                let baked_operand = matches!(
                    ((*t).op, src_index),
                    (GGML_OP_MUL_MAT, 0) | (GGML_OP_MUL_MAT_ID, 0) | (GGML_OP_SSM_CONV, 1)
                );
                let role = if has_name_prefix(sx, b"cache_") {
                    TensorRole::MutableState
                } else if baked_operand && sx.op == GGML_OP_NONE && !sx.data.is_null() {
                    TensorRole::Constant
                } else {
                    TensorRole::RuntimeInput
                };
                let id = graph.values.len() as ValueId;
                graph.values.push(describe(src, role));
                value_ptrs.push(sp);
                graph.inputs.push(id);
                id
            };
            inputs.push(id);
        }

        let output = graph.values.len() as ValueId;
        graph.values.push(describe(t, TensorRole::Intermediate));
        value_ptrs.push(t as usize);

        let mut params = [0u8; OP_PARAM_BYTES];
        let n_params = core::cmp::min(params.len(), core::mem::size_of_val(&(*t).op_params));
        core::ptr::copy_nonoverlapping(
            (*t).op_params.as_ptr() as *const u8,
            params.as_mut_ptr(),
            n_params,
        );
        graph.nodes.push(Node {
            op: (*t).op,
            inputs,
            output,
            op_params: params,
        });
    }

    for (i, &used) in consumed.iter().enumerate() {
        if !used {
            let id = find_ptr(&value_ptrs, node_ptrs[i])?;
            graph.values[id as usize].role = TensorRole::Output;
            graph.outputs.push(id);
        }
    }
    Some(graph)
}

impl GraphIr {
    pub fn wire_op(node: &Node) -> Option<WireOp> {
        Some(match node.op {
            GGML_OP_MUL_MAT => WireOp::MatMul,
            GGML_OP_MUL => WireOp::Mul,
            GGML_OP_ADD => WireOp::Add,
            GGML_OP_RMS_NORM => WireOp::RmsNorm,
            GGML_OP_RESHAPE => WireOp::Reshape,
            GGML_OP_VIEW => WireOp::View,
            GGML_OP_CONT => WireOp::Cont,
            GGML_OP_TRANSPOSE => WireOp::Transpose,
            GGML_OP_CONCAT => WireOp::Concat,
            GGML_OP_CPY => WireOp::Copy,
            GGML_OP_SSM_CONV => WireOp::SsmConv,
            GGML_OP_ROPE => WireOp::Rope,
            GGML_OP_SOFT_MAX => WireOp::SoftMax,
            GGML_OP_GET_ROWS => WireOp::GetRows,
            GGML_OP_ARGSORT => WireOp::Argsort,
            GGML_OP_MUL_MAT_ID => WireOp::MatMulId,
            GGML_OP_DIV => WireOp::Div,
            GGML_OP_SCALE => WireOp::Scale,
            GGML_OP_CLAMP => WireOp::Clamp,
            GGML_OP_PERMUTE => WireOp::Permute,
            GGML_OP_FLASH_ATTN_EXT => WireOp::FlashAttention,
            GGML_OP_GLU => WireOp::Swiglu,
            GGML_OP_SUM_ROWS => WireOp::SumRows,
            GGML_OP_UNARY => {
                let subop = i32::from_ne_bytes(node.op_params[..4].try_into().ok()?);
                if subop == GGML_UNARY_OP_SILU as i32 {
                    WireOp::Silu
                } else if subop == GGML_UNARY_OP_SIGMOID as i32 {
                    WireOp::Sigmoid
                } else {
                    return None;
                }
            }
            _ => return None,
        })
    }

    /// Stable structural identity within a model-buffer generation. Data addresses distinguish
    /// baked constants; runtime binding addresses intentionally do not.
    pub fn signature(&self) -> u64 {
        fn fold(mut h: u64, v: u64) -> u64 {
            h ^= v;
            h.wrapping_mul(0x0000_0100_0000_01b3)
        }
        let mut h = fold(0xcbf2_9ce4_8422_2325, self.generation);
        for node in &self.nodes {
            h = fold(h, node.op as u64);
            h = fold(h, node.inputs.len() as u64);
            for &i in &node.inputs {
                h = fold(h, i as u64);
            }
            let out = &self.values[node.output as usize];
            h = fold(h, out.dtype as u64);
            for d in out.dims {
                let v = match d {
                    Dim::Static(x) => x as u64,
                    Dim::Bounded { min, max } => (min as u64).rotate_left(17) ^ max as u64,
                };
                h = fold(h, v);
            }
            for stride in out.strides {
                h = fold(h, stride as u64);
            }
        }
        for &id in &self.inputs {
            let t = &self.values[id as usize];
            if t.role == TensorRole::Constant {
                h = fold(h, t.data as u64);
            }
        }
        h
    }
}

/// Per-node scheduler capability probe matching the initial Intel Graph IR lowerer. Full-graph
/// validation still runs at `graph_compute`; this function only groups compatible nodes into a
/// candidate split.
pub unsafe fn wire_op_for_tensor(t: *const ggml_tensor) -> Option<WireOp> {
    if t.is_null() {
        return None;
    }
    let x = &*t;
    // ggml deliberately emits zero-length state/gather views for empty sequence selections.
    // They are valid CPU no-ops but cannot be represented by the static OpenVINO/NPU ABI.
    if x.ne.iter().any(|&dim| dim <= 0) {
        return None;
    }
    Some(match x.op {
        GGML_OP_MUL_MAT => WireOp::MatMul,
        GGML_OP_MUL => WireOp::Mul,
        GGML_OP_ADD => WireOp::Add,
        GGML_OP_RMS_NORM => WireOp::RmsNorm,
        GGML_OP_RESHAPE => WireOp::Reshape,
        GGML_OP_VIEW => WireOp::View,
        GGML_OP_CONT => WireOp::Cont,
        GGML_OP_TRANSPOSE => WireOp::Transpose,
        GGML_OP_CONCAT => WireOp::Concat,
        GGML_OP_CPY => WireOp::Copy,
        GGML_OP_SSM_CONV => WireOp::SsmConv,
        GGML_OP_GET_ROWS => WireOp::GetRows,
        GGML_OP_SCALE => WireOp::Scale,
        GGML_OP_DIV => WireOp::Div,
        GGML_OP_CLAMP => WireOp::Clamp,
        GGML_OP_ARGSORT => WireOp::Argsort,
        GGML_OP_MUL_MAT_ID => WireOp::MatMulId,
        GGML_OP_SUM_ROWS => WireOp::SumRows,
        GGML_OP_SOFT_MAX => {
            let max_bias = f32::from_ne_bytes(x.op_params[1].to_ne_bytes());
            if max_bias != 0.0 {
                return None;
            }
            WireOp::SoftMax
        }
        GGML_OP_PERMUTE => WireOp::Permute,
        GGML_OP_GLU if x.op_params[0] == GGML_GLU_OP_SWIGLU as i32 => WireOp::Swiglu,
        GGML_OP_UNARY if x.op_params[0] == GGML_UNARY_OP_SILU as i32 => WireOp::Silu,
        GGML_OP_UNARY if x.op_params[0] == GGML_UNARY_OP_SIGMOID as i32 => WireOp::Sigmoid,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn value(role: TensorRole, data: usize, dim: Dim) -> TensorDesc {
        TensorDesc {
            dtype: GGML_TYPE_F32,
            dims: [dim, Dim::Static(1), Dim::Static(1), Dim::Static(1)],
            strides: [4, 4, 4, 4],
            role,
            data,
            owned: false,
        }
    }

    #[test]
    fn signature_tracks_structure_shape_and_constant_identity() {
        let make = |constant, dim| GraphIr {
            generation: 7,
            values: vec![
                value(TensorRole::Constant, constant, Dim::Static(4)),
                value(TensorRole::RuntimeInput, 0, Dim::Static(4)),
                value(TensorRole::Output, 0, dim),
            ],
            nodes: vec![Node {
                op: GGML_OP_MUL_MAT,
                inputs: vec![0, 1],
                output: 2,
                op_params: [0; OP_PARAM_BYTES],
            }],
            inputs: vec![0, 1],
            outputs: vec![2],
        };
        assert_eq!(
            make(11, Dim::Static(4)).signature(),
            make(11, Dim::Static(4)).signature()
        );
        assert_ne!(
            make(11, Dim::Static(4)).signature(),
            make(12, Dim::Static(4)).signature()
        );
        assert_ne!(
            make(11, Dim::Static(4)).signature(),
            make(11, Dim::Bounded { min: 1, max: 4 }).signature()
        );
    }

    #[test]
    fn every_corpus_wire_opcode_has_a_stable_nonzero_value() {
        let ops = [
            WireOp::MatMul,
            WireOp::SsmConv,
            WireOp::MatMulId,
            WireOp::FlashAttention,
            WireOp::Swiglu,
        ];
        assert!(ops.iter().all(|op| *op as i32 > 0));
    }
}
