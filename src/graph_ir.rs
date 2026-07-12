//! Vendor-neutral graph IR captured from one ggml scheduler split.
//!
//! This is deliberately richer than the original pattern-specific `Subgraph`: values retain rank,
//! strides, dtype and storage role; nodes retain all ggml operands and opaque op parameters; graph
//! outputs and mutable state are explicit. Intel lowering may recognize profitable supernodes while
//! other vendors can consume the same representation without inheriting OpenVINO semantics.

use alloc::vec;
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
    SetRows = 26,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpTraits {
    pub metadata: bool,
    pub stateful: bool,
    pub attention: bool,
    pub moe: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DTypeRule {
    Float,
    FloatWithI32Index,
    AnyBitwiseCompatible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeRule {
    Elementwise,
    MatMul,
    Layout,
    Gather,
    Scatter,
    Reduction,
    Attention,
    Custom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryEffect {
    Pure,
    AliasInput,
    WriteState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostModel {
    Metadata,
    Elements,
    MatMulMacs,
    AttentionQuadratic,
    Stateful,
}

/// How ggml should use a correct vendor lowering. This is descriptive capability metadata:
/// profitability remains exclusively in `policy.rs`/`offload_op`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlacementClass {
    /// Expensive compute that can anchor a scheduler split.
    Anchor,
    /// Correct only as inexpensive glue around an anchor; do not pull it by itself.
    FusibleOnly,
    /// May anchor a split once the profitability policy approves its concrete shape.
    StandaloneIfProfitable,
    /// Shape/layout bookkeeping that should travel with an anchored split.
    Metadata,
    /// Reads or writes persistent model state and must preserve ordering/alias semantics.
    Stateful,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpSpec {
    pub wire: WireOp,
    pub min_inputs: u8,
    pub max_inputs: u8,
    pub traits: OpTraits,
    pub dtype_rule: DTypeRule,
    pub shape_rule: ShapeRule,
    pub memory_effect: MemoryEffect,
    pub cost_model: CostModel,
    pub placement: PlacementClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpReject {
    NullTensor,
    EmptyShape,
    UnsupportedOp,
    UnsupportedVariant,
    UnsupportedDType,
    UnsupportedAliasing,
    InvalidArity,
}

/// One semantic contract used by scheduler probing, graph capture, vendor feasibility and tests.
pub trait OperationContract {
    unsafe fn decode_tensor(t: *const ggml_tensor) -> Result<OpSpec, OpReject>;
    fn decode_node(node: &Node) -> Result<OpSpec, OpReject>;
}

pub struct OpRegistry;

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
    pub spec: Result<OpSpec, OpReject>,
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

fn traits_for(wire: WireOp) -> OpTraits {
    OpTraits {
        metadata: matches!(
            wire,
            WireOp::Reshape | WireOp::View | WireOp::Cont | WireOp::Transpose | WireOp::Permute
        ),
        stateful: matches!(wire, WireOp::SsmConv | WireOp::SetRows),
        attention: matches!(
            wire,
            WireOp::Rope
                | WireOp::SoftMax
                | WireOp::Permute
                | WireOp::FlashAttention
                | WireOp::SetRows
        ),
        moe: matches!(
            wire,
            WireOp::Argsort | WireOp::MatMulId | WireOp::Sigmoid | WireOp::SumRows
        ),
    }
}

fn rules_for(
    wire: WireOp,
) -> (
    DTypeRule,
    ShapeRule,
    MemoryEffect,
    CostModel,
    PlacementClass,
) {
    use CostModel::*;
    use DTypeRule::*;
    use MemoryEffect::*;
    use ShapeRule::*;

    match wire {
        WireOp::Reshape | WireOp::View | WireOp::Transpose | WireOp::Permute => (
            AnyBitwiseCompatible,
            Layout,
            AliasInput,
            Metadata,
            PlacementClass::Metadata,
        ),
        WireOp::Cont => (
            AnyBitwiseCompatible,
            Layout,
            Pure,
            Elements,
            PlacementClass::Metadata,
        ),
        WireOp::Copy => (
            AnyBitwiseCompatible,
            Layout,
            Pure,
            Elements,
            PlacementClass::FusibleOnly,
        ),
        WireOp::MatMul | WireOp::MatMulId => {
            (Float, MatMul, Pure, MatMulMacs, PlacementClass::Anchor)
        }
        WireOp::GetRows | WireOp::Argsort => (
            FloatWithI32Index,
            Gather,
            Pure,
            Elements,
            PlacementClass::FusibleOnly,
        ),
        WireOp::SetRows => (
            FloatWithI32Index,
            Scatter,
            WriteState,
            Stateful,
            PlacementClass::Stateful,
        ),
        WireOp::SumRows => (
            Float,
            Reduction,
            Pure,
            Elements,
            PlacementClass::FusibleOnly,
        ),
        WireOp::FlashAttention => (
            Float,
            Attention,
            Pure,
            AttentionQuadratic,
            PlacementClass::Anchor,
        ),
        WireOp::Rope => (
            Float,
            Attention,
            Pure,
            AttentionQuadratic,
            PlacementClass::StandaloneIfProfitable,
        ),
        WireOp::SoftMax => (
            Float,
            Attention,
            Pure,
            AttentionQuadratic,
            PlacementClass::FusibleOnly,
        ),
        WireOp::SsmConv => (Float, Custom, Pure, Stateful, PlacementClass::Stateful),
        WireOp::Concat => (Float, Custom, Pure, Elements, PlacementClass::FusibleOnly),
        _ => (
            Float,
            Elementwise,
            Pure,
            Elements,
            PlacementClass::FusibleOnly,
        ),
    }
}

fn decode_spec(
    op: ggml_op,
    unary_or_glu: i32,
    softmax_max_bias: f32,
    rope_mode: i32,
    n_inputs: usize,
) -> Result<OpSpec, OpReject> {
    let (wire, min_inputs, max_inputs) = match op {
        GGML_OP_MUL_MAT => (WireOp::MatMul, 2, 2),
        GGML_OP_MUL => (WireOp::Mul, 2, 2),
        GGML_OP_ADD => (WireOp::Add, 2, 2),
        GGML_OP_RMS_NORM => (WireOp::RmsNorm, 1, 1),
        GGML_OP_RESHAPE => (WireOp::Reshape, 1, 1),
        GGML_OP_VIEW => (WireOp::View, 1, 1),
        GGML_OP_CONT => (WireOp::Cont, 1, 1),
        GGML_OP_TRANSPOSE => (WireOp::Transpose, 1, 1),
        GGML_OP_CONCAT => (WireOp::Concat, 2, 2),
        GGML_OP_CPY => (WireOp::Copy, 1, 2),
        GGML_OP_SSM_CONV => (WireOp::SsmConv, 2, 2),
        GGML_OP_ROPE if matches!(rope_mode, 0 | 2) => (WireOp::Rope, 2, 2),
        GGML_OP_ROPE => return Err(OpReject::UnsupportedVariant),
        GGML_OP_SOFT_MAX if softmax_max_bias == 0.0 => (WireOp::SoftMax, 1, 2),
        GGML_OP_SOFT_MAX => return Err(OpReject::UnsupportedVariant),
        GGML_OP_GET_ROWS => (WireOp::GetRows, 2, 2),
        GGML_OP_SET_ROWS => (WireOp::SetRows, 3, 3),
        GGML_OP_ARGSORT => (WireOp::Argsort, 1, 1),
        GGML_OP_MUL_MAT_ID => (WireOp::MatMulId, 3, 3),
        GGML_OP_DIV => (WireOp::Div, 2, 2),
        GGML_OP_SCALE => (WireOp::Scale, 1, 1),
        GGML_OP_CLAMP => (WireOp::Clamp, 1, 1),
        GGML_OP_PERMUTE => (WireOp::Permute, 1, 1),
        GGML_OP_FLASH_ATTN_EXT => (WireOp::FlashAttention, 3, 4),
        GGML_OP_GLU if unary_or_glu == GGML_GLU_OP_SWIGLU as i32 => (WireOp::Swiglu, 2, 2),
        GGML_OP_GLU => return Err(OpReject::UnsupportedVariant),
        GGML_OP_SUM_ROWS => (WireOp::SumRows, 1, 1),
        GGML_OP_UNARY if unary_or_glu == GGML_UNARY_OP_SILU as i32 => (WireOp::Silu, 1, 1),
        GGML_OP_UNARY if unary_or_glu == GGML_UNARY_OP_SIGMOID as i32 => (WireOp::Sigmoid, 1, 1),
        GGML_OP_UNARY => return Err(OpReject::UnsupportedVariant),
        _ => return Err(OpReject::UnsupportedOp),
    };
    if n_inputs < min_inputs as usize || n_inputs > max_inputs as usize {
        return Err(OpReject::InvalidArity);
    }
    let (dtype_rule, shape_rule, memory_effect, cost_model, placement) = rules_for(wire);
    Ok(OpSpec {
        wire,
        min_inputs,
        max_inputs,
        traits: traits_for(wire),
        dtype_rule,
        shape_rule,
        memory_effect,
        cost_model,
        placement,
    })
}

fn is_float_type(dtype: ggml_type) -> bool {
    matches!(dtype, GGML_TYPE_F32 | GGML_TYPE_F16)
}

fn is_index_type(dtype: ggml_type) -> bool {
    matches!(dtype, GGML_TYPE_I32 | GGML_TYPE_I64)
}

unsafe fn validate_tensor_dtypes(tensor: &ggml_tensor, spec: OpSpec) -> Result<(), OpReject> {
    let src_type = |index: usize| {
        tensor
            .src
            .get(index)
            .copied()
            .filter(|src| !src.is_null())
            .map(|src| (*src).type_)
    };
    let baked = |index: usize| {
        matches!(
            (tensor.op, index),
            (GGML_OP_MUL_MAT, 0) | (GGML_OP_MUL_MAT_ID, 0) | (GGML_OP_SSM_CONV, 1)
        ) && tensor.src.get(index).copied().is_some_and(|src| {
            !src.is_null() && (*src).op == GGML_OP_NONE && !(*src).data.is_null()
        })
    };

    let output_ok = match spec.wire {
        WireOp::Argsort => tensor.type_ == GGML_TYPE_I32,
        WireOp::Reshape | WireOp::View | WireOp::Cont | WireOp::Transpose | WireOp::Permute => {
            is_float_type(tensor.type_) || is_index_type(tensor.type_)
        }
        _ => is_float_type(tensor.type_),
    };
    if !output_ok {
        return Err(OpReject::UnsupportedDType);
    }

    for (index, src) in tensor.src.iter().copied().enumerate() {
        if src.is_null() || baked(index) {
            continue;
        }
        if !is_float_type((*src).type_) && !is_index_type((*src).type_) {
            return Err(OpReject::UnsupportedDType);
        }
    }

    let valid = match spec.wire {
        WireOp::GetRows => {
            src_type(0).is_some_and(is_float_type) && src_type(1).is_some_and(is_index_type)
        }
        WireOp::SetRows => {
            src_type(0).is_some_and(is_float_type)
                && src_type(1).is_some_and(is_index_type)
                && src_type(2).is_some_and(is_float_type)
        }
        WireOp::Argsort => src_type(0).is_some_and(is_float_type),
        WireOp::MatMulId => {
            src_type(1).is_some_and(is_float_type) && src_type(2).is_some_and(is_index_type)
        }
        WireOp::Rope => {
            src_type(0).is_some_and(is_float_type) && src_type(1).is_some_and(is_index_type)
        }
        WireOp::FlashAttention => tensor
            .src
            .iter()
            .take(spec.max_inputs as usize)
            .copied()
            .filter(|src| !src.is_null())
            .all(|src| is_float_type((*src).type_)),
        WireOp::Reshape | WireOp::View | WireOp::Cont | WireOp::Transpose | WireOp::Permute => {
            tensor
                .src
                .iter()
                .copied()
                .filter(|src| !src.is_null())
                .all(|src| is_float_type((*src).type_) || is_index_type((*src).type_))
        }
        _ => tensor
            .src
            .iter()
            .enumerate()
            .filter(|(_, src)| !src.is_null())
            .all(|(index, src)| baked(index) || is_float_type((*(*src)).type_)),
    };
    if !valid {
        return Err(OpReject::UnsupportedDType);
    }
    if spec.wire == WireOp::SetRows {
        let base = tensor.src[2];
        if !base.is_null() && !tensor.data.is_null() && (*base).data == tensor.data {
            return Err(OpReject::UnsupportedAliasing);
        }
    }
    Ok(())
}

impl OperationContract for OpRegistry {
    unsafe fn decode_tensor(t: *const ggml_tensor) -> Result<OpSpec, OpReject> {
        if t.is_null() {
            return Err(OpReject::NullTensor);
        }
        let x = &*t;
        if x.ne.iter().any(|&dim| dim <= 0) {
            return Err(OpReject::EmptyShape);
        }
        let n_inputs = x.src.iter().filter(|src| !src.is_null()).count();
        if x.op == GGML_OP_ROPE
            && (x.op_params[1] as i64 != x.ne[0] || f32::from_bits(x.op_params[7] as u32) != 0.0)
        {
            return Err(OpReject::UnsupportedVariant);
        }
        if x.op == GGML_OP_FLASH_ATTN_EXT
            && (f32::from_bits(x.op_params[1] as u32) != 0.0
                || f32::from_bits(x.op_params[2] as u32) != 0.0)
        {
            return Err(OpReject::UnsupportedVariant);
        }
        let subop = x.op_params[0];
        let max_bias = f32::from_bits(x.op_params[1] as u32);
        let rope_mode = x.op_params[2];
        let spec = decode_spec(x.op, subop, max_bias, rope_mode, n_inputs)?;
        validate_tensor_dtypes(x, spec)?;
        Ok(spec)
    }

    fn decode_node(node: &Node) -> Result<OpSpec, OpReject> {
        let subop = i32::from_ne_bytes(
            node.op_params[..4]
                .try_into()
                .map_err(|_| OpReject::UnsupportedVariant)?,
        );
        let max_bias = f32::from_ne_bytes(
            node.op_params[4..8]
                .try_into()
                .map_err(|_| OpReject::UnsupportedVariant)?,
        );
        let rope_mode = i32::from_ne_bytes(
            node.op_params[8..12]
                .try_into()
                .map_err(|_| OpReject::UnsupportedVariant)?,
        );
        if node.op == GGML_OP_FLASH_ATTN_EXT
            && (max_bias != 0.0
                || f32::from_ne_bytes(
                    node.op_params[8..12]
                        .try_into()
                        .map_err(|_| OpReject::UnsupportedVariant)?,
                ) != 0.0)
        {
            return Err(OpReject::UnsupportedVariant);
        }
        if node.op == GGML_OP_ROPE
            && f32::from_ne_bytes(
                node.op_params[28..32]
                    .try_into()
                    .map_err(|_| OpReject::UnsupportedVariant)?,
            ) != 0.0
        {
            return Err(OpReject::UnsupportedVariant);
        }
        decode_spec(node.op, subop, max_bias, rope_mode, node.inputs.len())
    }
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
        let mut output_role = TensorRole::Intermediate;
        if matches!((*t).op, GGML_OP_CPY | GGML_OP_SET_ROWS)
            && (inputs
                .iter()
                .any(|&id| graph.values[id as usize].role == TensorRole::MutableState)
                || has_name_prefix(&*t, b"cache_"))
        {
            output_role = TensorRole::MutableState;
        }
        graph.values.push(describe(t, output_role));
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
            spec: OpRegistry::decode_tensor(t),
            inputs,
            output,
            op_params: params,
        });
    }

    for (i, &used) in consumed.iter().enumerate() {
        let id = find_ptr(&value_ptrs, node_ptrs[i])?;
        if !used || graph.values[id as usize].role == TensorRole::MutableState {
            if !used && graph.values[id as usize].role != TensorRole::MutableState {
                graph.values[id as usize].role = TensorRole::Output;
            }
            if graph.outputs.contains(&id) {
                continue;
            }
            graph.outputs.push(id);
        }
    }
    Some(graph)
}

impl GraphIr {
    pub fn wire_op(node: &Node) -> Option<WireOp> {
        let captured = *node.spec.as_ref().ok()?;
        (OpRegistry::decode_node(node).ok() == Some(captured)).then_some(captured.wire)
    }

    /// Split a scheduler graph into independently compilable, sequential artifacts. Crossing
    /// values retain their ggml-allocated data pointers and become an output of one partition plus
    /// a runtime input of the next. Limiting routed expert banks is more important than raw node
    /// count: each `MatMulId` carries a very large constant bank and must become cacheable on its
    /// own rather than waiting for a monolithic multi-bank compile.
    pub fn execution_partitions(&self, max_nodes: usize, max_matmul_id: usize) -> Vec<GraphIr> {
        if self.nodes.is_empty() || max_nodes == 0 || max_matmul_id == 0 {
            return Vec::new();
        }
        let mut ranges = Vec::new();
        let mut start = 0usize;
        while start < self.nodes.len() {
            let mut end = start;
            let mut routed_banks = 0usize;
            while end < self.nodes.len() {
                let routed = Self::wire_op(&self.nodes[end]) == Some(WireOp::MatMulId);
                if end > start
                    && (end - start >= max_nodes || (routed && routed_banks >= max_matmul_id))
                {
                    break;
                }
                routed_banks += usize::from(routed);
                end += 1;
            }
            ranges.push((start, end));
            start = end;
        }
        if ranges.len() == 1 {
            return vec![self.clone()];
        }

        ranges
            .into_iter()
            .filter_map(|(start, end)| self.partition_range(start, end))
            .collect()
    }

    fn partition_range(&self, start: usize, end: usize) -> Option<GraphIr> {
        if start >= end || end > self.nodes.len() {
            return None;
        }
        let mut needed = vec![false; self.values.len()];
        let mut produced = vec![false; self.values.len()];
        for node in &self.nodes[start..end] {
            for &input in &node.inputs {
                *needed.get_mut(input as usize)? = true;
            }
            *needed.get_mut(node.output as usize)? = true;
            *produced.get_mut(node.output as usize)? = true;
        }

        let mut consumed_later = vec![false; self.values.len()];
        for node in &self.nodes[end..] {
            for &input in &node.inputs {
                *consumed_later.get_mut(input as usize)? = true;
            }
        }

        let mut remap = vec![None; self.values.len()];
        let mut values = Vec::new();
        for (old, value) in self.values.iter().enumerate() {
            if needed[old] {
                remap[old] = Some(values.len() as ValueId);
                values.push(value.clone());
            }
        }

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for (old, &new) in remap.iter().enumerate() {
            let Some(new) = new else {
                continue;
            };
            if self.values[old].role == TensorRole::Constant {
                values[new as usize].role = TensorRole::Constant;
            } else if !produced[old] {
                values[new as usize].role = if self.values[old].role == TensorRole::MutableState {
                    TensorRole::MutableState
                } else {
                    TensorRole::RuntimeInput
                };
                inputs.push(new);
            } else {
                let crosses = consumed_later[old]
                    || self.outputs.contains(&(old as ValueId))
                    || self.values[old].role == TensorRole::MutableState;
                values[new as usize].role = if crosses {
                    if self.values[old].role == TensorRole::MutableState {
                        TensorRole::MutableState
                    } else {
                        TensorRole::Output
                    }
                } else {
                    TensorRole::Intermediate
                };
                if crosses {
                    outputs.push(new);
                }
            }
        }
        if outputs.is_empty() {
            let old = self.nodes[end - 1].output as usize;
            let new = remap.get(old).copied().flatten()?;
            values[new as usize].role = TensorRole::Output;
            outputs.push(new);
        }

        let nodes = self.nodes[start..end]
            .iter()
            .map(|node| {
                Some(Node {
                    op: node.op,
                    spec: node.spec,
                    inputs: node
                        .inputs
                        .iter()
                        .map(|&id| remap.get(id as usize).copied().flatten())
                        .collect::<Option<Vec<_>>>()?,
                    output: remap.get(node.output as usize).copied().flatten()?,
                    op_params: node.op_params,
                })
            })
            .collect::<Option<Vec<_>>>()?;

        Some(GraphIr {
            generation: self.generation,
            values,
            nodes,
            inputs,
            outputs,
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
            h = fold(h, id as u64);
            h = fold(h, t.role as u64);
        }
        for (id, t) in self.values.iter().enumerate() {
            if t.role == TensorRole::Constant {
                h = fold(h, id as u64);
                h = fold(h, t.data as u64);
            }
        }
        for &id in &self.outputs {
            h = fold(h, id as u64);
        }
        h
    }
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
                spec: decode_spec(GGML_OP_MUL_MAT, 0, 0.0, 0, 2),
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
        let mut alternate_output = make(11, Dim::Static(4));
        alternate_output.outputs = vec![1];
        assert_ne!(
            make(11, Dim::Static(4)).signature(),
            alternate_output.signature()
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

    #[test]
    fn operation_contract_rejects_semantic_variants_and_bad_arity() {
        assert_eq!(
            decode_spec(GGML_OP_SOFT_MAX, 0, 1.0, 0, 2),
            Err(OpReject::UnsupportedVariant)
        );
        assert_eq!(
            decode_spec(GGML_OP_GLU, GGML_GLU_OP_GEGLU as i32, 0.0, 0, 2),
            Err(OpReject::UnsupportedVariant)
        );
        assert_eq!(
            decode_spec(GGML_OP_MUL_MAT, 0, 0.0, 0, 1),
            Err(OpReject::InvalidArity)
        );
        assert_eq!(
            decode_spec(GGML_OP_SET_ROWS, 0, 0.0, 0, 3)
                .map(|spec| (spec.wire, spec.traits.stateful)),
            Ok((WireOp::SetRows, true))
        );
    }

    #[test]
    fn operation_contract_assigns_explicit_placement_classes() {
        assert_eq!(
            decode_spec(GGML_OP_MUL_MAT, 0, 0.0, 0, 2).map(|spec| spec.placement),
            Ok(PlacementClass::Anchor)
        );
        assert_eq!(
            decode_spec(GGML_OP_UNARY, GGML_UNARY_OP_SILU as i32, 0.0, 0, 1)
                .map(|spec| spec.placement),
            Ok(PlacementClass::FusibleOnly)
        );
        assert_eq!(
            decode_spec(GGML_OP_ROPE, 0, 0.0, 0, 2).map(|spec| spec.placement),
            Ok(PlacementClass::StandaloneIfProfitable)
        );
        assert_eq!(
            decode_spec(GGML_OP_VIEW, 0, 0.0, 0, 1).map(|spec| spec.placement),
            Ok(PlacementClass::Metadata)
        );
        assert_eq!(
            decode_spec(GGML_OP_SSM_CONV, 0, 0.0, 0, 2).map(|spec| spec.placement),
            Ok(PlacementClass::Stateful)
        );
    }
}
