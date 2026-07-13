//! Graph semantics and the current ggml capture adapter.
//!
//! This is deliberately richer than the original pattern-specific `Subgraph`: values retain rank,
//! strides, dtype and storage role; nodes retain all ggml operands and opaque op parameters; graph
//! outputs and mutable state are explicit. Capability and profitability decisions remain outside
//! capture. The source-API decoding in this file is an adapter concern and is intentionally not
//! treated as part of the portable graph wire contract.

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

impl WireOp {
    /// Every wire opcode the backend can name. Coverage gates require a matrix row per entry.
    pub const ALL: &'static [WireOp] = &[
        WireOp::MatMul,
        WireOp::Mul,
        WireOp::Add,
        WireOp::RmsNorm,
        WireOp::Silu,
        WireOp::Reshape,
        WireOp::View,
        WireOp::Cont,
        WireOp::Transpose,
        WireOp::Concat,
        WireOp::Copy,
        WireOp::SsmConv,
        WireOp::Rope,
        WireOp::SoftMax,
        WireOp::GetRows,
        WireOp::Argsort,
        WireOp::MatMulId,
        WireOp::Div,
        WireOp::Scale,
        WireOp::Clamp,
        WireOp::Permute,
        WireOp::FlashAttention,
        WireOp::Swiglu,
        WireOp::Sigmoid,
        WireOp::SumRows,
        WireOp::SetRows,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            WireOp::MatMul => "MatMul",
            WireOp::Mul => "Mul",
            WireOp::Add => "Add",
            WireOp::RmsNorm => "RmsNorm",
            WireOp::Silu => "Silu",
            WireOp::Reshape => "Reshape",
            WireOp::View => "View",
            WireOp::Cont => "Cont",
            WireOp::Transpose => "Transpose",
            WireOp::Concat => "Concat",
            WireOp::Copy => "Copy",
            WireOp::SsmConv => "SsmConv",
            WireOp::Rope => "Rope",
            WireOp::SoftMax => "SoftMax",
            WireOp::GetRows => "GetRows",
            WireOp::Argsort => "Argsort",
            WireOp::MatMulId => "MatMulId",
            WireOp::Div => "Div",
            WireOp::Scale => "Scale",
            WireOp::Clamp => "Clamp",
            WireOp::Permute => "Permute",
            WireOp::FlashAttention => "FlashAttention",
            WireOp::Swiglu => "Swiglu",
            WireOp::Sigmoid => "Sigmoid",
            WireOp::SumRows => "SumRows",
            WireOp::SetRows => "SetRows",
        }
    }

    pub fn from_name(name: &str) -> Option<WireOp> {
        WireOp::ALL
            .iter()
            .copied()
            .find(|op| op.as_str() == name)
    }
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

/// How ggml should use a correct lowering. This is descriptive capability metadata:
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

/// One semantic contract used by scheduler probing, graph capture, feasibility checks, and tests.
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
    pub shared: Option<crate::interop::SharedAllocationView>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionReason {
    WholeGraph,
    DisconnectedRegion,
    MutableStateBoundary,
    ExpertBankLimit,
    FeasibilityLimit,
    /// Layout-only region inside a claimed split that also has real compute. OpenVINO cannot
    /// lower View/Reshape/Cont/Transpose/Permute alone, so the engine host-realizes them as
    /// glue rather than rejecting the whole claimed multi-op split (#5/#8).
    LayoutGlueHost,
}

impl PartitionReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WholeGraph => "whole_graph",
            Self::DisconnectedRegion => "disconnected_region",
            Self::MutableStateBoundary => "mutable_state_boundary",
            Self::ExpertBankLimit => "expert_bank_limit",
            Self::FeasibilityLimit => "feasibility_limit",
            Self::LayoutGlueHost => "layout_glue_host",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartitionPolicy {
    /// Routed expert banks are intentionally isolated: each can be hundreds of MiB and has
    /// independent selection semantics. Dense constants have no analogous arbitrary node cap.
    pub max_expert_banks: usize,
}

impl Default for PartitionPolicy {
    fn default() -> Self {
        Self {
            max_expert_banks: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExecutionPartition {
    pub graph: GraphIr,
    pub reason: PartitionReason,
    pub start: usize,
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
        shared: crate::backend::shared_view(t),
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
        // SoftMax / FlashAttention stay behind GGML_NPU_ATTENTION=full (runtime KQ/KQV core).
        // RoPE is QKV-adjacent position encoding (LFM NEOX mode 2 / normal mode 0); gating it
        // with SoftMax/FA forced prefill QKV-proj islands onto CPU/GPU (#10). Permute is layout
        // metadata like Reshape/View — fused-only via placement, not an attention-core gate.
        // SetRows is mutable-state scatter (ShortConv / KV cache), not SoftMax/FA (#9/#10).
        attention: matches!(wire, WireOp::SoftMax | WireOp::FlashAttention),
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
    matches!(dtype, GGML_TYPE_F32 | GGML_TYPE_F16 | GGML_TYPE_BF16)
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
/// decisions belong to execution adapters rather than the capture layer.
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
        // Omitting them keeps a statically shaped target graph valid; any non-empty consumer would reintroduce
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

    /// True when a node is shape/layout bookkeeping (View/Reshape/Cont/Transpose/Permute).
    pub fn node_is_layout_metadata(node: &Node) -> bool {
        node.spec.ok().is_some_and(|spec| {
            spec.traits.metadata || spec.placement == PlacementClass::Metadata
        })
    }

    /// True when every node is layout metadata. Standalone layout-only graphs must not be
    /// claimed for NPU compile; the same nodes may still appear as glue beside compute.
    pub fn is_layout_only(&self) -> bool {
        !self.nodes.is_empty() && self.nodes.iter().all(Self::node_is_layout_metadata)
    }

    /// True when at least one node is real compute (not layout metadata).
    pub fn has_lowerable_compute(&self) -> bool {
        self.nodes.iter().any(|node| !Self::node_is_layout_metadata(node))
    }

    /// Plan the largest semantically connected regions before vendor feasibility is considered.
    /// There is deliberately no raw node-count limit. Boundaries exist only for disconnected work,
    /// independent mutable-state effects, and multiple routed expert banks. A caller may further
    /// bisect a region when its concrete compiler/runtime rejects it.
    pub fn execution_regions(&self, policy: PartitionPolicy) -> Vec<ExecutionPartition> {
        if self.nodes.is_empty() || policy.max_expert_banks == 0 {
            return Vec::new();
        }
        let mut ranges: Vec<(usize, usize, PartitionReason)> = Vec::new();
        let mut start = 0usize;
        let mut touched = vec![false; self.values.len()];
        let mut routed_banks = 0usize;
        let mut mutable_effect_seen = false;
        let mut next_reason = PartitionReason::WholeGraph;
        for (index, node) in self.nodes.iter().enumerate() {
            let spec = node.spec.ok();
            let routed = Self::wire_op(node) == Some(WireOp::MatMulId);
            let mutable_effect = spec.is_some_and(|spec| {
                spec.traits.stateful
                    || node.inputs.iter().any(|&id| {
                        self.values
                            .get(id as usize)
                            .is_some_and(|value| value.role == TensorRole::MutableState)
                    })
                    || self
                        .values
                        .get(node.output as usize)
                        .is_some_and(|value| value.role == TensorRole::MutableState)
            });
            let connected = index == start
                || node
                    .inputs
                    .iter()
                    .any(|&id| touched.get(id as usize).copied().unwrap_or(false));
            let boundary = if index > start && routed && routed_banks >= policy.max_expert_banks {
                Some(PartitionReason::ExpertBankLimit)
            } else if index > start && mutable_effect && mutable_effect_seen && !connected {
                Some(PartitionReason::MutableStateBoundary)
            } else if index > start && !connected {
                Some(PartitionReason::DisconnectedRegion)
            } else {
                None
            };
            if let Some(reason) = boundary {
                ranges.push((start, index, next_reason));
                start = index;
                touched.fill(false);
                routed_banks = 0;
                mutable_effect_seen = false;
                next_reason = reason;
            }
            for &input in &node.inputs {
                if let Some(value) = touched.get_mut(input as usize) {
                    *value = true;
                }
            }
            if let Some(value) = touched.get_mut(node.output as usize) {
                *value = true;
            }
            routed_banks += usize::from(routed);
            mutable_effect_seen |= mutable_effect;
        }
        ranges.push((start, self.nodes.len(), next_reason));
        ranges
            .into_iter()
            .filter_map(|(start, end, reason)| {
                self.partition_range(start, end)
                    .map(|graph| ExecutionPartition {
                        graph,
                        reason,
                        start,
                    })
            })
            .collect()
    }

    pub(crate) fn partition_range(&self, start: usize, end: usize) -> Option<GraphIr> {
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

    /// Exact in-process program identity. BLAKE3 is only a lookup prefilter; cache equality also
    /// verifies the owned length-framed semantic bytes.
    pub fn semantic_key(&self) -> crate::identity::ExactSemanticKey {
        let mut key = crate::identity::ExactKeyBuilder::new(b"graph-ir-v2/program");
        key.u64(b"generation", self.generation)
            .u64(b"value-count", self.values.len() as u64)
            .u64(b"node-count", self.nodes.len() as u64)
            .u64(b"input-count", self.inputs.len() as u64)
            .u64(b"output-count", self.outputs.len() as u64);
        for (index, value) in self.values.iter().enumerate() {
            key.u64(b"value-index", index as u64)
                .u64(b"value-role", value.role as u64)
                .u64(b"value-dtype", value.dtype as u64)
                .u64(b"value-owned", u64::from(value.owned));
            for dim in value.dims {
                match dim {
                    Dim::Static(value) => {
                        key.u64(b"dim-kind", 0).i64(b"dim-static", value);
                    }
                    Dim::Bounded { min, max } => {
                        key.u64(b"dim-kind", 1)
                            .i64(b"dim-min", min)
                            .i64(b"dim-max", max);
                    }
                }
            }
            for stride in value.strides {
                key.u64(b"stride", stride as u64);
            }
            if value.role == TensorRole::Constant {
                key.u64(b"constant-address", value.data as u64);
            }
        }
        for (index, node) in self.nodes.iter().enumerate() {
            key.u64(b"node-index", index as u64)
                .u64(b"node-op", node.op as u64)
                .u64(
                    b"node-wire-op",
                    Self::wire_op(node).map_or(u64::MAX, |wire| wire as u64),
                )
                .u64(b"node-input-count", node.inputs.len() as u64);
            for &input in &node.inputs {
                key.u64(b"node-input", input as u64);
            }
            key.u64(b"node-output", node.output as u64)
                .field(b"node-op-params", &node.op_params);
        }
        for &input in &self.inputs {
            key.u64(b"graph-input", input as u64);
        }
        for &output in &self.outputs {
            key.u64(b"graph-output", output as u64);
        }
        key.finish()
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
            shared: None,
        }
    }

    fn add_node(inputs: Vec<ValueId>, output: ValueId) -> Node {
        Node {
            op: GGML_OP_ADD,
            spec: decode_spec(GGML_OP_ADD, 0, 0.0, 0, 2),
            inputs,
            output,
            op_params: [0; OP_PARAM_BYTES],
        }
    }

    #[test]
    fn partition_policy_fuses_connected_graphs_larger_than_eight_nodes() {
        let mut values = vec![
            value(TensorRole::RuntimeInput, 1, Dim::Static(4)),
            value(TensorRole::RuntimeInput, 2, Dim::Static(4)),
        ];
        let mut nodes = Vec::new();
        let mut previous = 0;
        for index in 0..12 {
            let output = values.len() as ValueId;
            values.push(value(
                if index == 11 {
                    TensorRole::Output
                } else {
                    TensorRole::Intermediate
                },
                10 + index,
                Dim::Static(4),
            ));
            nodes.push(add_node(vec![previous, 1], output));
            previous = output;
        }
        let graph = GraphIr {
            generation: 1,
            values,
            nodes,
            inputs: vec![0, 1],
            outputs: vec![previous],
        };
        let regions = graph.execution_regions(PartitionPolicy::default());
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].graph.nodes.len(), 12);
        assert_eq!(regions[0].reason, PartitionReason::WholeGraph);
    }

    #[test]
    fn partition_policy_limits_only_routed_expert_banks() {
        let values = vec![
            value(TensorRole::Constant, 1, Dim::Static(4)),
            value(TensorRole::RuntimeInput, 2, Dim::Static(4)),
            value(TensorRole::RuntimeInput, 3, Dim::Static(1)),
            value(TensorRole::Intermediate, 4, Dim::Static(4)),
            value(TensorRole::Constant, 5, Dim::Static(4)),
            value(TensorRole::Output, 6, Dim::Static(4)),
        ];
        let routed = |inputs, output| Node {
            op: GGML_OP_MUL_MAT_ID,
            spec: decode_spec(GGML_OP_MUL_MAT_ID, 0, 0.0, 0, 3),
            inputs,
            output,
            op_params: [0; OP_PARAM_BYTES],
        };
        let graph = GraphIr {
            generation: 1,
            values,
            nodes: vec![routed(vec![0, 1, 2], 3), routed(vec![4, 3, 2], 5)],
            inputs: vec![0, 1, 2, 4],
            outputs: vec![5],
        };
        let regions = graph.execution_regions(PartitionPolicy::default());
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].graph.nodes.len(), 1);
        assert_eq!(regions[1].graph.nodes.len(), 1);
        assert_eq!(regions[1].reason, PartitionReason::ExpertBankLimit);
        assert_eq!(regions[0].graph.outputs.len(), 1);
        assert_eq!(regions[1].graph.inputs.len(), 2);
    }

    #[test]
    fn partition_range_preserves_crossing_state_and_output_order() {
        let mut state = value(TensorRole::MutableState, 1, Dim::Static(4));
        state.owned = true;
        let values = vec![
            state,
            value(TensorRole::RuntimeInput, 2, Dim::Static(4)),
            value(TensorRole::MutableState, 3, Dim::Static(4)),
            value(TensorRole::Output, 4, Dim::Static(4)),
        ];
        let graph = GraphIr {
            generation: 1,
            values,
            nodes: vec![add_node(vec![0, 1], 2), add_node(vec![2, 1], 3)],
            inputs: vec![0, 1],
            outputs: vec![2, 3],
        };
        let first = graph.partition_range(0, 1).unwrap();
        assert_eq!(first.outputs.len(), 1);
        assert_eq!(
            first.values[first.outputs[0] as usize].role,
            TensorRole::MutableState
        );
        let second = graph.partition_range(1, 2).unwrap();
        assert!(second
            .inputs
            .iter()
            .any(|&input| { second.values[input as usize].role == TensorRole::MutableState }));
        assert_eq!(second.outputs.len(), 1);
    }

    #[test]
    fn semantic_key_tracks_structure_shape_and_constant_identity() {
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
            make(11, Dim::Static(4)).semantic_key(),
            make(11, Dim::Static(4)).semantic_key()
        );
        assert_ne!(
            make(11, Dim::Static(4)).semantic_key(),
            make(12, Dim::Static(4)).semantic_key()
        );
        assert_ne!(
            make(11, Dim::Static(4)).semantic_key(),
            make(11, Dim::Bounded { min: 1, max: 4 }).semantic_key()
        );
        let mut alternate_output = make(11, Dim::Static(4));
        alternate_output.outputs = vec![1];
        assert_ne!(
            make(11, Dim::Static(4)).semantic_key(),
            alternate_output.semantic_key()
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
                .map(|spec| (spec.wire, spec.traits.stateful, spec.traits.attention)),
            Ok((WireOp::SetRows, true, false))
        );
        // LFM uses NEOX (mode 2); normal (mode 0) remains admitted. Other modes stay rejected.
        assert!(decode_spec(GGML_OP_ROPE, 0, 0.0, 0, 2).is_ok());
        assert!(decode_spec(GGML_OP_ROPE, 0, 0.0, 2, 2).is_ok());
        assert_eq!(
            decode_spec(GGML_OP_ROPE, 0, 0.0, 8, 2),
            Err(OpReject::UnsupportedVariant)
        );
    }

    #[test]
    fn softmax_and_flash_attention_supported_modes() {
        // SoftMax: max_bias==0, arity 1 (logits) or 2 (logits+mask). Scale lives in op_params[0].
        assert_eq!(
            decode_spec(GGML_OP_SOFT_MAX, 0, 0.0, 0, 1).map(|s| s.placement),
            Ok(PlacementClass::FusibleOnly)
        );
        assert_eq!(
            decode_spec(GGML_OP_SOFT_MAX, 0, 0.0, 0, 2).map(|s| (s.wire, s.traits.attention)),
            Ok((WireOp::SoftMax, true))
        );
        assert_eq!(
            decode_spec(GGML_OP_SOFT_MAX, 0, 1.0, 0, 1),
            Err(OpReject::UnsupportedVariant)
        );
        assert_eq!(
            decode_spec(GGML_OP_SOFT_MAX, 0, 0.0, 0, 3),
            Err(OpReject::InvalidArity)
        );
        // FlashAttention: max_bias==0 and logit_softcap==0 (decode_node), arity 3–4, Anchor.
        assert_eq!(
            decode_spec(GGML_OP_FLASH_ATTN_EXT, 0, 0.0, 0, 3).map(|s| s.placement),
            Ok(PlacementClass::Anchor)
        );
        assert_eq!(
            decode_spec(GGML_OP_FLASH_ATTN_EXT, 0, 0.0, 0, 4).map(|s| s.wire),
            Ok(WireOp::FlashAttention)
        );
        let mut softcap = [0u8; OP_PARAM_BYTES];
        softcap[8..12].copy_from_slice(&1.0f32.to_ne_bytes());
        assert_eq!(
            OpRegistry::decode_node(&Node {
                op: GGML_OP_FLASH_ATTN_EXT,
                spec: Err(OpReject::UnsupportedOp),
                inputs: alloc::vec![0, 1, 2],
                output: 3,
                op_params: softcap,
            }),
            Err(OpReject::UnsupportedVariant)
        );
        let mut bias = [0u8; OP_PARAM_BYTES];
        bias[4..8].copy_from_slice(&1.0f32.to_ne_bytes());
        assert_eq!(
            OpRegistry::decode_node(&Node {
                op: GGML_OP_FLASH_ATTN_EXT,
                spec: Err(OpReject::UnsupportedOp),
                inputs: alloc::vec![0, 1, 2],
                output: 3,
                op_params: bias,
            }),
            Err(OpReject::UnsupportedVariant)
        );
    }

    #[test]
    fn prefill_qkv_rope_and_layout_are_not_attention_gated() {
        // RoPE + Permute must not require GGML_NPU_ATTENTION=full; SoftMax/FA still do (#10).
        assert_eq!(
            decode_spec(GGML_OP_ROPE, 0, 0.0, 2, 2).map(|s| (s.wire, s.traits.attention)),
            Ok((WireOp::Rope, false))
        );
        assert_eq!(
            decode_spec(GGML_OP_PERMUTE, 0, 0.0, 0, 1).map(|s| (s.wire, s.traits.attention)),
            Ok((WireOp::Permute, false))
        );
        assert_eq!(
            decode_spec(GGML_OP_SOFT_MAX, 0, 0.0, 0, 1).map(|s| (s.wire, s.traits.attention)),
            Ok((WireOp::SoftMax, true))
        );
        assert_eq!(
            decode_spec(GGML_OP_FLASH_ATTN_EXT, 0, 0.0, 0, 3)
                .map(|s| (s.wire, s.traits.attention)),
            Ok((WireOp::FlashAttention, true))
        );
    }

    #[test]
    fn moe_routing_ops_are_moe_tagged_not_attention() {
        // Sparse MoE ops are MoE-gated (not ATTENTION=full). Argsort/MatMulId stay MoE-tagged but
        // supports_op-unclaimed even under MOE=1 until model-scale ids/experts are proven (#11).
        for (op, unary, n_inputs, wire) in [
            (GGML_OP_ARGSORT, 0, 1usize, WireOp::Argsort),
            (GGML_OP_SUM_ROWS, 0, 1, WireOp::SumRows),
            (GGML_OP_MUL_MAT_ID, 0, 3, WireOp::MatMulId),
            (
                GGML_OP_UNARY,
                GGML_UNARY_OP_SIGMOID as i32,
                1,
                WireOp::Sigmoid,
            ),
        ] {
            let spec = decode_spec(op, unary, 0.0, 0, n_inputs).expect("MoE op must decode");
            assert_eq!(spec.wire, wire);
            assert!(spec.traits.moe);
            assert!(!spec.traits.attention);
        }
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

    #[test]
    fn connected_shortconv_gather_conv_write_stays_one_region() {
        // Connected gather state → SSM_CONV → SetRows writeback.
        // Connected mutable effects must not open a MutableStateBoundary (#9).
        // Intel may still refuse fusing SSM_CONV+SET_ROWS into one OV program; the engine then
        // peels FeasibilityLimit adjacent NPU programs (see engine tests) — never host-realize
        // these stateful edges as layout_glue_host.
        let mut state_in = value(TensorRole::MutableState, 1, Dim::Static(4));
        state_in.owned = true;
        let values = vec![
            state_in,                                                 // 0 base state
            value(TensorRole::RuntimeInput, 2, Dim::Static(2)),       // 1 gather indices
            value(TensorRole::Intermediate, 3, Dim::Static(4)),       // 2 gathered rows
            value(TensorRole::Constant, 4, Dim::Static(3)),           // 3 conv kernel
            value(TensorRole::Intermediate, 5, Dim::Static(2)),       // 4 conv output
            value(TensorRole::RuntimeInput, 6, Dim::Static(2)),       // 5 write indices
            value(TensorRole::MutableState, 7, Dim::Static(4)),       // 6 written state
        ];
        let get_rows = Node {
            op: GGML_OP_GET_ROWS,
            spec: decode_spec(GGML_OP_GET_ROWS, 0, 0.0, 0, 2),
            inputs: vec![0, 1],
            output: 2,
            op_params: [0; OP_PARAM_BYTES],
        };
        let ssm = Node {
            op: GGML_OP_SSM_CONV,
            spec: decode_spec(GGML_OP_SSM_CONV, 0, 0.0, 0, 2),
            inputs: vec![2, 3],
            output: 4,
            op_params: [0; OP_PARAM_BYTES],
        };
        let set_rows = Node {
            op: GGML_OP_SET_ROWS,
            spec: decode_spec(GGML_OP_SET_ROWS, 0, 0.0, 0, 3),
            inputs: vec![4, 5, 0],
            output: 6,
            op_params: [0; OP_PARAM_BYTES],
        };
        let graph = GraphIr {
            generation: 1,
            values,
            nodes: vec![get_rows, ssm, set_rows],
            inputs: vec![0, 1, 3, 5],
            outputs: vec![6],
        };
        let regions = graph.execution_regions(PartitionPolicy::default());
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].graph.nodes.len(), 3);
        assert_eq!(regions[0].reason, PartitionReason::WholeGraph);
    }

    #[test]
    fn disconnected_mutable_state_effects_still_partition() {
        let mut state_a = value(TensorRole::MutableState, 1, Dim::Static(4));
        state_a.owned = true;
        let mut state_b = value(TensorRole::MutableState, 2, Dim::Static(4));
        state_b.owned = true;
        let values = vec![
            state_a,                                                  // 0
            value(TensorRole::RuntimeInput, 3, Dim::Static(2)),       // 1 updates a
            value(TensorRole::RuntimeInput, 4, Dim::Static(1)),       // 2 indices a
            value(TensorRole::MutableState, 5, Dim::Static(4)),       // 3 written a
            state_b,                                                  // 4
            value(TensorRole::RuntimeInput, 6, Dim::Static(2)),       // 5 updates b
            value(TensorRole::RuntimeInput, 7, Dim::Static(1)),       // 6 indices b
            value(TensorRole::MutableState, 8, Dim::Static(4)),       // 7 written b
        ];
        let first = Node {
            op: GGML_OP_SET_ROWS,
            spec: decode_spec(GGML_OP_SET_ROWS, 0, 0.0, 0, 3),
            inputs: vec![1, 2, 0],
            output: 3,
            op_params: [0; OP_PARAM_BYTES],
        };
        let second = Node {
            op: GGML_OP_SET_ROWS,
            spec: decode_spec(GGML_OP_SET_ROWS, 0, 0.0, 0, 3),
            inputs: vec![5, 6, 4],
            output: 7,
            op_params: [0; OP_PARAM_BYTES],
        };
        let graph = GraphIr {
            generation: 1,
            values,
            nodes: vec![first, second],
            inputs: vec![0, 1, 2, 4, 5, 6],
            outputs: vec![3, 7],
        };
        let regions = graph.execution_regions(PartitionPolicy::default());
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[1].reason, PartitionReason::MutableStateBoundary);
    }
}
