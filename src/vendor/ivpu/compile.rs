//! Constant-blob compilation for the Intel NPU: shape + weights → a native NPU program blob.
//!
//! The OpenVINO compile shim (`libggml_npu_ovc.so`) is `dlopen`'d on demand and, given a weight
//! re-quantized to low-bit form ([`crate::ops::requant`]), bakes it as a compile-time `Constant` and
//! compiles a native NPU blob (the fused decompress-matmul). Each blob is a **transient** artifact:
//! the caller ([`super::session`]) parses it into a driver-owned graph once and then drops it,
//! caching *that graph* — so a program's weights are not also kept resident as blob bytes (the
//! double-keep this avoids). Cold-compile cost is amortized two ways:
//! - **on disk** by a content hash — managed inside the shim (C++) so this core stays `no_std` and
//!   file-I/O-free; a warm process reloads a ~few-ms blob instead of recompiling (~100 ms).
//! - **in memory** by the built graph (in [`super::session`], keyed by `(weight pointer, M)` for a
//!   single matmul or the exact [`sg_semantic_key`] for a subgraph) — so a program compiles at
//!   most once per process and steady-state decode is a pure graph-cache hit.
//!
//! `no_std`: the shim, `dlopen`, and `getenv` resolve from the host. A compile failure is cached
//! (as `None`) and not retried.

use alloc::boxed::Box;
use alloc::ffi::CString;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_void, CStr};
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::graph_ir::{Dim, GraphIr, TensorRole, OP_PARAM_BYTES};
use crate::identity::{SemanticDigest, SemanticHasher};
use crate::ops::requant::{self, Requant};
use crate::ops::subgraph::{Ref, SgOp, Subgraph};
use crate::ops::Operand;

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn getenv(name: *const c_char) -> *const c_char;
}
const RTLD_NOW: c_int = 2;

/// Persistent artifact namespace. Increment whenever compiler semantics or framing changes.
const FORMAT_VERSION: u64 = 7;

fn persistent_hasher(domain: &[u8]) -> SemanticHasher {
    let mut hasher = SemanticHasher::new(domain);
    hasher
        .u64(b"format-version", FORMAT_VERSION)
        .field(b"target", b"intel-npu")
        .field(b"compiler-adapter", b"openvino-native-graph")
        .u64(b"asymmetric-requant", u64::from(asymmetric()));
    hasher
}

fn persistent_key(prefix: &str, hasher: SemanticHasher, suffix: &str) -> CString {
    let digest = hasher.finish().hex();
    CString::new(alloc::format!(
        "{prefix}-b3v{FORMAT_VERSION}-{digest}-{suffix}"
    ))
    .unwrap_or_else(|_| CString::new("invalid-blake3-cache-key").unwrap())
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ConstantDigestKey {
    generation: u64,
    source_address: usize,
    source_dtype: u32,
    source_elements: usize,
    k: i64,
    n: i64,
    bits: i32,
    group: i32,
    asymmetric: bool,
}

static CONSTANT_DIGESTS: super::cache::Cache<ConstantDigestKey, SemanticDigest> =
    super::cache::Cache::new(2048);

fn compute_requant_digest(rq: &Requant) -> SemanticDigest {
    let mut hasher = SemanticHasher::new(b"requantized-constant/content");
    hasher
        .u64(b"bits", rq.bits as u64)
        .u64(b"group", rq.group as u64)
        .field(b"weight", &rq.weight)
        .field(b"scales", &rq.scales)
        .field(b"zeros", &rq.zeros);
    hasher.finish()
}

fn requant_digest(key: ConstantDigestKey, rq: &Requant) -> SemanticDigest {
    CONSTANT_DIGESTS
        .try_get_or_insert_with(key, || compute_requant_digest(rq))
        .unwrap_or_else(|| compute_requant_digest(rq))
}

// int ggml_npu_ov_compile_matmul_const(M, K, N, bits, group, weight, weight_len,
//                                      scales, scales_len, zeros, zeros_len, cache_key, &blob, &len)
type CompileConst = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    i32,
    *const u8,
    usize,
    *const u8,
    usize,
    *const u8,
    usize,
    *const c_char,
    *mut *mut u8,
    *mut usize,
) -> c_int;
type FreeBlob = unsafe extern "C" fn(*mut u8);

// int ggml_npu_ov_compile_subgraph(M, n_inputs, input_k, n_nodes, nodes, n_consts, consts,
//                                  out_nodes, n_outs, cache_key, &blob, &len)
type CompileSubgraph = unsafe extern "C" fn(
    i32,
    i32,
    *const i32,
    i32,
    *const SgNodeAbi,
    i32,
    *const SgConstAbi,
    *const i32,
    i32,
    *const c_char,
    *mut *mut u8,
    *mut usize,
) -> c_int;
type CompileGraph = unsafe extern "C" fn(
    i32,
    *const IrTensorAbi,
    i32,
    *const IrNodeAbi,
    i32,
    *const u32,
    i32,
    *const u32,
    *const c_char,
    *mut *mut u8,
    *mut usize,
) -> c_int;

/// Flat op-DAG node matching the C `npu_sg_node`. `a`/`b` are value refs (`< n_inputs` = input,
/// else node `ref - n_inputs`; `-1` = unused); `const_idx` = baked weight or `-1`.
#[repr(C)]
struct SgNodeAbi {
    op: i32,
    a: i32,
    b: i32,
    const_idx: i32,
    out_n: i32,
}

/// Flat baked constant matching the C `npu_sg_const`.
#[repr(C)]
struct SgConstAbi {
    bits: i32,
    group: i32,
    n: i32,
    k: i32,
    weight: *const u8,
    weight_len: usize,
    scales: *const u8,
    scales_len: usize,
    zeros: *const u8,
    zeros_len: usize,
}

const IR_MAX_INPUTS: usize = 16;

#[repr(C)]
struct IrTensorAbi {
    role: i32,
    dtype: i32,
    ne: [i64; 4],
    bits: i32,
    group: i32,
    weight: *const u8,
    weight_len: usize,
    scales: *const u8,
    scales_len: usize,
    zeros: *const u8,
    zeros_len: usize,
}

#[repr(C)]
struct IrNodeAbi {
    op: i32,
    n_inputs: i32,
    inputs: [i32; IR_MAX_INPUTS],
    output: i32,
    params: [u8; OP_PARAM_BYTES],
}

struct Shim {
    compile_const: CompileConst,
    compile_subgraph: Option<CompileSubgraph>,
    compile_graph: Option<CompileGraph>,
    free: FreeBlob,
}

/// `null` = untried, `FAILED` = tried & unavailable, else a leaked `*mut Shim`.
static SHIM: AtomicPtr<Shim> = AtomicPtr::new(ptr::null_mut());
const SHIM_FAILED: *mut Shim = 1usize as *mut Shim;

unsafe fn load_shim() -> Option<Shim> {
    let env = getenv(c"GGML_NPU_OVC".as_ptr());
    let h = if env.is_null() {
        dlopen(c"libggml_npu_ovc.so".as_ptr(), RTLD_NOW)
    } else {
        dlopen(env, RTLD_NOW)
    };
    if h.is_null() {
        let error = dlerror();
        if !error.is_null() {
            super::profile::note(&alloc::format!(
                "[npu ir2] shim dlopen failed: {}\n",
                CStr::from_ptr(error).to_string_lossy()
            ));
        }
        return None;
    }
    let compile_const = dlsym(h, c"ggml_npu_ov_compile_matmul_const".as_ptr());
    let free = dlsym(h, c"ggml_npu_ov_free".as_ptr());
    if compile_const.is_null() || free.is_null() {
        return None;
    }
    // Optional: an older shim without subgraph support still serves the single-matmul path.
    let sg = dlsym(h, c"ggml_npu_ov_compile_subgraph".as_ptr());
    let compile_subgraph = if sg.is_null() {
        None
    } else {
        Some(mem::transmute::<*mut c_void, CompileSubgraph>(sg))
    };
    let ir = dlsym(h, c"ggml_npu_ov_compile_graph".as_ptr());
    let compile_graph = if ir.is_null() {
        None
    } else {
        Some(mem::transmute::<*mut c_void, CompileGraph>(ir))
    };
    Some(Shim {
        compile_const: mem::transmute::<*mut c_void, CompileConst>(compile_const),
        compile_subgraph,
        compile_graph,
        free: mem::transmute::<*mut c_void, FreeBlob>(free),
    })
}

fn shim() -> Option<&'static Shim> {
    let cur = SHIM.load(Ordering::Acquire);
    if cur == SHIM_FAILED {
        return None;
    }
    if !cur.is_null() {
        return Some(unsafe { &*cur });
    }
    match unsafe { load_shim() } {
        Some(s) => {
            let boxed = Box::into_raw(Box::new(s));
            match SHIM.compare_exchange(ptr::null_mut(), boxed, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => Some(unsafe { &*boxed }),
                Err(w) => {
                    drop(unsafe { Box::from_raw(boxed) });
                    if w == SHIM_FAILED {
                        None
                    } else {
                        Some(unsafe { &*w })
                    }
                }
            }
        }
        None => {
            let _ = SHIM.compare_exchange(
                ptr::null_mut(),
                SHIM_FAILED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            None
        }
    }
}

/// Cheap feasibility check (no compile): is the shim present and the weight re-quantizable at this
/// shape? A `true` means [`compile_const_blob`] will (barring a rare OpenVINO failure) succeed, so
/// `can_lower` can claim the matmul and its blob compiles **lazily in `run`** — avoiding a wasted
/// standalone compile for a matmul that then fuses into an FFN subgraph (the common case). Same
/// contract as the fused-subgraph path, which also compiles lazily at `graph_compute`.
pub(super) fn const_feasible(weight: &Operand, m: i64, k: i64, n: i64) -> bool {
    m > 0
        && k > 0
        && n > 0
        && m <= i32::MAX as i64
        && k <= i32::MAX as i64
        && n <= i32::MAX as i64
        && shim().is_some()
        && requant::can_requant(weight, k as usize)
}

/// Compile the Constant blob for `weight` at shape `(M, K, N)` — re-quantize + compile (or
/// disk-load) — as a **transient** `Vec<u8>` the caller parses into a graph and then drops. `None`
/// if the shim/NPU is absent or the weight can't take this path. Called once per `(weight, M)` on a
/// [`session::run_const`](super::session::run_const) graph-cache miss; the built graph is what stays
/// resident, so the blob is never itself cached (no double-keep of every program's weights).
pub(super) fn compile_const_blob(weight: &Operand, m: i64, k: i64, n: i64) -> Option<Vec<u8>> {
    if m <= 0
        || k <= 0
        || n <= 0
        || m > i32::MAX as i64
        || k > i32::MAX as i64
        || n > i32::MAX as i64
    {
        return None;
    }
    compile_const(weight, m, k, n)
}

/// Optional weight-width override from `GGML_NPU_BITS` (`"4"`, `"8"`, or `"16"`); unset/other →
/// auto per source type. Reads the first two bytes of the C string (enough to distinguish).
fn forced_bits() -> Option<i32> {
    let p = unsafe { getenv(c"GGML_NPU_BITS".as_ptr()) };
    if p.is_null() {
        return None;
    }
    let c0 = unsafe { *p };
    let c1 = if c0 == 0 { 0 } else { unsafe { *p.add(1) } };
    match (c0 as u8, c1 as u8) {
        (b'4', _) => Some(4),
        (b'8', _) => Some(8),
        (b'1', b'6') => Some(16),
        _ => None,
    }
}

/// Vendor policy: pick the re-quant width for `weight`. `GGML_NPU_BITS` overrides; else int4 for a
/// ≤5-bpw quantized source (Q4_*), int8 for ≥5-bpw (Q6_K/Q8_0), f16 for an f32 source. The NPU keeps
/// low-bit weights compressed (and fuses the dequant) only via the asymmetric-unsigned canonical
/// form, which [`compile_const`]/[`compile_subgraph`] always request (`asymmetric = true`).
fn choose_bits(weight: &Operand, force: Option<i32>) -> i32 {
    if let Some(b) = force {
        return b;
    }
    let traits = unsafe { crate::ffi::ggml_get_type_traits(weight.dtype) };
    if traits.is_null() {
        return 8;
    }
    let tr = unsafe { &*traits };
    if !tr.is_quantized {
        return 16;
    }
    let bpw = if tr.blck_size > 0 {
        (tr.type_size as f32) * 8.0 / (tr.blck_size as f32)
    } else {
        8.0
    };
    if bpw <= 5.0 {
        4
    } else {
        8
    }
}

/// Asymmetric (canonical, kept-compressed) unless `GGML_NPU_SYM` is set (a debug toggle).
fn asymmetric() -> bool {
    unsafe { getenv(c"GGML_NPU_SYM".as_ptr()).is_null() }
}

fn compile_const(weight: &Operand, m: i64, k: i64, n: i64) -> Option<Vec<u8>> {
    let s = shim()?;
    let rq = requant::requant(
        weight,
        k as usize,
        n as usize,
        choose_bits(weight, forced_bits()),
        asymmetric(),
    )?;
    let key = cache_key(&rq, weight.ptr as usize, weight.dtype as u32, m, k, n);

    let mut blob: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    let rc = unsafe {
        (s.compile_const)(
            m as i32,
            k as i32,
            n as i32,
            rq.bits,
            rq.group,
            rq.weight.as_ptr(),
            rq.weight.len(),
            rq.scales.as_ptr(),
            rq.scales.len(),
            rq.zeros.as_ptr(),
            rq.zeros.len(),
            key.as_ptr(),
            &mut blob,
            &mut len,
        )
    };
    if rc != 0 || blob.is_null() || len == 0 {
        return None;
    }
    let out = unsafe { core::slice::from_raw_parts(blob, len) }.to_vec();
    unsafe { (s.free)(blob) };
    Some(out)
}

// ---------------------------------------------------------------------------
// Graph IR v2: one ABI for complete scheduler splits. The first Intel lowering supports the
// dense/ShortConv building blocks; capability remains conservative as more WireOps are added.
// ---------------------------------------------------------------------------

pub(super) fn graph_feasible(graph: &GraphIr) -> bool {
    let Some(s) = shim() else {
        super::profile::note("[npu ir2] unavailable: OpenVINO compiler shim not loaded\n");
        return false;
    };
    if s.compile_graph.is_none() {
        super::profile::note("[npu ir2] unavailable: shim lacks graph compiler symbol\n");
        return false;
    }
    if graph.nodes.is_empty()
        || graph.outputs.is_empty()
        || graph.inputs.len() > 64
        || graph.outputs.len() > 8
    {
        super::profile::note("[npu ir2] rejected graph size or binding limits\n");
        return false;
    }
    for (index, value) in graph.values.iter().enumerate() {
        if value.role != TensorRole::Constant
            && !matches!(
                value.dtype,
                crate::ffi::GGML_TYPE_F32
                    | crate::ffi::GGML_TYPE_F16
                    | crate::ffi::GGML_TYPE_I32
                    | crate::ffi::GGML_TYPE_I64
            )
        {
            super::profile::note(&alloc::format!(
                "[npu ir2] value {index} rejected: unsupported runtime dtype {}\n",
                value.dtype
            ));
            return false;
        }
    }
    for (index, node) in graph.nodes.iter().enumerate() {
        let Ok(spec) = node.spec else {
            super::profile::note(&alloc::format!(
                "[npu ir2] node {index} rejected by operation contract: {:?}\n",
                node.spec
            ));
            return false;
        };
        let Some(op) = GraphIr::wire_op(node) else {
            super::profile::note(&alloc::format!(
                "[npu ir2] node {index} rejected: unmapped ggml op {}\n",
                node.op
            ));
            return false;
        };
        if spec.traits.moe && !moe_enabled() {
            super::profile::note(&alloc::format!(
                "[npu ir2] node {index} rejected: MoE op {op:?} requires GGML_NPU_MOE=1\n"
            ));
            return false;
        }
        let runtime_matmul = op == crate::graph_ir::WireOp::MatMul
            && node
                .inputs
                .first()
                .and_then(|&id| graph.values.get(id as usize))
                .is_some_and(|value| value.role != TensorRole::Constant);
        if (spec.traits.attention || runtime_matmul) && !attention_full() {
            super::profile::note(&alloc::format!(
                "[npu ir2] node {index} rejected: attention op {op:?} requires GGML_NPU_ATTENTION=full\n"
            ));
            return false;
        }
        let supported = matches!(
            op,
            crate::graph_ir::WireOp::MatMul
                | crate::graph_ir::WireOp::Mul
                | crate::graph_ir::WireOp::Add
                | crate::graph_ir::WireOp::RmsNorm
                | crate::graph_ir::WireOp::Silu
                | crate::graph_ir::WireOp::Reshape
                | crate::graph_ir::WireOp::View
                | crate::graph_ir::WireOp::Cont
                | crate::graph_ir::WireOp::Transpose
                | crate::graph_ir::WireOp::Concat
                | crate::graph_ir::WireOp::Copy
                | crate::graph_ir::WireOp::SsmConv
                | crate::graph_ir::WireOp::Rope
                | crate::graph_ir::WireOp::GetRows
                | crate::graph_ir::WireOp::SetRows
                | crate::graph_ir::WireOp::Scale
                | crate::graph_ir::WireOp::Div
                | crate::graph_ir::WireOp::Clamp
                | crate::graph_ir::WireOp::Argsort
                | crate::graph_ir::WireOp::MatMulId
                | crate::graph_ir::WireOp::Sigmoid
                | crate::graph_ir::WireOp::SumRows
                | crate::graph_ir::WireOp::SoftMax
                | crate::graph_ir::WireOp::FlashAttention
                | crate::graph_ir::WireOp::Permute
                | crate::graph_ir::WireOp::Swiglu
        );
        if !supported || node.inputs.len() > IR_MAX_INPUTS {
            super::profile::note(&alloc::format!(
                "[npu ir2] node {index} rejected: wire op {op:?}, inputs={}\n",
                node.inputs.len()
            ));
            return false;
        }
    }
    true
}

pub(super) unsafe fn graph_node_feasible(
    node: *const crate::ffi::ggml_tensor,
    spec: &crate::graph_ir::OpSpec,
) -> bool {
    let op = Some(spec.wire);
    let moe_only = matches!(
        op,
        Some(
            crate::graph_ir::WireOp::MatMulId
                | crate::graph_ir::WireOp::Argsort
                | crate::graph_ir::WireOp::Sigmoid
                | crate::graph_ir::WireOp::SumRows
        )
    );
    let attention_only = matches!(
        op,
        Some(crate::graph_ir::WireOp::SoftMax | crate::graph_ir::WireOp::Permute)
    ) || (op == Some(crate::graph_ir::WireOp::MatMul)
        && !node.is_null()
        && !(*node).src[0].is_null()
        && (*(*node).src[0]).op != crate::ffi::GGML_OP_NONE);
    shim().and_then(|s| s.compile_graph).is_some()
        && op.is_some()
        && (!moe_only || moe_enabled())
        && (!attention_only || attention_full())
}

fn moe_enabled() -> bool {
    !unsafe { getenv(c"GGML_NPU_MOE".as_ptr()) }.is_null()
}

fn attention_full() -> bool {
    let p = unsafe { getenv(c"GGML_NPU_ATTENTION".as_ptr()) };
    !p.is_null() && unsafe { CStr::from_ptr(p) }.to_bytes() == b"full"
}

pub(super) fn compile_graph_blob(graph: &GraphIr) -> Option<Vec<u8>> {
    if !graph_feasible(graph) {
        return None;
    }
    let s = shim()?;
    let compile = s.compile_graph?;

    // Materialize constants before building pointer-bearing ABI records so Vec relocation cannot
    // invalidate pointers during construction.
    let mut requants: Vec<Requant> = Vec::new();
    let mut req_index: Vec<Option<usize>> = vec![None; graph.values.len()];
    for (id, t) in graph.values.iter().enumerate() {
        if t.role != TensorRole::Constant {
            continue;
        }
        let ne = t.dims.map(|d| match d {
            Dim::Static(x) => x,
            Dim::Bounded { max, .. } => max,
        });
        if ne.iter().any(|&x| x <= 0) {
            return None;
        }
        let k = ne[0] as usize;
        let elems = ne
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b as usize))?;
        let n = elems.checked_div(k)?;
        let operand = Operand {
            ptr: t.data as *mut c_void,
            elems,
            dtype: t.dtype,
            owned: t.owned,
        };
        // Tiny convolution kernels are not group-quantizable; preserve them as f16.
        let bits = if k % 32 == 0 {
            choose_bits(&operand, forced_bits())
        } else {
            16
        };
        let rq = requant::requant(&operand, k, n, bits, asymmetric())?;
        req_index[id] = Some(requants.len());
        requants.push(rq);
    }

    let tensors: Vec<IrTensorAbi> = graph
        .values
        .iter()
        .enumerate()
        .map(|(id, t)| {
            let ne = t.dims.map(|d| match d {
                Dim::Static(x) => x,
                Dim::Bounded { max, .. } => max,
            });
            let rq = req_index[id].map(|i| &requants[i]);
            IrTensorAbi {
                role: match t.role {
                    TensorRole::RuntimeInput => 0,
                    TensorRole::Constant => 1,
                    TensorRole::MutableState => 2,
                    TensorRole::Intermediate => 3,
                    TensorRole::Output => 4,
                },
                dtype: if t.dtype == crate::ffi::GGML_TYPE_I32 {
                    1
                } else if t.dtype == crate::ffi::GGML_TYPE_I64 {
                    2
                } else if t.dtype == crate::ffi::GGML_TYPE_F16 {
                    3
                } else {
                    0
                },
                ne,
                bits: rq.map_or(0, |x| x.bits),
                group: rq.map_or(0, |x| x.group),
                weight: rq.map_or(ptr::null(), |x| x.weight.as_ptr()),
                weight_len: rq.map_or(0, |x| x.weight.len()),
                scales: rq.map_or(ptr::null(), |x| x.scales.as_ptr()),
                scales_len: rq.map_or(0, |x| x.scales.len()),
                zeros: rq.map_or(ptr::null(), |x| x.zeros.as_ptr()),
                zeros_len: rq.map_or(0, |x| x.zeros.len()),
            }
        })
        .collect();

    let nodes: Option<Vec<IrNodeAbi>> = graph
        .nodes
        .iter()
        .map(|node| {
            let mut inputs = [-1i32; IR_MAX_INPUTS];
            for (dst, &src) in inputs.iter_mut().zip(&node.inputs) {
                *dst = src as i32;
            }
            Some(IrNodeAbi {
                op: GraphIr::wire_op(node)? as i32,
                n_inputs: node.inputs.len() as i32,
                inputs,
                output: node.output as i32,
                params: node.op_params,
            })
        })
        .collect();
    let nodes = nodes?;
    let runtime_inputs: Vec<u32> = graph
        .inputs
        .iter()
        .copied()
        .filter(|&id| graph.values[id as usize].role != TensorRole::Constant)
        .collect();

    let mut key_hasher = persistent_hasher(b"persistent/graph-ir-v2");
    key_hasher
        .u64(b"tensor-count", tensors.len() as u64)
        .u64(b"node-count", nodes.len() as u64)
        .u64(b"runtime-input-count", runtime_inputs.len() as u64)
        .u64(b"output-count", graph.outputs.len() as u64)
        .u64(b"attention-policy-full", u64::from(attention_full()))
        .u64(b"moe-policy-enabled", u64::from(moe_enabled()));
    for (index, node) in nodes.iter().enumerate() {
        key_hasher
            .u64(b"node-index", index as u64)
            .i32(b"node-op", node.op)
            .i32(b"node-input-count", node.n_inputs)
            .i32(b"node-output", node.output)
            .field(b"node-op-params", &node.params);
        for &input in node.inputs.iter().take(node.n_inputs as usize) {
            key_hasher.i32(b"node-input", input);
        }
    }
    for (index, (tensor, value)) in tensors.iter().zip(&graph.values).enumerate() {
        key_hasher
            .u64(b"tensor-index", index as u64)
            .i32(b"tensor-role", tensor.role)
            .i32(b"tensor-dtype", tensor.dtype)
            .i32(b"tensor-bits", tensor.bits)
            .i32(b"tensor-group", tensor.group);
        for dim in tensor.ne {
            key_hasher.i64(b"tensor-dimension", dim);
        }
        for dim in value.dims {
            match dim {
                Dim::Static(value) => {
                    key_hasher
                        .u8(b"tensor-dim-kind", 0)
                        .i64(b"tensor-dim", value);
                }
                Dim::Bounded { min, max } => {
                    key_hasher
                        .u8(b"tensor-dim-kind", 1)
                        .i64(b"tensor-dim-min", min)
                        .i64(b"tensor-dim-max", max);
                }
            }
        }
        for stride in value.strides {
            key_hasher.u64(b"tensor-stride", stride as u64);
        }
        if let Some(rq) = req_index[index].map(|rq_index| &requants[rq_index]) {
            let source_elements = value
                .dims
                .iter()
                .try_fold(1usize, |elements, dim| {
                    let value = match dim {
                        Dim::Static(value) => *value,
                        Dim::Bounded { max, .. } => *max,
                    };
                    elements.checked_mul(value as usize)
                })
                .unwrap_or(0);
            let digest = requant_digest(
                ConstantDigestKey {
                    generation: graph.generation,
                    source_address: value.data,
                    source_dtype: value.dtype as u32,
                    source_elements,
                    k: tensor.ne[0],
                    n: tensor.ne[1] * tensor.ne[2] * tensor.ne[3],
                    bits: rq.bits,
                    group: rq.group,
                    asymmetric: asymmetric(),
                },
                rq,
            );
            key_hasher.field(b"constant-content-digest", &digest.0);
        }
    }
    for &id in &runtime_inputs {
        key_hasher.u32(b"runtime-input-id", id);
    }
    for &id in &graph.outputs {
        key_hasher.u32(b"output-id", id);
    }
    let key = persistent_key("ir2", key_hasher, &alloc::format!("n{}", nodes.len()));
    let mut blob = ptr::null_mut();
    let mut len = 0usize;
    let rc = unsafe {
        compile(
            tensors.len() as i32,
            tensors.as_ptr(),
            nodes.len() as i32,
            nodes.as_ptr(),
            runtime_inputs.len() as i32,
            runtime_inputs.as_ptr(),
            graph.outputs.len() as i32,
            graph.outputs.as_ptr(),
            key.as_ptr(),
            &mut blob,
            &mut len,
        )
    };
    if rc != 0 || blob.is_null() || len == 0 {
        return None;
    }
    let out = unsafe { core::slice::from_raw_parts(blob, len) }.to_vec();
    unsafe { (s.free)(blob) };
    Some(out)
}

fn cache_key(
    rq: &Requant,
    source_address: usize,
    source_dtype: u32,
    m: i64,
    k: i64,
    n: i64,
) -> CString {
    let mut hasher = persistent_hasher(b"persistent/matmul-constant");
    let digest = requant_digest(
        ConstantDigestKey {
            generation: crate::backend::buffer_generation(),
            source_address,
            source_dtype,
            source_elements: (k * n) as usize,
            k,
            n,
            bits: rq.bits,
            group: rq.group,
            asymmetric: asymmetric(),
        },
        rq,
    );
    hasher
        .i64(b"m", m)
        .i64(b"k", k)
        .i64(b"n", n)
        .u32(b"source-dtype", source_dtype)
        .u64(b"requant-bits", rq.bits as u64)
        .u64(b"requant-group", rq.group as u64)
        .field(b"requant-content-digest", &digest.0);
    persistent_key(
        "matmul",
        hasher,
        &alloc::format!("m{m}-k{k}-n{n}-b{}", rq.bits),
    )
}

// ---------------------------------------------------------------------------
// Fused subgraph: re-quantize every baked weight, serialize the op-DAG to the flat ABI, and compile
// it as one transient blob (disk-cached in the shim). The built graph is cached in memory by the
// exact **semantic key** ([`sg_semantic_key`]) — NOT the output pointer — inside
// [`session::run_subgraph`](super::session::run_subgraph): ggml's graph allocator recycles buffers
// across non-overlapping tensors, so one output address can belong to two structurally-different
// subgraphs within a single graph; keying on it aliased them (a 2-input program served to a 4-input
// subgraph → arg-count mismatch → dispatch fail). The signature (M, input shapes, node DAG, const
// weight pointers + shapes, output selection) identifies the compiled program exactly, is stable
// across tokens (so the graph is reused, not recompiled), and needs no re-quantization to compute.
// ---------------------------------------------------------------------------

/// Compile a recognized subgraph into one **transient** native blob (weights baked as Constants),
/// which the caller parses into a graph and then drops. `None` for a malformed subgraph or a
/// shim/requant failure. Called once per distinct [`sg_semantic_key`] on a
/// [`session::run_subgraph`](super::session::run_subgraph) graph-cache miss.
pub(super) fn compile_subgraph_blob(sg: &Subgraph) -> Option<Vec<u8>> {
    if sg.m <= 0
        || sg.inputs.is_empty()
        || sg.nodes.is_empty()
        || sg.outputs.is_empty()
        || sg.outs.is_empty()
    {
        return None;
    }
    if sg.outputs.iter().any(|&o| o as usize >= sg.nodes.len()) {
        return None;
    }
    compile_subgraph(sg)
}

/// Exact owned in-process key for a subgraph's compiled program: M, input metadata, node DAG,
/// op parameters, baked-constant identity/shape, and ordered outputs. BLAKE3 is a prefilter only;
/// cache equality also compares the full length-framed semantic bytes. Persistent files use the
/// separate content-derived [`subgraph_cache_key`].
pub(super) fn sg_semantic_key(sg: &Subgraph) -> crate::identity::ExactSemanticKey {
    let mut key = crate::identity::ExactKeyBuilder::new(b"fused-subgraph/program");
    key.i64(b"m", sg.m)
        .u64(b"input-count", sg.inputs.len() as u64)
        .u64(b"node-count", sg.nodes.len() as u64)
        .u64(b"constant-count", sg.consts.len() as u64)
        .u64(b"output-count", sg.outputs.len() as u64);
    for inp in &sg.inputs {
        key.u64(b"input-elements", inp.elems as u64)
            .u64(b"input-dtype", inp.dtype as u64);
    }
    let enc = |r: Ref| -> u32 {
        match r {
            Ref::Input(i) => i,
            Ref::Node(j) => 0x8000_0000 | j,
        }
    };
    for nd in &sg.nodes {
        let op: u8 = match nd.op {
            SgOp::MatMul { n } => {
                key.i64(b"matmul-n", n);
                1
            }
            SgOp::Mul => 2,
            SgOp::Silu => 3,
            SgOp::Add => 4,
            SgOp::RmsNorm { eps } => {
                key.u32(b"rms-eps", eps.to_bits());
                5
            }
        };
        key.u8(b"node-op", op)
            .u32(b"node-a", enc(nd.a))
            .u32(b"node-b", nd.b.map(enc).unwrap_or(u32::MAX))
            .u32(
                b"node-weight",
                nd.weight.map(|weight| weight as u32).unwrap_or(u32::MAX),
            );
    }
    for c in &sg.consts {
        key.u64(b"constant-address", c.weight.ptr as usize as u64)
            .u64(b"constant-elements", c.weight.elems as u64)
            .u64(b"constant-dtype", c.weight.dtype as u64)
            .i64(b"constant-k", c.k)
            .i64(b"constant-n", c.n);
    }
    for &o in &sg.outputs {
        key.u64(b"output-node", o as u64);
    }
    key.finish()
}

fn compile_subgraph(sg: &Subgraph) -> Option<Vec<u8>> {
    let s = shim()?;
    let compile = s.compile_subgraph?;
    let force = forced_bits();

    // Re-quantize every baked weight (bytes must outlive the shim call → held in `requants`).
    let mut requants: Vec<Requant> = Vec::with_capacity(sg.consts.len());
    for c in &sg.consts {
        requants.push(requant::requant(
            &c.weight,
            c.k as usize,
            c.n as usize,
            choose_bits(&c.weight, force),
            asymmetric(),
        )?);
    }
    let consts: Vec<SgConstAbi> = requants
        .iter()
        .zip(&sg.consts)
        .map(|(rq, c)| SgConstAbi {
            bits: rq.bits,
            group: rq.group,
            n: c.n as i32,
            k: c.k as i32,
            weight: rq.weight.as_ptr(),
            weight_len: rq.weight.len(),
            scales: rq.scales.as_ptr(),
            scales_len: rq.scales.len(),
            zeros: rq.zeros.as_ptr(),
            zeros_len: rq.zeros.len(),
        })
        .collect();

    let n_inputs = sg.inputs.len() as i32;
    let enc = |r: Ref| -> i32 {
        match r {
            Ref::Input(i) => i as i32,
            Ref::Node(j) => n_inputs + j as i32,
        }
    };
    let nodes: Vec<SgNodeAbi> = sg
        .nodes
        .iter()
        .map(|nd| {
            let (op, out_n) = match nd.op {
                SgOp::MatMul { n } => (1, n as i32),
                SgOp::Mul => (2, 0),
                SgOp::Add => (4, 0),
                SgOp::Silu => (3, 0),
                SgOp::RmsNorm { eps } => (5, eps.to_bits() as i32),
            };
            SgNodeAbi {
                op,
                a: enc(nd.a),
                b: nd.b.map(enc).unwrap_or(-1),
                const_idx: nd.weight.map(|w| w as i32).unwrap_or(-1),
                out_n,
            }
        })
        .collect();

    let out_nodes: Vec<i32> = sg.outputs.iter().map(|&o| o as i32).collect();
    let input_k: Vec<i32> = sg
        .inputs
        .iter()
        .map(|op| (op.elems / sg.m as usize) as i32)
        .collect();
    let key = subgraph_cache_key(sg, &requants, &nodes, &input_k, &out_nodes);

    let mut blob: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    let rc = unsafe {
        compile(
            sg.m as i32,
            n_inputs,
            input_k.as_ptr(),
            nodes.len() as i32,
            nodes.as_ptr(),
            consts.len() as i32,
            consts.as_ptr(),
            out_nodes.as_ptr(),
            out_nodes.len() as i32,
            key.as_ptr(),
            &mut blob,
            &mut len,
        )
    };
    if rc != 0 || blob.is_null() || len == 0 {
        return None;
    }
    let out = unsafe { core::slice::from_raw_parts(blob, len) }.to_vec();
    unsafe { (s.free)(blob) };
    Some(out)
}

fn subgraph_cache_key(
    sg: &Subgraph,
    requants: &[Requant],
    nodes: &[SgNodeAbi],
    input_k: &[i32],
    out_nodes: &[i32],
) -> CString {
    let mut hasher = persistent_hasher(b"persistent/fused-subgraph");
    hasher
        .i64(b"m", sg.m)
        .u64(b"input-count", input_k.len() as u64)
        .u64(b"node-count", nodes.len() as u64)
        .u64(b"constant-count", requants.len() as u64)
        .u64(b"output-count", out_nodes.len() as u64);
    for k in input_k {
        hasher.i32(b"input-k", *k);
    }
    for input in &sg.inputs {
        hasher
            .u64(b"input-elements", input.elems as u64)
            .u64(b"input-dtype", input.dtype as u64);
    }
    for nd in nodes {
        for v in [nd.op, nd.a, nd.b, nd.const_idx, nd.out_n] {
            hasher.i32(b"node-field", v);
        }
    }
    // The output selection (which nodes are Results, and their order) is part of the compiled model.
    for o in out_nodes {
        hasher.i32(b"output-node", *o);
    }
    for (index, rq) in requants.iter().enumerate() {
        if let Some(constant) = sg.consts.get(index) {
            let digest = requant_digest(
                ConstantDigestKey {
                    generation: crate::backend::buffer_generation(),
                    source_address: constant.weight.ptr as usize,
                    source_dtype: constant.weight.dtype as u32,
                    source_elements: constant.weight.elems,
                    k: constant.k,
                    n: constant.n,
                    bits: rq.bits,
                    group: rq.group,
                    asymmetric: asymmetric(),
                },
                rq,
            );
            hasher
                .i64(b"constant-k", constant.k)
                .i64(b"constant-n", constant.n)
                .u64(b"constant-source-dtype", constant.weight.dtype as u64)
                .field(b"constant-content-digest", &digest.0);
        }
        hasher
            .u64(b"requant-bits", rq.bits as u64)
            .u64(b"requant-group", rq.group as u64);
    }
    persistent_key(
        "sg",
        hasher,
        &alloc::format!("m{}-n{}-o{}", sg.m, sg.nodes.len(), out_nodes.len()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requant(last: u8) -> Requant {
        Requant {
            weight: vec![1, 2, 3, last],
            zeros: vec![4],
            scales: vec![5, 6],
            bits: 4,
            group: 32,
        }
    }

    #[test]
    fn persistent_key_is_full_blake3_and_field_sensitive() {
        let first = cache_key(&requant(7), 0x1000, 0, 1, 32, 1);
        let second = cache_key(&requant(8), 0x2000, 0, 1, 32, 1);
        let first = first.to_str().unwrap();
        let second = second.to_str().unwrap();
        assert!(first.starts_with("matmul-b3v7-"));
        assert_eq!(first.split('-').nth(2).unwrap().len(), 64);
        assert_ne!(first, second);
        assert_ne!(
            cache_key(&requant(7), 0x1000, 0, 1, 32, 1),
            cache_key(&requant(7), 0x1000, 1, 1, 32, 1)
        );
    }
}
