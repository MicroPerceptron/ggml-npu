//! Constant-blob compilation for the Intel NPU: shape + weights → a native NPU program blob.
//!
//! The OpenVINO compile shim (`libggml_npu_ovc.so`) is `dlopen`'d on demand and, given a weight
//! re-quantized to low-bit form ([`crate::ops::requant`]), bakes it as a compile-time `Constant` and
//! compiles a native NPU blob (the fused decompress-matmul). Canonical keys are computed from raw
//! constant identity before requantization, then the neutral host cache checks compressed memory
//! and durable content-addressed tiers. The returned uncompressed blob is transient: the selected
//! runtime parses/prepares it and drops it, while T0 retains only its prepared program and T1/T2
//! retain only the compressed artifact record.
//!
//! `no_std`: the selected host provider supplies configuration and dynamic loading. A compile
//! failure is cached (as `None`) and not retried.

use alloc::boxed::Box;
use alloc::ffi::CString;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_void};
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use crate::artifact_cache::{ArtifactKey, CacheCompatibility};
use crate::graph::{Dim, GraphIr, TensorRole, OP_PARAM_BYTES};
use crate::host::{ConfigService, DynamicLibraryService, HostEnvironment, LibraryVisibility, MemoryStatusService};
use crate::identity::{SemanticDigest, SemanticHasher};
use crate::ops::requant::{self, Requant};
use crate::ops::subgraph::{Ref, SgOp, Subgraph};
use crate::ops::Operand;

/// Persistent artifact namespace. Increment whenever compiler semantics or framing changes.
const FORMAT_VERSION: u64 = 9;

const MIB: u64 = 1024 * 1024;
const DEFAULT_HOST_FLOOR_BYTES: u64 = 4096 * MIB;
const DEFAULT_EXPANDED_CONSTANT_FRACTION: u64 = 8;
const DEFAULT_MAX_EXPANDED_CONSTANT_BYTES: u64 = 256 * MIB;
static EXPANDED_CONSTANT_BUDGET: AtomicU64 = AtomicU64::new(u64::MAX);

fn config_mib(name: &core::ffi::CStr) -> Option<u64> {
    let value = crate::host::builtin().config().value(name)?;
    let mut parsed = 0u64;
    for &byte in value.to_bytes() {
        if !byte.is_ascii_digit() {
            return None;
        }
        parsed = parsed
            .checked_mul(10)?
            .checked_add(u64::from(byte - b'0'))?;
    }
    parsed.checked_mul(MIB)
}

fn predicted_compiler_constant_bytes(elements: u64) -> u64 {
    // Current native ELF evidence is approximately f16 plus compiler alignment/metadata. Keep a
    // small safety margin so placement rejects before a 500+ MiB payload enters the T0 cycle.
    elements.saturating_mul(33).saturating_add(15) / 16
}

fn expanded_constant_budget() -> u64 {
    let cached = EXPANDED_CONSTANT_BUDGET.load(Ordering::Acquire);
    if cached != u64::MAX {
        return cached;
    }
    let budget = config_mib(c"GGML_NPU_EXPANDED_CONSTANT_MB").unwrap_or_else(|| {
        let resident = config_mib(c"GGML_NPU_RESIDENT_MB").unwrap_or_else(|| {
            crate::host::builtin()
                .memory()
                .available_memory_bytes()
                .unwrap_or(0)
                .saturating_sub(DEFAULT_HOST_FLOOR_BYTES)
        });
        (resident / DEFAULT_EXPANDED_CONSTANT_FRACTION).min(DEFAULT_MAX_EXPANDED_CONSTANT_BYTES)
    });
    match EXPANDED_CONSTANT_BUDGET.compare_exchange(
        u64::MAX,
        budget,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => budget,
        Err(existing) => existing,
    }
}

fn expanded_constant_admitted(predicted: u64, budget: u64, override_enabled: bool) -> bool {
    override_enabled || (budget != 0 && predicted <= budget)
}

fn constant_elements(value: &crate::graph::TensorDesc) -> u64 {
    value.dims.into_iter().fold(1u64, |product, dim| {
        let extent = match dim {
            Dim::Static(value) => value,
            Dim::Bounded { max, .. } => max,
        };
        product.saturating_mul(extent.max(0) as u64)
    })
}

fn graph_constant_admitted(graph: &GraphIr, node: &crate::graph::Node) -> bool {
    if GraphIr::wire_op(node) != Some(crate::graph::WireOp::MatMul) {
        return true;
    }
    let Some(weight) = node
        .inputs
        .first()
        .and_then(|&id| graph.values.get(id as usize))
        .filter(|value| value.role == TensorRole::Constant)
    else {
        return true;
    };
    let predicted = predicted_compiler_constant_bytes(constant_elements(weight));
    let budget = expanded_constant_budget();
    let override_enabled = crate::host::builtin()
        .config()
        .is_set(c"GGML_NPU_ALLOW_GIANT_CONSTANTS");
    let admitted = expanded_constant_admitted(predicted, budget, override_enabled);
    if !admitted {
        crate::trace::perf(
            "placement_rejection",
            &alloc::format!(
                ",\"reason\":\"expanded_constant_budget\",\"predicted_bytes\":{predicted},\"budget_bytes\":{budget},\"override\":\"GGML_NPU_ALLOW_GIANT_CONSTANTS\""
            ),
        );
    }
    admitted
}

unsafe fn tensor_constant_admitted(node: *const crate::ffi::ggml_tensor) -> bool {
    if node.is_null() || (*node).op != crate::ffi::GGML_OP_MUL_MAT {
        return true;
    }
    let weight = (*node).src[0];
    if weight.is_null() || (*weight).op != crate::ffi::GGML_OP_NONE {
        return true;
    }
    let elements = (*weight).ne.iter().fold(1u64, |product, &extent| {
        product.saturating_mul(extent.max(0) as u64)
    });
    let predicted = predicted_compiler_constant_bytes(elements);
    let budget = expanded_constant_budget();
    let override_enabled = crate::host::builtin()
        .config()
        .is_set(c"GGML_NPU_ALLOW_GIANT_CONSTANTS");
    let admitted = expanded_constant_admitted(predicted, budget, override_enabled);
    if !admitted {
        crate::trace::perf(
            "placement_rejection",
            &alloc::format!(
                ",\"reason\":\"expanded_constant_budget\",\"predicted_bytes\":{predicted},\"budget_bytes\":{budget},\"override\":\"GGML_NPU_ALLOW_GIANT_CONSTANTS\""
            ),
        );
    }
    admitted
}


fn execution_runtime_available() -> bool {
    #[cfg(feature = "level-zero-compat")]
    {
        super::ze::runtime().is_some()
    }
    #[cfg(not(feature = "level-zero-compat"))]
    {
        super::session::runtime_available()
    }
}

fn persistent_hasher(domain: &[u8]) -> SemanticHasher {
    let mut hasher = SemanticHasher::new(domain);
    hasher
        .u64(b"format-version", FORMAT_VERSION)
        .field(b"target", b"intel-npu")
        .field(b"compiler-adapter", b"openvino-native-graph")
        .u64(b"asymmetric-requant", u64::from(asymmetric()));
    hasher
}

fn identity(domain: &[u8], value: &[u8]) -> SemanticDigest {
    let mut hasher = SemanticHasher::new(domain);
    hasher.field(b"identity", value);
    hasher.finish()
}

fn cache_compatibility() -> CacheCompatibility {
    CacheCompatibility {
        artifact_schema: FORMAT_VERSION as u32,
        compiler: identity(b"artifact/compiler", b"native-graph-compiler"),
        pipeline: identity(b"artifact/pipeline", b"graph-ir-v2-requant-v9"),
        toolchain: identity(b"artifact/toolchain", b"compiler-provider-abi-v1"),
        target: identity(b"artifact/target", b"native-npu-elf"),
        firmware: identity(b"artifact/firmware", b"firmware-command-api-v1"),
        runtime: identity(b"artifact/runtime", b"native-graph-loader-v1"),
    }
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

#[derive(Clone, Copy, PartialEq, Eq)]
struct RawConstantDigestKey {
    generation: u64,
    source_address: usize,
    source_dtype: u32,
    source_elements: usize,
}

static RAW_CONSTANT_DIGESTS: super::cache::Cache<RawConstantDigestKey, SemanticDigest> =
    super::cache::Cache::new(4096);

fn operand_storage_bytes(operand: &Operand) -> Option<usize> {
    let traits = unsafe { crate::ffi::ggml_get_type_traits(operand.dtype) };
    if traits.is_null() {
        return None;
    }
    let traits = unsafe { &*traits };
    let block = usize::try_from(traits.blck_size).ok()?;
    if block == 0 || !operand.elems.is_multiple_of(block) {
        return None;
    }
    operand
        .elems
        .checked_div(block)?
        .checked_mul(traits.type_size)
}

fn compute_raw_constant_digest(operand: &Operand) -> Option<SemanticDigest> {
    let bytes = operand_storage_bytes(operand)?;
    if operand.ptr.is_null() {
        return None;
    }
    let source = unsafe { core::slice::from_raw_parts(operand.ptr.cast::<u8>(), bytes) };
    let mut hasher = SemanticHasher::new(b"raw-constant/content");
    hasher
        .u32(b"dtype", operand.dtype)
        .u64(b"elements", operand.elems as u64)
        .field(b"bytes", source);
    Some(hasher.finish())
}

fn raw_constant_digest(generation: u64, operand: &Operand) -> Option<SemanticDigest> {
    let key = RawConstantDigestKey {
        generation,
        source_address: operand.ptr as usize,
        source_dtype: operand.dtype,
        source_elements: operand.elems,
    };
    if let Some(digest) = RAW_CONSTANT_DIGESTS.get(&key) {
        return Some(digest);
    }
    let digest = compute_raw_constant_digest(operand)?;
    Some(
        RAW_CONSTANT_DIGESTS
            .try_get_or_insert_with(key, || digest)
            .unwrap_or(digest),
    )
}

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
    _library: crate::host::BuiltinLibrary,
    compile_const: CompileConst,
    compile_subgraph: Option<CompileSubgraph>,
    compile_graph: Option<CompileGraph>,
    free: FreeBlob,
}

/// `null` = untried, `FAILED` = tried & unavailable, else a leaked `*mut Shim`.
static SHIM: AtomicPtr<Shim> = AtomicPtr::new(ptr::null_mut());
const SHIM_FAILED: *mut Shim = ptr::dangling_mut();

unsafe fn load_shim() -> Option<Shim> {
    let host = crate::host::builtin();
    let path = host
        .config()
        .value(c"GGML_NPU_OVC")
        .unwrap_or(c"libggml_npu_ovc.so");
    let library = match host.libraries().open(path, LibraryVisibility::Local) {
        Ok(library) => library,
        Err(_) => {
            if let Some(error) = host.libraries().last_error() {
                super::profile::note(&alloc::format!(
                    "[npu ir2] shim load failed: {}\n",
                    error.to_string_lossy()
                ));
            }
            return None;
        }
    };
    let compile_const = unsafe {
        host.libraries()
            .symbol(&library, c"ggml_npu_ov_compile_matmul_const")
            .ok()?
    };
    let free = unsafe {
        host.libraries()
            .symbol(&library, c"ggml_npu_ov_free")
            .ok()?
    };
    // Optional: an older shim without subgraph support still serves the single-matmul path.
    let compile_subgraph = unsafe {
        host.libraries()
            .symbol(&library, c"ggml_npu_ov_compile_subgraph")
            .ok()
            .map(|pointer| mem::transmute::<*mut c_void, CompileSubgraph>(pointer))
    };
    let compile_graph = unsafe {
        host.libraries()
            .symbol(&library, c"ggml_npu_ov_compile_graph")
            .ok()
            .map(|pointer| mem::transmute::<*mut c_void, CompileGraph>(pointer))
    };
    Some(Shim {
        _library: library,
        compile_const: unsafe { mem::transmute::<*mut c_void, CompileConst>(compile_const) },
        compile_subgraph,
        compile_graph,
        free: unsafe { mem::transmute::<*mut c_void, FreeBlob>(free) },
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
        && execution_runtime_available()
        && shim().is_some()
        && requant::can_requant(weight, k as usize)
}

fn const_artifact_key(weight: &Operand, m: i64, k: i64, n: i64) -> Option<ArtifactKey> {
    let source_elements = usize::try_from(k.checked_mul(n)?).ok()?;
    if weight.elems < source_elements {
        return None;
    }
    let bits = choose_bits(weight, forced_bits());
    let digest = raw_constant_digest(crate::backend::buffer_generation(), weight)?;
    let mut hasher = persistent_hasher(b"artifact/matmul-constant");
    hasher
        .i64(b"m", m)
        .i64(b"k", k)
        .i64(b"n", n)
        .u32(b"source-dtype", weight.dtype)
        .u64(b"source-elements", source_elements as u64)
        .u64(b"requant-bits", bits as u64)
        .u64(b"asymmetric", u64::from(asymmetric()))
        .field(b"raw-constant-digest", &digest.0);
    Some(ArtifactKey(hasher.finish()))
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
    let key = const_artifact_key(weight, m, k, n)?;
    crate::host::artifact_cache::get_or_compile(
        key,
        cache_compatibility(),
        &super::profile::ARTIFACT_CACHE_OBSERVER,
        || compile_const(weight, m, k, n),
    )
}

/// Optional weight-width override from `GGML_NPU_BITS` (`"4"`, `"8"`, or `"16"`); unset/other →
/// auto per source type. Reads the first two bytes of the C string (enough to distinguish).
fn forced_bits() -> Option<i32> {
    match crate::host::builtin()
        .config()
        .value(c"GGML_NPU_BITS")?
        .to_bytes()
    {
        [b'4', ..] => Some(4),
        [b'8', ..] => Some(8),
        [b'1', b'6', ..] => Some(16),
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
    !crate::host::builtin().config().is_set(c"GGML_NPU_SYM")
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
            c"".as_ptr(),
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
    if !execution_runtime_available() {
        super::profile::note("[npu ir2] unavailable: selected execution runtime rejected device\n");
        return false;
    }
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
        if !graph_constant_admitted(graph, node) {
            super::profile::note(&alloc::format!(
                "[npu ir2] node {index} rejected: expanded_constant_budget\n"
            ));
            return false;
        }
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
        let runtime_matmul = op == crate::graph::WireOp::MatMul
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
            crate::graph::WireOp::MatMul
                | crate::graph::WireOp::Mul
                | crate::graph::WireOp::Add
                | crate::graph::WireOp::RmsNorm
                | crate::graph::WireOp::Silu
                | crate::graph::WireOp::Reshape
                | crate::graph::WireOp::View
                | crate::graph::WireOp::Cont
                | crate::graph::WireOp::Transpose
                | crate::graph::WireOp::Concat
                | crate::graph::WireOp::Copy
                | crate::graph::WireOp::SsmConv
                | crate::graph::WireOp::Rope
                | crate::graph::WireOp::GetRows
                | crate::graph::WireOp::SetRows
                | crate::graph::WireOp::Scale
                | crate::graph::WireOp::Div
                | crate::graph::WireOp::Clamp
                | crate::graph::WireOp::Argsort
                | crate::graph::WireOp::MatMulId
                | crate::graph::WireOp::Sigmoid
                | crate::graph::WireOp::SumRows
                | crate::graph::WireOp::SoftMax
                | crate::graph::WireOp::FlashAttention
                | crate::graph::WireOp::Permute
                | crate::graph::WireOp::Swiglu
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
    spec: &crate::graph::OpSpec,
) -> bool {
    if !tensor_constant_admitted(node) {
        return false;
    }
    let op = Some(spec.wire);
    let moe_only = matches!(
        op,
        Some(
            crate::graph::WireOp::MatMulId
                | crate::graph::WireOp::Argsort
                | crate::graph::WireOp::Sigmoid
                | crate::graph::WireOp::SumRows
        )
    );
    let attention_only = matches!(
        op,
        Some(crate::graph::WireOp::SoftMax | crate::graph::WireOp::Permute)
    ) || (op == Some(crate::graph::WireOp::MatMul)
        && !node.is_null()
        && !(*node).src[0].is_null()
        && (*(*node).src[0]).op != crate::ffi::GGML_OP_NONE);
    shim().and_then(|s| s.compile_graph).is_some()
        && op.is_some()
        && (!moe_only || moe_enabled())
        && (!attention_only || attention_full())
}

fn moe_enabled() -> bool {
    crate::host::builtin().config().is_set(c"GGML_NPU_MOE")
}

fn attention_full() -> bool {
    crate::host::builtin()
        .config()
        .value(c"GGML_NPU_ATTENTION")
        .is_some_and(|value| value.to_bytes() == b"full")
}

fn graph_artifact_key(graph: &GraphIr) -> Option<ArtifactKey> {
    let mut hasher = persistent_hasher(b"artifact/graph-ir-v2");
    hasher
        .u64(b"value-count", graph.values.len() as u64)
        .u64(b"node-count", graph.nodes.len() as u64)
        .u64(b"input-count", graph.inputs.len() as u64)
        .u64(b"output-count", graph.outputs.len() as u64)
        .u64(b"attention-policy-full", u64::from(attention_full()))
        .u64(b"moe-policy-enabled", u64::from(moe_enabled()));
    for (index, node) in graph.nodes.iter().enumerate() {
        hasher
            .u64(b"node-index", index as u64)
            .u32(b"node-op", GraphIr::wire_op(node)? as u32)
            .u64(b"node-input-count", node.inputs.len() as u64)
            .u32(b"node-output", node.output)
            .field(b"node-op-params", &node.op_params);
        for &input in &node.inputs {
            hasher.u32(b"node-input", input);
        }
    }
    for (index, value) in graph.values.iter().enumerate() {
        let role = match value.role {
            TensorRole::RuntimeInput => 0,
            TensorRole::Constant => 1,
            TensorRole::MutableState => 2,
            TensorRole::Intermediate => 3,
            TensorRole::Output => 4,
        };
        hasher
            .u64(b"value-index", index as u64)
            .u32(b"value-role", role)
            .u32(b"value-source-dtype", value.dtype);
        let mut elements = 1usize;
        for dimension in value.dims {
            match dimension {
                Dim::Static(dimension) => {
                    hasher
                        .u8(b"value-dimension-kind", 0)
                        .i64(b"value-dimension", dimension);
                    elements = elements.checked_mul(usize::try_from(dimension).ok()?)?;
                }
                Dim::Bounded { min, max } => {
                    hasher
                        .u8(b"value-dimension-kind", 1)
                        .i64(b"value-dimension-min", min)
                        .i64(b"value-dimension-max", max);
                    elements = elements.checked_mul(usize::try_from(max).ok()?)?;
                }
            }
        }
        for stride in value.strides {
            hasher.u64(b"value-stride", stride as u64);
        }
        if value.role == TensorRole::Constant {
            let operand = Operand {
                ptr: value.data as *mut c_void,
                elems: elements,
                dtype: value.dtype,
                owned: value.owned,
            };
            let k = match value.dims[0] {
                Dim::Static(value) => value,
                Dim::Bounded { max, .. } => max,
            };
            let bits = if k % 32 == 0 {
                choose_bits(&operand, forced_bits())
            } else {
                16
            };
            let digest = raw_constant_digest(graph.generation, &operand)?;
            hasher
                .u64(b"constant-requant-bits", bits as u64)
                .field(b"constant-raw-digest", &digest.0);
        }
    }
    for &input in &graph.inputs {
        hasher.u32(b"graph-input", input);
    }
    for &output in &graph.outputs {
        hasher.u32(b"graph-output", output);
    }
    Some(ArtifactKey(hasher.finish()))
}

pub(super) fn compile_graph_blob(graph: &GraphIr) -> Option<Vec<u8>> {
    if !graph_feasible(graph) {
        return None;
    }
    let key = graph_artifact_key(graph)?;
    crate::host::artifact_cache::get_or_compile(
        key,
        cache_compatibility(),
        &super::profile::ARTIFACT_CACHE_OBSERVER,
        || compile_graph_uncached(graph),
    )
}

fn compile_graph_uncached(graph: &GraphIr) -> Option<Vec<u8>> {
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
        let bits = if k.is_multiple_of(32) {
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
            c"".as_ptr(),
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
fn subgraph_artifact_key(sg: &Subgraph) -> Option<ArtifactKey> {
    let mut hasher = persistent_hasher(b"artifact/fused-subgraph");
    hasher
        .i64(b"m", sg.m)
        .u64(b"input-count", sg.inputs.len() as u64)
        .u64(b"node-count", sg.nodes.len() as u64)
        .u64(b"constant-count", sg.consts.len() as u64)
        .u64(b"output-count", sg.outputs.len() as u64);
    for input in &sg.inputs {
        hasher
            .u64(b"input-elements", input.elems as u64)
            .u32(b"input-dtype", input.dtype);
    }
    let encode = |reference: Ref| match reference {
        Ref::Input(index) => index,
        Ref::Node(index) => 0x8000_0000 | index,
    };
    for node in &sg.nodes {
        let opcode = match node.op {
            SgOp::MatMul { n } => {
                hasher.i64(b"matmul-n", n);
                1
            }
            SgOp::Mul => 2,
            SgOp::Silu => 3,
            SgOp::Add => 4,
            SgOp::RmsNorm { eps } => {
                hasher.u32(b"rms-eps", eps.to_bits());
                5
            }
        };
        hasher
            .u8(b"node-op", opcode)
            .u32(b"node-a", encode(node.a))
            .u32(b"node-b", node.b.map(encode).unwrap_or(u32::MAX))
            .u32(b"node-constant", node.weight.unwrap_or(u32::MAX));
    }
    let force = forced_bits();
    for constant in &sg.consts {
        let bits = choose_bits(&constant.weight, force);
        let digest = raw_constant_digest(crate::backend::buffer_generation(), &constant.weight)?;
        hasher
            .i64(b"constant-k", constant.k)
            .i64(b"constant-n", constant.n)
            .u32(b"constant-source-dtype", constant.weight.dtype)
            .u64(b"constant-elements", constant.weight.elems as u64)
            .u64(b"constant-requant-bits", bits as u64)
            .field(b"constant-raw-digest", &digest.0);
    }
    for &output in &sg.outputs {
        hasher.u32(b"output-node", output);
    }
    Some(ArtifactKey(hasher.finish()))
}

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
    let key = subgraph_artifact_key(sg)?;
    crate::host::artifact_cache::get_or_compile(
        key,
        cache_compatibility(),
        &super::profile::ARTIFACT_CACHE_OBSERVER,
        || compile_subgraph(sg),
    )
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
            .u32(b"node-weight", nd.weight.unwrap_or(u32::MAX));
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
            c"".as_ptr(),
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
                    source_dtype: constant.weight.dtype,
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
        assert!(first.starts_with("matmul-b3v9-"));
        assert_eq!(first.split('-').nth(2).unwrap().len(), 64);
        assert_ne!(first, second);
        assert_ne!(
            cache_key(&requant(7), 0x1000, 0, 1, 32, 1),
            cache_key(&requant(7), 0x1000, 1, 1, 32, 1)
        );
    }

    #[test]
    fn expanded_constant_admission_is_budget_based_and_overridable() {
        let output_head = predicted_compiler_constant_bytes(128_000 * 2_048);
        assert!(output_head > 500 * MIB);
        assert!(!expanded_constant_admitted(output_head, 384 * MIB, false));
        assert!(expanded_constant_admitted(output_head, 600 * MIB, false));
        assert!(expanded_constant_admitted(output_head, 1, true));
    }

    #[test]
    fn expanded_constant_prediction_tracks_f16_sized_native_payload() {
        let elements = 512 * 2_048;
        let predicted = predicted_compiler_constant_bytes(elements);
        assert!(predicted >= elements * 2);
        assert!(predicted <= elements * 3);
    }
}
