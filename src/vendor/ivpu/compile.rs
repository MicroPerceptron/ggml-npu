//! Per-weight Constant-blob compilation + cache for the Intel NPU.
//!
//! The OpenVINO compile shim (`libggml_npu_ovc.so`) is `dlopen`'d on demand and, given a weight
//! re-quantized to low-bit form ([`super::weights`]), bakes it as a compile-time `Constant` and
//! compiles a native NPU blob (the fused decompress-matmul). Blobs are cached two ways:
//! - **in memory** by `(weight pointer, M)` — so a weight compiles at most once per process per M
//!   (decode M=1 and any prefill M each get their own blob); reads are the lock-free [`super::cache`].
//! - **on disk** by a content hash — managed inside the shim (C++) so this core stays `no_std` and
//!   file-I/O-free; a warm process reloads a ~few-ms blob instead of recompiling (~100 ms).
//!
//! `no_std`: the shim, `dlopen`, and `getenv` resolve from the host. A compile failure is cached
//! (as `None`) and not retried.

use alloc::boxed::Box;
use alloc::ffi::CString;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_void};
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use super::cache::Cache;
use super::weights::{self, Requant};
use crate::ops::subgraph::{Ref, SgOp, Subgraph};
use crate::ops::Operand;

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn getenv(name: *const c_char) -> *const c_char;
}
const RTLD_NOW: c_int = 2;

// int ggml_npu_ov_compile_matmul_const(M, K, N, bits, group, weight, weight_len,
//                                      scales, scales_len, cache_key, &blob, &len)
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
    *const c_char,
    *mut *mut u8,
    *mut usize,
) -> c_int;
type FreeBlob = unsafe extern "C" fn(*mut u8);

// int ggml_npu_ov_compile_subgraph(M, n_inputs, input_k, n_nodes, nodes, n_consts, consts,
//                                  out_node, cache_key, &blob, &len)
type CompileSubgraph = unsafe extern "C" fn(
    i32,
    i32,
    *const i32,
    i32,
    *const SgNodeAbi,
    i32,
    *const SgConstAbi,
    i32,
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
}

struct Shim {
    compile_const: CompileConst,
    compile_subgraph: Option<CompileSubgraph>,
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
    Some(Shim {
        compile_const: mem::transmute::<*mut c_void, CompileConst>(compile_const),
        compile_subgraph,
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
            match SHIM.compare_exchange(ptr::null_mut(), boxed, Ordering::AcqRel, Ordering::Acquire) {
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
            let _ = SHIM.compare_exchange(ptr::null_mut(), SHIM_FAILED, Ordering::AcqRel, Ordering::Acquire);
            None
        }
    }
}

/// `None` = compiled and failed (cached, not retried); `Some` = the leaked native blob.
/// Key = `(weight data pointer, M)`: a weight's blob is specific to it *and* the activation rows.
static CACHE: Cache<(usize, i64), Option<&'static [u8]>> = Cache::new();

/// Get the cached Constant blob for `weight` at shape `(M, K, N)`, re-quantizing + compiling (or
/// disk-loading) on a miss. `None` if the shim/NPU is absent or the weight can't take this path.
pub(super) fn get_or_compile_const(weight: &Operand, m: i64, k: i64, n: i64) -> Option<&'static [u8]> {
    if m <= 0 || k <= 0 || n <= 0 || m > i32::MAX as i64 || k > i32::MAX as i64 || n > i32::MAX as i64 {
        return None;
    }
    CACHE.get_or_insert_with((weight.ptr as usize, m), || {
        compile_const(weight, m, k, n).map(|v| &*Vec::leak(v))
    })
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

fn compile_const(weight: &Operand, m: i64, k: i64, n: i64) -> Option<Vec<u8>> {
    let s = shim()?;
    let rq = weights::requant(weight, k as usize, n as usize, forced_bits())?;
    let key = cache_key(&rq, m, k, n);

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

/// Format version folded into the disk-cache key so a shim/requant/format change invalidates
/// stale blobs automatically.
const FORMAT_VERSION: u64 = 1;

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Content-derived, cross-run-stable disk-cache key: a 64-bit FNV-1a over the format version, the
/// shape/quant params, and the re-quantized weight + scales bytes, plus a human-readable suffix.
fn cache_key(rq: &Requant, m: i64, k: i64, n: i64) -> CString {
    let mut h = 0xcbf2_9ce4_8422_2325u64; // FNV-1a offset basis
    for v in [FORMAT_VERSION, m as u64, k as u64, n as u64, rq.bits as u64, rq.group as u64] {
        h = fnv1a(h, &v.to_le_bytes());
    }
    h = fnv1a(h, &rq.weight);
    h = fnv1a(h, &rq.scales);
    let s = alloc::format!("{h:016x}-m{m}-k{k}-n{n}-b{}", rq.bits);
    CString::new(s).unwrap_or_else(|_| CString::new("badkey").unwrap())
}

// ---------------------------------------------------------------------------
// Fused subgraph: re-quantize every baked weight, serialize the op-DAG to the flat ABI, and
// compile it as one blob (disk-cached in the shim). Keyed in memory by `(output pointer, M)` — the
// output tensor identifies this subgraph instance, and its addresses are stable across tokens.
// ---------------------------------------------------------------------------
static SG_CACHE: Cache<(usize, i64), Option<&'static [u8]>> = Cache::new();

pub(super) fn get_or_compile_subgraph(sg: &Subgraph) -> Option<&'static [u8]> {
    if sg.m <= 0 || sg.inputs.is_empty() || sg.nodes.is_empty() || sg.output as usize >= sg.nodes.len() {
        return None;
    }
    SG_CACHE.get_or_insert_with((sg.out.ptr as usize, sg.m), || {
        compile_subgraph(sg).map(|v| &*Vec::leak(v))
    })
}

fn compile_subgraph(sg: &Subgraph) -> Option<Vec<u8>> {
    let s = shim()?;
    let compile = s.compile_subgraph?;
    let bits = forced_bits();

    // Re-quantize every baked weight (bytes must outlive the shim call → held in `requants`).
    let mut requants: Vec<Requant> = Vec::with_capacity(sg.consts.len());
    for c in &sg.consts {
        requants.push(weights::requant(&c.weight, c.k as usize, c.n as usize, bits)?);
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
                SgOp::Silu => (3, 0),
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

    let input_k: Vec<i32> = sg.inputs.iter().map(|op| (op.elems / sg.m as usize) as i32).collect();
    let key = subgraph_cache_key(sg, &requants, &nodes, &input_k);

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
            sg.output as i32,
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

fn subgraph_cache_key(sg: &Subgraph, requants: &[Requant], nodes: &[SgNodeAbi], input_k: &[i32]) -> CString {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    h = fnv1a(h, &FORMAT_VERSION.to_le_bytes());
    h = fnv1a(h, &sg.m.to_le_bytes());
    for k in input_k {
        h = fnv1a(h, &k.to_le_bytes());
    }
    for nd in nodes {
        for v in [nd.op, nd.a, nd.b, nd.const_idx, nd.out_n] {
            h = fnv1a(h, &v.to_le_bytes());
        }
    }
    for rq in requants {
        h = fnv1a(h, &[rq.bits as u8, rq.group as u8]);
        h = fnv1a(h, &rq.weight);
        h = fnv1a(h, &rq.scales);
    }
    let s = alloc::format!("sg-{h:016x}-m{}-n{}", sg.m, sg.nodes.len());
    CString::new(s).unwrap_or_else(|_| CString::new("badkey").unwrap())
}
