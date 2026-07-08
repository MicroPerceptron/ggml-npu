//! Runtime blob compilation + shape-keyed cache for the Intel NPU.
//!
//! The OpenVINO compile shim (`libggml_npu_ovc.so`) is `dlopen`'d on demand and compiles a
//! native NPU blob for a MatMul shape; blobs are cached by `(M, K, N)` so each distinct shape
//! is compiled at most once — the JIT-on-miss model (§7.3). `no_std`: the shim, `dlopen`, and
//! `getenv` all resolve from the host. The cache is a `spin::Mutex` over a `hashbrown::HashMap`,
//! held only for short lookups/inserts — the (slow) compile runs with the lock released.

use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_void};
use core::mem;

use hashbrown::HashMap;
use spin::{Mutex, Once};

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn getenv(name: *const c_char) -> *const c_char;
}
const RTLD_NOW: c_int = 2;

type CompileMatmul = unsafe extern "C" fn(i32, i32, i32, *mut *mut u8, *mut usize) -> c_int;
type FreeBlob = unsafe extern "C" fn(*mut u8);

struct Shim {
    compile: CompileMatmul,
    free: FreeBlob,
}

static SHIM: Once<Option<Shim>> = Once::new();

unsafe fn load_shim() -> Option<Shim> {
    // GGML_NPU_OVC overrides the shim path; else the default soname on the loader path.
    let env = getenv(c"GGML_NPU_OVC".as_ptr());
    let h = if env.is_null() {
        dlopen(c"libggml_npu_ovc.so".as_ptr(), RTLD_NOW)
    } else {
        dlopen(env, RTLD_NOW)
    };
    if h.is_null() {
        return None;
    }
    let compile = dlsym(h, c"ggml_npu_ov_compile_matmul".as_ptr());
    let free = dlsym(h, c"ggml_npu_ov_free".as_ptr());
    if compile.is_null() || free.is_null() {
        return None;
    }
    Some(Shim {
        compile: mem::transmute::<*mut c_void, CompileMatmul>(compile),
        free: mem::transmute::<*mut c_void, FreeBlob>(free),
    })
}

fn shim() -> Option<&'static Shim> {
    SHIM.call_once(|| unsafe { load_shim() }).as_ref()
}

/// MatMul shape key `(M, K, N)`.
type Key = (i64, i64, i64);

/// `None` value = compiled and failed (don't retry); `Some` = the leaked, process-lifetime
/// blob. The map itself is lazily created inside the lock.
static CACHE: Mutex<Option<HashMap<Key, Option<&'static [u8]>>>> = Mutex::new(None);

fn compile_matmul(m: i64, k: i64, n: i64) -> Option<Vec<u8>> {
    let s = shim()?;
    if m <= 0
        || k <= 0
        || n <= 0
        || m > i32::MAX as i64
        || k > i32::MAX as i64
        || n > i32::MAX as i64
    {
        return None;
    }
    let mut blob: *mut u8 = core::ptr::null_mut();
    let mut len: usize = 0;
    let rc = unsafe { (s.compile)(m as i32, k as i32, n as i32, &mut blob, &mut len) };
    if rc != 0 || blob.is_null() || len == 0 {
        return None;
    }
    let out = unsafe { core::slice::from_raw_parts(blob, len) }.to_vec();
    unsafe { (s.free)(blob) };
    Some(out)
}

/// Get the cached native blob for a MatMul of shape `(M, K, N)`, compiling on a miss. Returns
/// `None` if there is no shim, or the shape does not compile (cached so it is not retried).
pub(super) fn get_or_compile_matmul(m: i64, k: i64, n: i64) -> Option<&'static [u8]> {
    let key = (m, k, n);
    {
        let mut guard = CACHE.lock();
        if let Some(v) = guard.get_or_insert_with(HashMap::new).get(&key) {
            return *v;
        }
    }
    // Compile with the lock released (seconds); other shapes proceed concurrently.
    let compiled = compile_matmul(m, k, n);

    let mut guard = CACHE.lock();
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(v) = map.get(&key) {
        return *v; // another thread won the race; our `compiled` drops here (freed)
    }
    let stored = compiled.map(|v| &*Vec::leak(v));
    map.insert(key, stored);
    stored
}
