//! Runtime blob compilation + shape-keyed cache for the Intel NPU.
//!
//! The OpenVINO compile shim (`libggml_npu_ovc.so`) is `dlopen`'d on demand and compiles a native
//! NPU blob for a MatMul shape; blobs are cached by `(M, K, N)` so each shape compiles at most
//! once (JIT-on-miss, §7.3). `no_std`: the shim, `dlopen`, and `getenv` resolve from the host. The
//! cache and the shim handle are both lock-free ([`super::cache`] / an `AtomicPtr` once-init).

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_void};
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use super::cache::ShapeCache;

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
static CACHE: ShapeCache<Option<&'static [u8]>> = ShapeCache::new();

fn compile_matmul(m: i64, k: i64, n: i64) -> Option<Vec<u8>> {
    let s = shim()?;
    if m <= 0 || k <= 0 || n <= 0 || m > i32::MAX as i64 || k > i32::MAX as i64 || n > i32::MAX as i64 {
        return None;
    }
    let mut blob: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    let rc = unsafe { (s.compile)(m as i32, k as i32, n as i32, &mut blob, &mut len) };
    if rc != 0 || blob.is_null() || len == 0 {
        return None;
    }
    let out = unsafe { core::slice::from_raw_parts(blob, len) }.to_vec();
    unsafe { (s.free)(blob) };
    Some(out)
}

/// Get the cached native blob for a MatMul of shape `(M, K, N)`, compiling on a miss.
pub(super) fn get_or_compile_matmul(m: i64, k: i64, n: i64) -> Option<&'static [u8]> {
    CACHE.get_or_insert_with((m, k, n), || compile_matmul(m, k, n).map(|v| &*Vec::leak(v)))
}
