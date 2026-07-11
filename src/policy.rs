//! Conservative heterogeneous placement policy.
//!
//! Pre-placed NPU tensors use the normal capability path. `offload_op` uses this cost floor before
//! pulling work from CPU/GPU memory, avoiding per-op ping-pong for tiny latency-bound kernels.

use core::ffi::{c_char, CStr};

use crate::ffi::*;

extern "C" {
    fn getenv(name: *const c_char) -> *const c_char;
}

fn env_u64(name: &'static CStr, default: u64) -> u64 {
    let p = unsafe { getenv(name.as_ptr()) };
    if p.is_null() {
        return default;
    }
    let bytes = unsafe { CStr::from_ptr(p) }.to_bytes();
    let mut value = 0u64;
    if bytes.is_empty() {
        return default;
    }
    for &b in bytes {
        if !b.is_ascii_digit() {
            return default;
        }
        value = match value.checked_mul(10).and_then(|x| x.checked_add((b - b'0') as u64)) {
            Some(v) => v,
            None => return default,
        };
    }
    value
}

pub(crate) unsafe fn should_pull(node: *const ggml_tensor) -> bool {
    if node.is_null() {
        return false;
    }
    let x = &*node;
    let min_work = env_u64(c"GGML_NPU_MIN_OFFLOAD_WORK", 4_000_000);
    let work = match x.op {
        GGML_OP_MUL_MAT | GGML_OP_MUL_MAT_ID => {
            let a = x.src[0];
            if a.is_null() {
                return false;
            }
            (x.ne[0].max(0) as u64)
                .saturating_mul(x.ne[1].max(0) as u64)
                .saturating_mul((*a).ne[0].max(0) as u64)
        }
        GGML_OP_SSM_CONV => x.ne.iter().fold(1u64, |a, &b| a.saturating_mul(b.max(0) as u64)),
        // Elementwise/state nodes should move only as part of a larger split formed by an
        // expensive anchor; pulling them independently creates synchronization islands.
        _ => return false,
    };
    work >= min_work
}
