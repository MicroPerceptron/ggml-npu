//! Product sched auto-policy (`GGML_NPU_SCHED=auto`).
//!
//! llama.cpp owns the master env and explicit `GGML_SCHED_*` overrides. This
//! module:
//! 1. Reports Direct eligibility so the host can enable the prefill duty-cycle
//!    trio (coalesce + cut-cost + claim-stage) when an NPU backend is present.
//! 2. Classifies decode/M=1 tensors so `offload_op` prefers unclaim under auto
//!    (Vulkan/host gen) — Direct decode is a dispatch floor today.
//!
//! Level Zero / non-Direct builds return 0 from eligibility so `auto` does not
//! change their default placement. Explicit `GGML_SCHED_*` always win in the
//! host for claim/cut/coalesce; `GGML_NPU_FORCE_GRAPH` still wins in policy.

use core::ffi::{c_char, c_int, c_void, CStr};

use crate::backend::direct_narrow_buft_contract;
use crate::ffi::ggml_tensor;
use crate::host::{ConfigService, HostEnvironment};

const PROC_NAME: &[u8] = b"ggml_backend_npu_sched_auto_eligible";

/// Exact `GGML_NPU_SCHED=auto` (product prefill trio + decode unclaim).
pub(crate) fn sched_auto_enabled() -> bool {
    crate::host::builtin()
        .config()
        .is_exact(c"GGML_NPU_SCHED", b"auto")
}

/// Decode / token-gen batch: output rows `ne[1] <= 1` on a plain 2D tensor.
/// Matches llama.cpp `ggml_backend_sched_graph_is_decode` (MUL_MAT `ne[1] <= 1`).
/// Prefill smokes (pp16+) keep `ne[1] >= 16` and stay on the auto trio path.
pub(crate) unsafe fn is_decode_batch_tensor(t: *const ggml_tensor) -> bool {
    if t.is_null() {
        return false;
    }
    let n = &*t;
    n.ne[1] <= 1 && n.ne[2] == 1 && n.ne[3] == 1
}

/// Under `GGML_NPU_SCHED=auto`, refuse profitable NPU pull for decode-batch
/// compute so ggml places gen on Vulkan/BLAS/CPU. Capability (`supports_op`) is
/// unchanged — this is profitability / placement only (#5/#8).
pub(crate) unsafe fn auto_decode_prefer_unclaim(node: *const ggml_tensor) -> bool {
    sched_auto_enabled() && is_decode_batch_tensor(node)
}

/// C ABI: `1` when Direct narrow `supports_buft` is in force; else `0`.
#[no_mangle]
pub unsafe extern "C" fn ggml_backend_npu_sched_auto_eligible() -> c_int {
    if direct_narrow_buft_contract() {
        1
    } else {
        0
    }
}

pub(crate) unsafe fn proc_address(name: *const c_char) -> *mut c_void {
    if name.is_null() {
        return core::ptr::null_mut();
    }
    let bytes = unsafe { CStr::from_ptr(name) }.to_bytes();
    if bytes == PROC_NAME {
        ggml_backend_npu_sched_auto_eligible as *mut c_void
    } else {
        core::ptr::null_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::*;
    use core::ptr;

    #[test]
    fn direct_default_build_is_auto_eligible() {
        // Default crate features are `direct-ivpu` without `level-zero-compat`.
        assert_eq!(unsafe { ggml_backend_npu_sched_auto_eligible() }, 1);
        assert!(direct_narrow_buft_contract());
    }

    #[test]
    fn proc_address_resolves_eligible_symbol() {
        let name = c"ggml_backend_npu_sched_auto_eligible";
        let p = unsafe { proc_address(name.as_ptr()) };
        assert!(!p.is_null());
        let f: unsafe extern "C" fn() -> c_int =
            unsafe { core::mem::transmute(p) };
        assert_eq!(unsafe { f() }, 1);
    }

    #[test]
    fn decode_batch_is_ne1_le_one_plain_2d() {
        let mut t: ggml_tensor = unsafe { core::mem::zeroed() };
        t.ne = [2048, 1, 1, 1];
        assert!(unsafe { is_decode_batch_tensor(&t) });
        t.ne[1] = 16;
        assert!(!unsafe { is_decode_batch_tensor(&t) });
        t.ne[1] = 1;
        t.ne[2] = 2;
        assert!(!unsafe { is_decode_batch_tensor(&t) });
        assert!(!unsafe { is_decode_batch_tensor(ptr::null()) });
    }

    #[test]
    fn auto_decode_unclaim_requires_exact_auto_env() {
        // Without GGML_NPU_SCHED=auto in the test process, prefer-unclaim stays off.
        let mut t: ggml_tensor = unsafe { core::mem::zeroed() };
        t.ne = [2048, 1, 1, 1];
        assert!(!unsafe { auto_decode_prefer_unclaim(&t) });
    }
}
