//! Product sched auto-policy eligibility (`GGML_NPU_SCHED=auto`).
//!
//! llama.cpp owns the master env and explicit `GGML_SCHED_*` overrides. This
//! symbol tells the thin host hook whether the loaded NPU backend is the Direct
//! narrow-buft build that needs claim-stage + coalesce + cut-cost for L0-class
//! duty cycle. Level Zero / non-Direct builds return 0 so `auto` does not change
//! their default placement.

use core::ffi::{c_char, c_int, c_void, CStr};

use crate::backend::direct_narrow_buft_contract;

const PROC_NAME: &[u8] = b"ggml_backend_npu_sched_auto_eligible";

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
}
