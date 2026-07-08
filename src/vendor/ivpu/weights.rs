//! Weight materialization: dequantize a ggml weight (quantized or f32) into an **f16 shadow** in
//! NPU memory, once, keyed by the weight's data pointer.
//!
//! The device MatMul takes an f16 weight (half the bytes of f32, the NPU's native precision), so
//! each distinct weight is dequantized a single time — via ggml's own `to_float` for a quantized
//! type, or a straight narrowing for f32 — into an NPU-resident f16 buffer that is then bound
//! directly (zero copy) for every token. Weights are constant across tokens, so the shadow never
//! needs refreshing. Shadows are process-lifetime (leaked), like the graph/list caches.

use alloc::vec::Vec;
use core::ffi::c_void;

use super::cache::Cache;
use super::ze;
use crate::ffi::{ggml_get_type_traits, GGML_TYPE_F16, GGML_TYPE_F32};
use crate::ops::Operand;

// key = weight data pointer (addr); value = f16 shadow pointer (addr), or None on failure.
static SHADOWS: Cache<usize, Option<usize>> = Cache::new();

/// The f16 NPU-memory shadow of `weight` (`elems = K*N`), dequantized on first sight and cached
/// by pointer. Returns an owned f16 [`Operand`] the session binds directly (zero copy per token).
pub(super) fn shadow(weight: &Operand, elems: usize) -> Option<Operand> {
    let sp = SHADOWS.get_or_insert_with(weight.ptr as usize, || unsafe { build(weight, elems) })?;
    Some(Operand { ptr: sp as *mut c_void, elems, dtype: GGML_TYPE_F16, owned: true })
}

unsafe fn build(weight: &Operand, elems: usize) -> Option<usize> {
    let shadow = ze::alloc_host(elems * 2); // f16 = 2 bytes/elem
    if shadow.is_null() {
        return None;
    }
    let dst = shadow as *mut f16;
    if weight.dtype == GGML_TYPE_F32 {
        let src = weight.ptr as *const f32;
        for i in 0..elems {
            *dst.add(i) = *src.add(i) as f16;
        }
    } else {
        // Quantized: dequantize to f32 via ggml's type traits, then narrow to f16.
        let traits = ggml_get_type_traits(weight.dtype);
        let to_float = if traits.is_null() { None } else { (*traits).to_float };
        let Some(to_float) = to_float else {
            ze::free_host(shadow);
            return None;
        };
        let mut tmp: Vec<f32> = Vec::new();
        tmp.resize(elems, 0.0);
        to_float(weight.ptr, tmp.as_mut_ptr(), elems as i64);
        for i in 0..elems {
            *dst.add(i) = tmp[i] as f16;
        }
    }
    Some(shadow as usize)
}
