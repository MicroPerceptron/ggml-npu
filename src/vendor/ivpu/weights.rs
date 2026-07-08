//! Weight requantization for the **Constant path**: turn a ggml weight (quantized) into the
//! symmetric per-group low-bit form the NPU bakes as a compile-time `Constant`.
//!
//! The device MatMul gets its fused decompress-matmul only when the weight is a compile-time
//! Constant (a runtime low-bit *input* is not fused). So each weight is dequantized to f32 (via
//! ggml's `to_float`), then re-quantized on the host to **symmetric per-group int4 or int8** with
//! **f16 per-group scales** — the bytes the shim bakes as `Constant`s.
//!
//! **int4 vs int8** is chosen to match the source's bit width (overridable via `GGML_NPU_BITS`):
//! a 4-bit GGUF (Q4_*, IQ4_*) re-quants to **int4** — 0.5 byte/elem, so it does not inflate the
//! model — while a ≥5-bit source (Q6_K, Q8_0, …) uses **int8**. The re-quant is a second step on
//! top of the already-quantized source; int8-per-32 is ~lossless, int4-per-32 costs a little more.
//!
//! For int4 the weight is emitted **unpacked** (one signed byte per nibble value, in `[-7, 7]`) —
//! the shim's `Constant::create` packs it into the baked i4 blob, so we never depend on OpenVINO's
//! sub-byte nibble order. Only quantized weights take this path; an f32 weight returns `None` (the
//! op then falls back to CPU). The output [`Requant`] is transient — the shim copies the bytes into
//! the OpenVINO `Constant` during compilation, so nothing here needs to outlive the compile call.

use alloc::vec;
use alloc::vec::Vec;

use crate::ffi::{ggml_get_type_traits, GGML_TYPE_F32};
use crate::ops::Operand;

/// Per-group length along the contraction dim K (matches ggml's common block size). K must be a
/// whole number of groups; `match_node` already guarantees this for the quant block size, and 32
/// divides every ggml quant block size, so a K that is block-aligned is also group-aligned.
pub(super) const GROUP: usize = 32;

/// A weight re-quantized for the Constant path: low-bit weight bytes (`[N, K]` row-major — raw
/// int8 for `bits==8`, unpacked nibble values for `bits==4`), f16 per-group `scales`
/// (`[N, K/GROUP]` row-major, little-endian bytes), and the `bits`/`group` the shim needs to
/// rebuild the dequant subgraph.
pub(super) struct Requant {
    pub weight: Vec<u8>,
    pub scales: Vec<u8>,
    pub bits: i32,
    pub group: i32,
}

/// Pick the re-quant width, unless `force` overrides: a quantized source picks int4 (≤5 bpw, e.g.
/// Q4_*) or int8 (≥5 bpw) from its bits-per-weight; a non-quantized (f32) source defaults to an
/// f16 Constant (`bits==16`, no scales) to preserve the baseline's precision without inflating it.
fn choose_bits(tr: &crate::ffi::ggml_type_traits, force: Option<i32>) -> i32 {
    if let Some(b) = force {
        return b;
    }
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

/// Little-endian f16 bytes of `x`, appended to `out`.
fn push_f16(out: &mut Vec<u8>, x: f32) {
    let b = (x as f16).to_bits();
    out.push((b & 0xff) as u8);
    out.push((b >> 8) as u8);
}

/// Re-quantize `weight` (`elems = k*n`, logical `[N, K]` row-major) for the Constant path: an f32
/// weight → f16 Constant, a quantized weight → symmetric per-group int4/int8 (see [`choose_bits`];
/// `force` = an explicit width). Returns `None` for an unsupported source type or a K that is not
/// group-aligned (caller → CPU).
pub(super) fn requant(weight: &Operand, k: usize, n: usize, force: Option<i32>) -> Option<Requant> {
    if k == 0 || n == 0 || k % GROUP != 0 || weight.ptr.is_null() {
        return None;
    }
    let traits = unsafe { ggml_get_type_traits(weight.dtype) };
    if traits.is_null() {
        return None;
    }
    let tr = unsafe { &*traits };
    let elems = k * n;

    // Materialize the weight as f32 once (row-major [N, K]): dequantize a quantized source via
    // ggml's `to_float`, or read an f32 source directly. Other source types (e.g. f16) are not
    // claimed by `match_node`, so they never reach here.
    let mut f: Vec<f32> = vec![0.0; elems];
    if tr.is_quantized {
        let to_float = tr.to_float?;
        unsafe { to_float(weight.ptr, f.as_mut_ptr(), elems as i64) };
    } else if weight.dtype == GGML_TYPE_F32 {
        unsafe { core::ptr::copy_nonoverlapping(weight.ptr as *const f32, f.as_mut_ptr(), elems) };
    } else {
        return None;
    }

    let bits = choose_bits(tr, force);

    // f16 Constant: no per-group scales, weight is the narrowed f16 bytes.
    if bits == 16 {
        let mut wq: Vec<u8> = Vec::with_capacity(elems * 2);
        for &v in &f {
            push_f16(&mut wq, v);
        }
        return Some(Requant { weight: wq, scales: Vec::new(), bits: 16, group: 0 });
    }

    // Symmetric per-group int4/int8: per (row, group) scale = max|v| / qmax, q = round(v / scale).
    // For int4 the byte holds the signed nibble value directly (the shim packs it via
    // Constant::create).
    let qlim: i32 = if bits == 4 { 7 } else { 127 };
    let qmax = qlim as f32;
    let kg = k / GROUP;
    let mut wq: Vec<u8> = vec![0; elems];
    let mut scales: Vec<u8> = Vec::with_capacity(n * kg * 2);

    for row in 0..n {
        for g in 0..kg {
            let base = row * k + g * GROUP;
            let mut amax = 0.0f32;
            for j in 0..GROUP {
                let x = f[base + j];
                let a = if x < 0.0 { -x } else { x }; // no_std: avoid f32::abs (std-only)
                if a > amax {
                    amax = a;
                }
            }
            let scale = amax / qmax;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for j in 0..GROUP {
                // round half away from zero, then clamp (float error can nudge past the limit).
                let v = f[base + j] * inv;
                let r = if v >= 0.0 { v + 0.5 } else { v - 0.5 };
                let qi = (r as i32).clamp(-qlim, qlim);
                wq[base + j] = qi as i8 as u8;
            }
            push_f16(&mut scales, scale);
        }
    }

    Some(Requant { weight: wq, scales, bits, group: GROUP as i32 })
}
