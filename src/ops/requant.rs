//! Vendor-agnostic weight requantization. Dequantize a ggml weight (any type ggml can `to_float`,
//! or f32) and re-quantize it to a **grouped low-bit** form a vendor bakes or binds as a compressed
//! constant.
//!
//! Two forms, selected by `asymmetric`:
//! - **asymmetric unsigned** (`u4`/`u8` + per-group **zero-point** + f16 scale) — the *canonical*
//!   compressed-weights shape that NPU compilers pattern-match and keep compressed (dequant fused
//!   in-kernel: `(q - zero) * scale`), so the weight never expands to f16 in device memory. This is
//!   the form to prefer; it is both smaller (stays low-bit) and faster (less weight bandwidth).
//! - **symmetric signed** (`i4`/`i8` + f16 scale, no zero-point) — the simpler form; a fallback for
//!   vendors/compilers that don't need the canonical shape.
//!
//! `bits == 16` emits an f16 constant (no scales/zeros) for an f32 source. The weight is emitted
//! **unpacked** (one value per byte) so we never depend on a compiler's sub-byte packing order; the
//! consumer narrows it (e.g. OpenVINO `Constant::create(u4, …)` packs). The output [`Requant`] is
//! transient — the vendor copies the bytes during compilation, so nothing here needs to outlive it.
//!
//! The `bits`/`asymmetric` *policy* is the vendor's (source width, accuracy, what its compiler
//! fuses); only the *algorithm* is here. The per-group length is picked adaptively — the largest of
//! 128/64/32 dividing K (128 is fastest; every ggml quant block size is a multiple of 32).

use alloc::vec;
use alloc::vec::Vec;

use crate::ffi::{ggml_get_type_traits, GGML_TYPE_F32};
use crate::ops::Operand;

/// A weight re-quantized to a grouped low-bit form. `weight` is `[N, K]` row-major, one value per
/// byte; `zeros`/`scales` are `[N, K/group]` row-major (zeros unpacked; scales little-endian f16).
/// `zeros` is empty for the symmetric and f16 forms.
pub struct Requant {
    pub weight: Vec<u8>,
    pub zeros: Vec<u8>,
    pub scales: Vec<u8>,
    pub bits: i32,
    pub group: i32,
}

/// Largest supported per-group length dividing `k` (128 → fastest, fewest per-group ops).
fn pick_group(k: usize) -> Option<usize> {
    [128usize, 64, 32].into_iter().find(|&g| k % g == 0)
}

/// Cheap feasibility check (no dequant, no allocation): can this weight be re-quantized at width
/// `k`? A quantized type ggml can `to_float`, or f32, with `k` groupable (a multiple of 32). Lets a
/// vendor claim a matmul without compiling its blob.
pub fn can_requant(weight: &Operand, k: usize) -> bool {
    if weight.ptr.is_null() || pick_group(k).is_none() {
        return false;
    }
    let traits = unsafe { ggml_get_type_traits(weight.dtype) };
    if traits.is_null() {
        return false;
    }
    let tr = unsafe { &*traits };
    (tr.is_quantized && tr.to_float.is_some()) || weight.dtype == GGML_TYPE_F32
}

/// Little-endian f16 bytes of `x`, appended to `out`.
fn push_f16(out: &mut Vec<u8>, x: f32) {
    let b = (x as f16).to_bits();
    out.push((b & 0xff) as u8);
    out.push((b >> 8) as u8);
}

/// Materialize `weight` (`elems = k*n`, logical `[N, K]`) as f32: dequantize a quantized source via
/// ggml's `to_float`, or read an f32 source directly. `None` for any other source type.
unsafe fn to_f32(weight: &Operand, elems: usize) -> Option<Vec<f32>> {
    let tr = &*{
        let p = ggml_get_type_traits(weight.dtype);
        if p.is_null() {
            return None;
        }
        p
    };
    let mut f: Vec<f32> = vec![0.0; elems];
    if tr.is_quantized {
        (tr.to_float?)(weight.ptr, f.as_mut_ptr(), elems as i64);
    } else if weight.dtype == GGML_TYPE_F32 {
        core::ptr::copy_nonoverlapping(weight.ptr as *const f32, f.as_mut_ptr(), elems);
    } else {
        return None;
    }
    Some(f)
}

/// Re-quantize `weight` (`[N, K]` row-major) to `bits`-wide grouped form. `asymmetric` → unsigned
/// `q` + per-group zero-point (canonical, compressible); else symmetric signed. `bits == 16` → f16.
/// `None` if the source can't be materialized or K isn't groupable (caller → CPU).
pub fn requant(weight: &Operand, k: usize, n: usize, bits: i32, asymmetric: bool) -> Option<Requant> {
    if k == 0 || n == 0 || weight.ptr.is_null() {
        return None;
    }
    let elems = k * n;
    let f = unsafe { to_f32(weight, elems) }?;

    // f16 constant: no per-group scales/zeros; weight is the narrowed f16 bytes.
    if bits == 16 {
        let mut wq: Vec<u8> = Vec::with_capacity(elems * 2);
        for &v in &f {
            push_f16(&mut wq, v);
        }
        return Some(Requant { weight: wq, zeros: Vec::new(), scales: Vec::new(), bits: 16, group: 0 });
    }

    let group = pick_group(k)?;
    let kg = k / group;
    let mut wq: Vec<u8> = vec![0; elems];
    let mut zeros: Vec<u8> = Vec::new();
    let mut scales: Vec<u8> = Vec::with_capacity(n * kg * 2);

    if asymmetric {
        // Asymmetric unsigned per group: scale = (max-min)/qmax, zero = round(-min/scale),
        // q = round(x/scale)+zero. Dequant = (q-zero)*scale. Constant groups map exactly (see below).
        let qmax = ((1i32 << bits) - 1) as f32;
        zeros = vec![0; n * kg];
        for row in 0..n {
            for g in 0..kg {
                let base = row * k + g * group;
                let (mut lo, mut hi) = (f[base], f[base]);
                for j in 1..group {
                    let x = f[base + j];
                    if x < lo {
                        lo = x;
                    }
                    if x > hi {
                        hi = x;
                    }
                }
                // Include 0 in the range so the zero-point (the q that dequants to exactly 0) is
                // representable. Without this, an all-positive or all-negative group forces zp to
                // clamp and the whole group collapses into a tiny q range → garbage.
                if lo > 0.0 {
                    lo = 0.0;
                }
                if hi < 0.0 {
                    hi = 0.0;
                }
                let range = hi - lo;
                let scale = if range > 1e-12 { range / qmax } else { 1.0 };
                let inv = 1.0 / scale;
                let zp = round_i(-lo * inv).clamp(0, qmax as i32);
                for j in 0..group {
                    let q = (round_i(f[base + j] * inv) + zp).clamp(0, qmax as i32);
                    wq[base + j] = q as u8;
                }
                zeros[row * kg + g] = zp as u8;
                push_f16(&mut scales, scale);
            }
        }
    } else {
        // Symmetric signed per group: scale = max|v|/qlim, q = round(v/scale), no zero-point.
        let qlim = ((1i32 << (bits - 1)) - 1).max(1);
        let qmax = qlim as f32;
        for row in 0..n {
            for g in 0..kg {
                let base = row * k + g * group;
                let mut amax = 0.0f32;
                for j in 0..group {
                    let a = fabs(f[base + j]);
                    if a > amax {
                        amax = a;
                    }
                }
                let scale = amax / qmax;
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                for j in 0..group {
                    let q = round_i(f[base + j] * inv).clamp(-qlim, qlim);
                    wq[base + j] = q as i8 as u8;
                }
                push_f16(&mut scales, scale);
            }
        }
    }

    Some(Requant { weight: wq, zeros, scales, bits, group: group as i32 })
}

// The scalar float ops `core` doesn't expose (they may lower to libm) resolve from the host's libm
// at dlopen — the same way `malloc`/`getenv`/`clock_gettime` do elsewhere, so the crate keeps its
// zero runtime Cargo deps. `roundf` is round-half-away-from-zero, matching the quantizer's rounding.
extern "C" {
    fn roundf(x: f32) -> f32;
    fn fabsf(x: f32) -> f32;
}

/// Round half away from zero to `i32` (host libm `roundf`).
fn round_i(v: f32) -> i32 {
    unsafe { roundf(v) as i32 }
}

/// `|x|` (host libm `fabsf`).
fn fabs(x: f32) -> f32 {
    unsafe { fabsf(x) }
}
