//! Vendor-agnostic weight requantization. Prefer **source-native** adapters in `ggml_quant` when
//! available. This module is the fallback: decode a ggml weight and re-quantize it to a **grouped
//! low-bit** form a vendor bakes or binds as a compressed constant.
//!
//! # Intermediate width policy
//!
//! The NPU interior baseline is nightly **`f16`**. Dequant / requant scratch must **not** default
//! to f32 unless an API or format has no f16 path:
//! - Working row buffers after decode are nightly `f16`.
//! - ggml type-traits `to_float` is currently **f32-only ABI** — that is the explicit exception:
//!   decode into a transient f32 row, then narrow immediately to the f16 working buffer. Do not
//!   keep f32 as the primary intermediate. Never use the `half` crate for this path.
//! - An f32 *source* tensor is consumed in place (it is already f32; that is not a dequant default).
//!
//! Two forms, selected by `asymmetric`:
//! - **asymmetric unsigned** (`u4`/`u8` + per-group **zero-point** + f16 scale) — the *canonical*
//!   compressed-weights shape that NPU compilers pattern-match and keep compressed (dequant fused
//!   in-kernel: `(q - zero) * scale`), so the weight never expands to f16 in device memory. This is
//!   the form to prefer; it is both smaller (stays low-bit) and faster (less weight bandwidth).
//! - **symmetric signed** (`i4`/`i8` + f16 scale, no zero-point) — the simpler form; a fallback for
//!   vendors/compilers that don't need the canonical shape.
//!
//! `bits == 16` emits an f16 constant (no scales/zeros) for an f32 source. This is fallback
//! narrowing (via `core::simd` under the default `portable-simd` feature), not native f16 source
//! pass-through. The weight is emitted **unpacked** (one value per byte) so we never depend on a
//! compiler's sub-byte packing order; the consumer narrows it (e.g. OpenVINO
//! `Constant::create(u4, …)` packs). The output [`Requant`] is transient — the vendor copies the
//! bytes during compilation, so nothing here needs to outlive it.
//!
//! The `bits`/`asymmetric` *policy* is the vendor's (source width, accuracy, what its compiler
//! fuses); only the *algorithm* is here. The per-group length is picked adaptively — the largest of
//! 128/64/32 dividing K (128 is fastest; every ggml quant block size is a multiple of 32).

use alloc::vec::Vec;

use crate::ffi::{ggml_get_type_traits, GGML_TYPE_F32};
use crate::host::{HostEnvironment, MathService};
use crate::ops::Operand;

/// A weight re-quantized to a grouped low-bit form. `weight` is `[N, K]` row-major, one value per
/// byte; `zeros`/`scales` are `[N, K/group]` row-major (zeros unpacked; scales little-endian f16).
/// `zeros` is empty for the symmetric and f16 forms.
#[derive(Debug, PartialEq, Eq)]
pub struct Requant {
    pub weight: Vec<u8>,
    pub zeros: Vec<u8>,
    pub scales: Vec<u8>,
    pub bits: i32,
    pub group: i32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequantStats {
    /// Maximum temporary allocation used to materialize source values, excluding output buffers.
    pub peak_scratch_bytes: usize,
    pub rows_decoded: usize,
}

/// Largest supported per-group length dividing `k` (128 → fastest, fewest per-group ops).
fn pick_group(k: usize) -> Option<usize> {
    [128usize, 64, 32]
        .into_iter()
        .find(|&g| k.is_multiple_of(g))
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

/// Little-endian f16 bytes of `x`, appended to `out`. Uses nightly `f16`, never `half`.
fn push_f16(out: &mut Vec<u8>, x: f32) {
    let b = (x as f16).to_bits();
    out.push((b & 0xff) as u8);
    out.push((b >> 8) as u8);
}

/// Baseline lane width for requant float hot loops. `core::simd` lowers to scalar where needed.
#[cfg(feature = "portable-simd")]
const REQUANT_LANES: usize = 8;

#[cfg(feature = "portable-simd")]
fn narrow_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    use core::simd::prelude::*;
    debug_assert_eq!(src.len(), dst.len());
    let mut i = 0;
    while i + REQUANT_LANES <= src.len() {
        let v = Simd::<f32, REQUANT_LANES>::from_slice(&src[i..]);
        let narrowed: Simd<f16, REQUANT_LANES> = v.cast();
        narrowed.copy_to_slice(&mut dst[i..]);
        i += REQUANT_LANES;
    }
    for (dst, &src) in dst[i..].iter_mut().zip(src[i..].iter()) {
        *dst = src as f16;
    }
}

#[cfg(not(feature = "portable-simd"))]
fn narrow_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    debug_assert_eq!(src.len(), dst.len());
    for (dst, &src) in dst.iter_mut().zip(src.iter()) {
        *dst = src as f16;
    }
}

#[cfg(feature = "portable-simd")]
fn widen_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    use core::simd::prelude::*;
    debug_assert_eq!(src.len(), dst.len());
    let mut i = 0;
    while i + REQUANT_LANES <= src.len() {
        let v = Simd::<f16, REQUANT_LANES>::from_slice(&src[i..]);
        let widened: Simd<f32, REQUANT_LANES> = v.cast();
        widened.copy_to_slice(&mut dst[i..]);
        i += REQUANT_LANES;
    }
    for (dst, &src) in dst[i..].iter_mut().zip(src[i..].iter()) {
        *dst = src as f32;
    }
}

#[cfg(not(feature = "portable-simd"))]
fn widen_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    for (dst, &src) in dst.iter_mut().zip(src.iter()) {
        *dst = src as f32;
    }
}

#[cfg(feature = "portable-simd")]
fn push_f16_row(out: &mut Vec<u8>, values: &[f32]) {
    use core::simd::prelude::*;
    let mut i = 0;
    while i + REQUANT_LANES <= values.len() {
        let v = Simd::<f32, REQUANT_LANES>::from_slice(&values[i..]);
        let narrowed: Simd<f16, REQUANT_LANES> = v.cast();
        for bits in narrowed.to_bits().to_array() {
            out.push((bits & 0xff) as u8);
            out.push((bits >> 8) as u8);
        }
        i += REQUANT_LANES;
    }
    for &value in &values[i..] {
        push_f16(out, value);
    }
}

#[cfg(not(feature = "portable-simd"))]
fn push_f16_row(out: &mut Vec<u8>, values: &[f32]) {
    for &value in values {
        push_f16(out, value);
    }
}

#[cfg(feature = "portable-simd")]
fn push_f16_bits_row(out: &mut Vec<u8>, values: &[f16]) {
    use core::simd::prelude::*;
    let mut i = 0;
    while i + REQUANT_LANES <= values.len() {
        let v = Simd::<f16, REQUANT_LANES>::from_slice(&values[i..]);
        for bits in v.to_bits().to_array() {
            out.push((bits & 0xff) as u8);
            out.push((bits >> 8) as u8);
        }
        i += REQUANT_LANES;
    }
    for &value in &values[i..] {
        let bits = value.to_bits();
        out.push((bits & 0xff) as u8);
        out.push((bits >> 8) as u8);
    }
}

#[cfg(not(feature = "portable-simd"))]
fn push_f16_bits_row(out: &mut Vec<u8>, values: &[f16]) {
    for &value in values {
        let bits = value.to_bits();
        out.push((bits & 0xff) as u8);
        out.push((bits >> 8) as u8);
    }
}

#[cfg(feature = "portable-simd")]
fn group_min_max(values: &[f32]) -> (f32, f32) {
    use core::simd::prelude::*;
    debug_assert!(!values.is_empty());
    let mut i = 0;
    let mut lo = Simd::<f32, REQUANT_LANES>::splat(values[0]);
    let mut hi = Simd::<f32, REQUANT_LANES>::splat(values[0]);
    while i + REQUANT_LANES <= values.len() {
        let v = Simd::<f32, REQUANT_LANES>::from_slice(&values[i..]);
        lo = lo.simd_min(v);
        hi = hi.simd_max(v);
        i += REQUANT_LANES;
    }
    let mut lo_s = lo.reduce_min();
    let mut hi_s = hi.reduce_max();
    for &value in &values[i..] {
        lo_s = lo_s.min(value);
        hi_s = hi_s.max(value);
    }
    (lo_s, hi_s)
}

#[cfg(not(feature = "portable-simd"))]
fn group_min_max(values: &[f32]) -> (f32, f32) {
    let mut lo = values[0];
    let mut hi = values[0];
    for &value in &values[1..] {
        if value < lo {
            lo = value;
        }
        if value > hi {
            hi = value;
        }
    }
    (lo, hi)
}

#[cfg(feature = "portable-simd")]
fn group_abs_max(values: &[f32]) -> f32 {
    use core::simd::prelude::*;
    debug_assert!(!values.is_empty());
    let mut i = 0;
    let mut maximum = Simd::<f32, REQUANT_LANES>::splat(0.0);
    while i + REQUANT_LANES <= values.len() {
        let v = Simd::<f32, REQUANT_LANES>::from_slice(&values[i..]).abs();
        maximum = maximum.simd_max(v);
        i += REQUANT_LANES;
    }
    let mut maximum_s = maximum.reduce_max();
    for &value in &values[i..] {
        maximum_s = maximum_s.max(fabs(value));
    }
    maximum_s
}

#[cfg(not(feature = "portable-simd"))]
fn group_abs_max(values: &[f32]) -> f32 {
    let mut maximum = 0.0f32;
    for &value in values {
        maximum = maximum.max(fabs(value));
    }
    maximum
}

#[cfg(feature = "portable-simd")]
fn store_asymmetric_quants(values: &[f32], inv: f32, zero: i32, qmax: i32, out: &mut [u8]) {
    use core::simd::prelude::*;
    debug_assert_eq!(values.len(), out.len());
    let inv_v = Simd::<f32, REQUANT_LANES>::splat(inv);
    let mut i = 0;
    while i + REQUANT_LANES <= values.len() {
        let scaled = Simd::<f32, REQUANT_LANES>::from_slice(&values[i..]) * inv_v;
        for (lane, &scaled) in scaled.to_array().iter().enumerate() {
            out[i + lane] = (round_i(scaled) + zero).clamp(0, qmax) as u8;
        }
        i += REQUANT_LANES;
    }
    for (dst, &value) in out[i..].iter_mut().zip(values[i..].iter()) {
        *dst = (round_i(value * inv) + zero).clamp(0, qmax) as u8;
    }
}

#[cfg(not(feature = "portable-simd"))]
fn store_asymmetric_quants(values: &[f32], inv: f32, zero: i32, qmax: i32, out: &mut [u8]) {
    debug_assert_eq!(values.len(), out.len());
    for (dst, &value) in out.iter_mut().zip(values.iter()) {
        *dst = (round_i(value * inv) + zero).clamp(0, qmax) as u8;
    }
}

#[cfg(feature = "portable-simd")]
fn store_symmetric_quants(values: &[f32], inverse: f32, limit: i32, out: &mut [u8]) {
    use core::simd::prelude::*;
    debug_assert_eq!(values.len(), out.len());
    let inv_v = Simd::<f32, REQUANT_LANES>::splat(inverse);
    let mut i = 0;
    while i + REQUANT_LANES <= values.len() {
        let scaled = Simd::<f32, REQUANT_LANES>::from_slice(&values[i..]) * inv_v;
        for (lane, &scaled) in scaled.to_array().iter().enumerate() {
            out[i + lane] = round_i(scaled).clamp(-limit, limit) as i8 as u8;
        }
        i += REQUANT_LANES;
    }
    for (dst, &value) in out[i..].iter_mut().zip(values[i..].iter()) {
        *dst = round_i(value * inverse).clamp(-limit, limit) as i8 as u8;
    }
}

#[cfg(not(feature = "portable-simd"))]
fn store_symmetric_quants(values: &[f32], inverse: f32, limit: i32, out: &mut [u8]) {
    debug_assert_eq!(values.len(), out.len());
    for (dst, &value) in out.iter_mut().zip(values.iter()) {
        *dst = round_i(value * inverse).clamp(-limit, limit) as i8 as u8;
    }
}

/// Re-quantize `weight` (`[N, K]` row-major) to `bits`-wide grouped form. `asymmetric` → unsigned
/// `q` + per-group zero-point (canonical, compressible); else symmetric signed. `bits == 16` → f16.
///
/// Quantized sources are decoded one row at a time. ggml `to_float` fills a transient f32 row
/// (ABI exception), which is immediately narrowed to an f16 working buffer — scratch is therefore
/// `K * (sizeof(f32) + sizeof(f16))` for quantized rows, not a persistent f32 intermediate.
/// F32 source rows are consumed in place and need no value scratch.
pub fn requant(
    weight: &Operand,
    k: usize,
    n: usize,
    bits: i32,
    asymmetric: bool,
) -> Option<Requant> {
    requant_with_stats(weight, k, n, bits, asymmetric).map(|(result, _)| result)
}

pub fn requant_with_stats(
    weight: &Operand,
    k: usize,
    n: usize,
    bits: i32,
    asymmetric: bool,
) -> Option<(Requant, RequantStats)> {
    if k == 0 || n == 0 || weight.ptr.is_null() {
        return None;
    }
    let elems = k.checked_mul(n)?;
    if weight.elems != elems {
        return None;
    }

    let traits = if weight.dtype == GGML_TYPE_F32 {
        None
    } else {
        let pointer = unsafe { ggml_get_type_traits(weight.dtype) };
        if pointer.is_null() {
            return None;
        }
        let traits = unsafe { &*pointer };
        if !traits.is_quantized || traits.blck_size <= 0 || traits.type_size == 0 {
            return None;
        }
        if !k.is_multiple_of(traits.blck_size as usize) {
            return None;
        }
        Some(traits)
    };

    let group = if bits == 16 { 0 } else { pick_group(k)? };
    let kg = k.checked_div(group).unwrap_or(0);
    let mut result = Requant {
        weight: if bits == 16 {
            Vec::with_capacity(elems.checked_mul(2)?)
        } else {
            alloc::vec![0; elems]
        },
        zeros: if bits != 16 && asymmetric {
            alloc::vec![0; n.checked_mul(kg)?]
        } else {
            Vec::new()
        },
        scales: if bits == 16 {
            Vec::new()
        } else {
            Vec::with_capacity(n.checked_mul(kg)?.checked_mul(2)?)
        },
        bits,
        group: group as i32,
    };
    // ABI exception buffer (f32) + policy working buffer (nightly f16).
    let mut abi_f32 = if traits.is_some() {
        alloc::vec![0.0f32; k]
    } else {
        Vec::new()
    };
    let mut row_f16 = if traits.is_some() {
        alloc::vec![0.0f16; k]
    } else {
        Vec::new()
    };
    let stats = RequantStats {
        peak_scratch_bytes: abi_f32.len() * core::mem::size_of::<f32>()
            + row_f16.len() * core::mem::size_of::<f16>(),
        rows_decoded: n,
    };

    for row in 0..n {
        if let Some(traits) = traits {
            let block_size = traits.blck_size as usize;
            let row_bytes = (k / block_size).checked_mul(traits.type_size)?;
            let source = unsafe { (weight.ptr as *const u8).add(row.checked_mul(row_bytes)?) };
            // ggml to_float is f32-only; narrow to f16 working row immediately.
            unsafe { (traits.to_float?)(source.cast(), abi_f32.as_mut_ptr(), k as i64) };
            narrow_f32_to_f16(&abi_f32, &mut row_f16);
            quantize_row_f16(&row_f16, row, k, group, bits, asymmetric, &mut result)?;
        } else {
            let values =
                unsafe { core::slice::from_raw_parts((weight.ptr as *const f32).add(row * k), k) };
            quantize_row_f32(values, row, k, group, bits, asymmetric, &mut result)?;
        }
    }

    Some((result, stats))
}

fn quantize_row_f16(
    values: &[f16],
    row: usize,
    k: usize,
    group: usize,
    bits: i32,
    asymmetric: bool,
    result: &mut Requant,
) -> Option<()> {
    if bits == 16 {
        push_f16_bits_row(&mut result.weight, values);
        return Some(());
    }
    // Widen elementwise for the established scale/round algorithm; the row storage is f16.
    let mut widened = alloc::vec![0.0f32; values.len()];
    widen_f16_to_f32(values, &mut widened);
    quantize_row_f32(&widened, row, k, group, bits, asymmetric, result)
}

fn quantize_row_f32(
    values: &[f32],
    row: usize,
    k: usize,
    group: usize,
    bits: i32,
    asymmetric: bool,
    result: &mut Requant,
) -> Option<()> {
    if bits == 16 {
        push_f16_row(&mut result.weight, values);
        return Some(());
    }
    if !matches!(bits, 4 | 8) || values.len() != k {
        return None;
    }
    let kg = k / group;
    if asymmetric {
        // Keep the established canonical algorithm byte-for-byte stable for f32 sources.
        let qmax = ((1i32 << bits) - 1) as f32;
        for g in 0..kg {
            let input_base = g * group;
            let output_base = row * k + input_base;
            let group_values = &values[input_base..input_base + group];
            let (mut lo, mut hi) = group_min_max(group_values);
            if lo > 0.0 {
                lo = 0.0;
            }
            if hi < 0.0 {
                hi = 0.0;
            }
            let range = hi - lo;
            let scale = if range > 1e-12 { range / qmax } else { 1.0 };
            let inv = 1.0 / scale;
            let zero = round_i(-lo * inv).clamp(0, qmax as i32);
            store_asymmetric_quants(
                group_values,
                inv,
                zero,
                qmax as i32,
                &mut result.weight[output_base..output_base + group],
            );
            result.zeros[row * kg + g] = zero as u8;
            push_f16(&mut result.scales, scale);
        }
    } else {
        let limit = ((1i32 << (bits - 1)) - 1).max(1);
        let qmax = limit as f32;
        for g in 0..kg {
            let input_base = g * group;
            let output_base = row * k + input_base;
            let group_values = &values[input_base..input_base + group];
            let maximum = group_abs_max(group_values);
            let scale = maximum / qmax;
            let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            store_symmetric_quants(
                group_values,
                inverse,
                limit,
                &mut result.weight[output_base..output_base + group],
            );
            push_f16(&mut result.scales, scale);
        }
    }
    Some(())
}

/// Round half away from zero to `i32` through the selected host math service.
fn round_i(v: f32) -> i32 {
    crate::host::builtin().math().round_f32(v) as i32
}

fn fabs(x: f32) -> f32 {
    x.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::ffi::c_void;

    // Unit tests exercise the F32 source branch. Supplying the otherwise host-resolved symbol keeps
    // the no_std cdylib test binary self-contained without pretending to implement quant traits.
    #[export_name = "ggml_get_type_traits"]
    extern "C" fn test_ggml_get_type_traits(
        _type_: crate::ffi::ggml_type,
    ) -> *const crate::ffi::ggml_type_traits {
        core::ptr::null()
    }

    fn operand(values: &mut [f32]) -> Operand {
        Operand {
            ptr: values.as_mut_ptr().cast::<c_void>(),
            elems: values.len(),
            dtype: GGML_TYPE_F32,
            owned: false,
            shared: None,
}
    }

    fn read_f16(bytes: &[u8]) -> f32 {
        f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])) as f32
    }

    #[test]
    fn canonical_u4_f32_golden_is_stable() {
        let mut values: Vec<f32> = (-16..16).map(|value| value as f32).collect();
        values.extend(core::iter::repeat_n(3.0, 32));
        let (actual, stats) = requant_with_stats(&operand(&mut values), 32, 2, 4, true).unwrap();

        let mut expected = vec![
            0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
            13, 14, 14, 15, 15,
        ];
        expected.extend(core::iter::repeat_n(15, 32));
        assert_eq!(actual.weight, expected);
        assert_eq!(actual.zeros, [8, 0]);
        assert_eq!(actual.scales, [0x22, 0x40, 0x66, 0x32]);
        assert_eq!(actual.bits, 4);
        assert_eq!(actual.group, 32);
        assert_eq!(stats.peak_scratch_bytes, 0);
        assert_eq!(stats.rows_decoded, 2);

        let scale = read_f16(&actual.scales[0..2]);
        for (index, &source) in values[..32].iter().enumerate() {
            let restored = (actual.weight[index] as f32 - actual.zeros[0] as f32) * scale;
            assert!(
                (restored - source).abs() <= scale * 0.5 + 0.002,
                "index={index} source={source} restored={restored} scale={scale}"
            );
        }
    }

    #[test]
    fn streamed_rows_match_the_full_tensor_reference() {
        let k = 128;
        let n = 3;
        let mut values: Vec<f32> = (0..k * n)
            .map(|index| {
                let centered = (index as i32 % 41) - 20;
                centered as f32 * 0.1375 + (index / k) as f32 * 0.01
            })
            .collect();

        for &(bits, asymmetric) in &[(4, true), (8, true), (4, false), (8, false), (16, true)] {
            let expected = legacy_full_tensor_reference(&values, k, n, bits, asymmetric);
            let (actual, stats) =
                requant_with_stats(&operand(&mut values), k, n, bits, asymmetric).unwrap();
            assert_eq!(actual, expected, "bits={bits} asymmetric={asymmetric}");
            assert_eq!(stats.peak_scratch_bytes, 0);
        }
    }

    /// Frozen copy of the pre-streaming full-tensor algorithm. It intentionally does not call
    /// `quantize_row`, so row/block refactors are checked against the established byte stream.
    fn legacy_full_tensor_reference(
        values: &[f32],
        k: usize,
        n: usize,
        bits: i32,
        asymmetric: bool,
    ) -> Requant {
        if bits == 16 {
            let mut weight = Vec::with_capacity(values.len() * 2);
            for &value in values {
                push_f16(&mut weight, value);
            }
            return Requant {
                weight,
                zeros: Vec::new(),
                scales: Vec::new(),
                bits,
                group: 0,
            };
        }
        let group = pick_group(k).unwrap();
        let groups = k / group;
        let mut result = Requant {
            weight: vec![0; k * n],
            zeros: if asymmetric {
                vec![0; n * groups]
            } else {
                Vec::new()
            },
            scales: Vec::with_capacity(n * groups * 2),
            bits,
            group: group as i32,
        };
        if asymmetric {
            let maximum = ((1i32 << bits) - 1) as f32;
            for row in 0..n {
                for group_index in 0..groups {
                    let base = row * k + group_index * group;
                    let mut low = values[base];
                    let mut high = values[base];
                    for &value in &values[base + 1..base + group] {
                        low = low.min(value);
                        high = high.max(value);
                    }
                    low = low.min(0.0);
                    high = high.max(0.0);
                    let range = high - low;
                    let scale = if range > 1e-12 { range / maximum } else { 1.0 };
                    let inverse = 1.0 / scale;
                    let zero = round_i(-low * inverse).clamp(0, maximum as i32);
                    for index in 0..group {
                        result.weight[base + index] =
                            (round_i(values[base + index] * inverse) + zero)
                                .clamp(0, maximum as i32) as u8;
                    }
                    result.zeros[row * groups + group_index] = zero as u8;
                    push_f16(&mut result.scales, scale);
                }
            }
        } else {
            let limit = ((1i32 << (bits - 1)) - 1).max(1);
            for row in 0..n {
                for group_index in 0..groups {
                    let base = row * k + group_index * group;
                    let maximum = values[base..base + group]
                        .iter()
                        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
                    let scale = maximum / limit as f32;
                    let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    for index in 0..group {
                        result.weight[base + index] = round_i(values[base + index] * inverse)
                            .clamp(-limit, limit)
                            as i8 as u8;
                    }
                    push_f16(&mut result.scales, scale);
                }
            }
        }
        result
    }
}
