//! GGML source-format adapter for compile-time constants.
//!
//! Concrete block layouts stop here. Compiler adapters consume [`ConstantDescriptor`] and never
//! need to know a GGML type name.
//!
//! Prefer source-native pass-through / transcode. When a fallback decode is unavoidable, the
//! working intermediate is nightly **`f16`** (see `requant`); do not default to f32, and never use
//! the `half` crate for IEEE F16. BF16 uses `simsimd::bf16` only.

use alloc::vec;
use alloc::vec::Vec;
use core::ffi::c_void;

use super::constant::{
    ConstantData, ConstantDescriptor, Exactness, NumericKind, OffsetEncoding, Packing,
    ScaleEncoding, Signedness,
};
use super::quant_kernel::{ProductionKernel, QuantKernel};
use super::requant::{self, Requant};
use super::Operand;
#[cfg(test)]
use crate::ffi::GGML_TYPE_F32;
use crate::ffi::{
    GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_I8, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_K,
};

const SUPER_BLOCK: usize = 256;
const Q4_K_BYTES: usize = 144;
const Q6_K_BYTES: usize = 210;
const Q8_0_BLOCK: usize = 32;
const Q8_0_BYTES: usize = 34; // f16 d + i8[32]
const Q8_K_BYTES: usize = 292; // f32 d + i8[256] + i16 bsums[16]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstantUse {
    Dense,
    RoutedGather,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrepareError {
    NullSource,
    Shape,
    SourceLength,
    Unsupported,
}

pub enum PreparedConstant {
    /// Byte-equivalent canonical u4/u8 (or f16 narrowing) used before source-native support.
    Fallback(Requant),
    /// Source integers and metadata, or source scalar bytes, without a dequant/requant cycle.
    SourceNative(ConstantData),
}

pub fn source_native_supported(dtype: crate::ffi::ggml_type, use_: ConstantUse) -> bool {
    use_ == ConstantUse::Dense
        && matches!(
            dtype,
            GGML_TYPE_Q4_K
                | GGML_TYPE_Q6_K
                | GGML_TYPE_Q8_0
                | GGML_TYPE_Q8_K
                | GGML_TYPE_F16
                | GGML_TYPE_BF16
                | GGML_TYPE_I8
        )
}

impl PreparedConstant {
    pub fn descriptor(&self) -> ConstantDescriptor {
        match self {
            Self::Fallback(value) => fallback_descriptor(value),
            Self::SourceNative(value) => value.descriptor,
        }
    }

    pub fn values(&self) -> &[u8] {
        match self {
            Self::Fallback(value) => &value.weight,
            Self::SourceNative(value) => &value.values,
        }
    }

    pub fn scale_base(&self) -> &[u8] {
        match self {
            Self::Fallback(value) => &value.scales,
            Self::SourceNative(value) => &value.scale_base,
        }
    }

    pub fn scale_factor(&self) -> &[u8] {
        match self {
            Self::Fallback(_) => &[],
            Self::SourceNative(value) => &value.scale_factor,
        }
    }

    pub fn offset_base(&self) -> &[u8] {
        match self {
            Self::Fallback(value) => &value.zeros,
            Self::SourceNative(value) => &value.offset_base,
        }
    }

    pub fn offset_factor(&self) -> &[u8] {
        match self {
            Self::Fallback(_) => &[],
            Self::SourceNative(value) => &value.offset_factor,
        }
    }

    pub fn fallback(&self) -> Option<&Requant> {
        match self {
            Self::Fallback(value) => Some(value),
            Self::SourceNative(_) => None,
        }
    }

    pub fn is_source_native(&self) -> bool {
        matches!(self, Self::SourceNative(_))
    }
}

fn fallback_descriptor(value: &Requant) -> ConstantDescriptor {
    if value.bits == 16 {
        return ConstantDescriptor {
            numeric_kind: NumericKind::Float,
            bit_width: 16,
            storage_bits: 16,
            signedness: Signedness::NotApplicable,
            block_elems: 1,
            group_elems: 1,
            packing: Packing::Unpacked,
            scale_encoding: ScaleEncoding::None,
            offset_encoding: OffsetEncoding::None,
            exactness: Exactness::Requantized,
        };
    }
    ConstantDescriptor {
        numeric_kind: NumericKind::Integer,
        bit_width: value.bits,
        storage_bits: 8,
        signedness: if value.zeros.is_empty() {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        },
        block_elems: value.group,
        group_elems: value.group,
        packing: Packing::Unpacked,
        scale_encoding: ScaleEncoding::PerGroupF16,
        offset_encoding: if value.zeros.is_empty() {
            OffsetEncoding::None
        } else {
            OffsetEncoding::ZeroPointU8
        },
        exactness: Exactness::Requantized,
    }
}

/// Cheap source and shape check. `source_native` reflects compiler capability; routed gathers are
/// deliberately tested through the canonical fallback because the Intel path cannot gather packed
/// sub-byte expert banks.
pub fn can_prepare(weight: &Operand, k: usize, use_: ConstantUse, source_native: bool) -> bool {
    if weight.ptr.is_null() || k == 0 || weight.elems == 0 || !weight.elems.is_multiple_of(k) {
        return false;
    }
    match weight.dtype {
        GGML_TYPE_F16 | GGML_TYPE_BF16 => source_native,
        GGML_TYPE_I8 => source_native,
        GGML_TYPE_Q8_0 if source_native && use_ == ConstantUse::Dense => {
            k.is_multiple_of(Q8_0_BLOCK)
        }
        GGML_TYPE_Q4_K | GGML_TYPE_Q6_K | GGML_TYPE_Q8_K
            if source_native && use_ == ConstantUse::Dense =>
        {
            k.is_multiple_of(SUPER_BLOCK)
        }
        _ => requant::can_requant(weight, k),
    }
}

/// Prepare one dense/routed constant. Once the source-native Q4_K/Q6_K branch is selected,
/// malformed source data is rejected; it is never silently dequantized and requantized.
pub fn prepare(
    weight: &Operand,
    k: usize,
    n: usize,
    fallback_bits: i32,
    asymmetric: bool,
    use_: ConstantUse,
    source_native: bool,
) -> Result<PreparedConstant, PrepareError> {
    let elems = k.checked_mul(n).ok_or(PrepareError::Shape)?;
    if weight.ptr.is_null() {
        return Err(PrepareError::NullSource);
    }
    if k == 0 || n == 0 || weight.elems != elems {
        return Err(PrepareError::Shape);
    }

    if source_native && source_native_supported(weight.dtype, use_) {
        match weight.dtype {
            GGML_TYPE_Q4_K => {
                let bytes =
                    unsafe { source_bytes(weight.ptr, expected_blocks(k, n)? * Q4_K_BYTES) };
                return transcode_q4_k(bytes, k, n).map(PreparedConstant::SourceNative);
            }
            GGML_TYPE_Q6_K => {
                let bytes =
                    unsafe { source_bytes(weight.ptr, expected_blocks(k, n)? * Q6_K_BYTES) };
                return transcode_q6_k(bytes, k, n).map(PreparedConstant::SourceNative);
            }
            GGML_TYPE_Q8_0 => {
                let blocks = expected_q8_0_blocks(k, n)?;
                let bytes = unsafe { source_bytes(weight.ptr, blocks * Q8_0_BYTES) };
                return transcode_q8_0(bytes, k, n).map(PreparedConstant::SourceNative);
            }
            GGML_TYPE_Q8_K => {
                let bytes =
                    unsafe { source_bytes(weight.ptr, expected_blocks(k, n)? * Q8_K_BYTES) };
                return transcode_q8_k(bytes, k, n).map(PreparedConstant::SourceNative);
            }
            GGML_TYPE_F16 => {
                let bytes = unsafe {
                    source_bytes(weight.ptr, elems.checked_mul(2).ok_or(PrepareError::Shape)?)
                };
                return Ok(PreparedConstant::SourceNative(pass_through_f16(
                    bytes, elems,
                )?));
            }
            GGML_TYPE_BF16 => {
                let bytes = unsafe {
                    source_bytes(weight.ptr, elems.checked_mul(2).ok_or(PrepareError::Shape)?)
                };
                return Ok(PreparedConstant::SourceNative(pass_through_bf16(
                    bytes, elems,
                )?));
            }
            GGML_TYPE_I8 => {
                let bytes = unsafe { source_bytes(weight.ptr, elems) };
                return Ok(PreparedConstant::SourceNative(pass_through_i8(
                    bytes, elems,
                )?));
            }
            _ => {}
        }
    }

    // Explicit canonical fallback: unsupported source, compiler capability rejection, policy
    // override, or routed gather. Scalar float/int8 sources have no proven legacy conversion.
    if matches!(
        weight.dtype,
        GGML_TYPE_F16 | GGML_TYPE_BF16 | GGML_TYPE_I8
    ) {
        return Err(PrepareError::Unsupported);
    }
    requant::requant(weight, k, n, fallback_bits, asymmetric)
        .map(PreparedConstant::Fallback)
        .ok_or(PrepareError::Unsupported)
}

unsafe fn source_bytes<'a>(pointer: *mut c_void, len: usize) -> &'a [u8] {
    core::slice::from_raw_parts(pointer.cast(), len)
}

fn expected_blocks(k: usize, n: usize) -> Result<usize, PrepareError> {
    if k == 0 || n == 0 || !k.is_multiple_of(SUPER_BLOCK) {
        return Err(PrepareError::Shape);
    }
    (k / SUPER_BLOCK).checked_mul(n).ok_or(PrepareError::Shape)
}

fn expected_q8_0_blocks(k: usize, n: usize) -> Result<usize, PrepareError> {
    if k == 0 || n == 0 || !k.is_multiple_of(Q8_0_BLOCK) {
        return Err(PrepareError::Shape);
    }
    (k / Q8_0_BLOCK).checked_mul(n).ok_or(PrepareError::Shape)
}

fn native_descriptor(
    bits: i32,
    signedness: Signedness,
    group: i32,
    scale: ScaleEncoding,
    offset: OffsetEncoding,
) -> ConstantDescriptor {
    ConstantDescriptor {
        numeric_kind: NumericKind::Integer,
        bit_width: bits,
        storage_bits: bits,
        signedness,
        block_elems: SUPER_BLOCK as i32,
        group_elems: group,
        packing: Packing::DenseLsb,
        scale_encoding: scale,
        offset_encoding: offset,
        exactness: Exactness::SourceNumeric,
    }
}

pub fn pass_through_f16(source: &[u8], elems: usize) -> Result<ConstantData, PrepareError> {
    if source.len() != elems.checked_mul(2).ok_or(PrepareError::Shape)? {
        return Err(PrepareError::SourceLength);
    }
    Ok(ConstantData {
        descriptor: float_pass_through_descriptor(16),
        values: source.to_vec(),
        scale_base: Vec::new(),
        scale_factor: Vec::new(),
        offset_base: Vec::new(),
        offset_factor: Vec::new(),
    })
}

/// Byte-exact BF16 pass-through. Source dtype (GGML BF16 vs F16) is carried separately for the
/// compiler; the descriptor is float/16. Typed helpers use `simsimd::bf16`, never `half`.
pub fn pass_through_bf16(source: &[u8], elems: usize) -> Result<ConstantData, PrepareError> {
    if source.len() != elems.checked_mul(2).ok_or(PrepareError::Shape)? {
        return Err(PrepareError::SourceLength);
    }
    Ok(ConstantData {
        descriptor: float_pass_through_descriptor(16),
        values: source.to_vec(),
        scale_base: Vec::new(),
        scale_factor: Vec::new(),
        offset_base: Vec::new(),
        offset_factor: Vec::new(),
    })
}

pub fn pass_through_i8(source: &[u8], elems: usize) -> Result<ConstantData, PrepareError> {
    if source.len() != elems {
        return Err(PrepareError::SourceLength);
    }
    Ok(ConstantData {
        descriptor: ConstantDescriptor {
            numeric_kind: NumericKind::Integer,
            bit_width: 8,
            storage_bits: 8,
            signedness: Signedness::Signed,
            block_elems: 1,
            group_elems: 1,
            packing: Packing::Unpacked,
            scale_encoding: ScaleEncoding::None,
            offset_encoding: OffsetEncoding::None,
            exactness: Exactness::SourceBytes,
        },
        values: source.to_vec(),
        scale_base: Vec::new(),
        scale_factor: Vec::new(),
        offset_base: Vec::new(),
        offset_factor: Vec::new(),
    })
}

/// Byte-exact F8 pass-through foundation (`bit_width = 8`). Vendors without FP8 consumption
/// (current iVPU path) must reject at bake time. Gated crate types live behind `dtype-f8`.
pub fn pass_through_f8(source: &[u8], elems: usize) -> Result<ConstantData, PrepareError> {
    if source.len() != elems {
        return Err(PrepareError::SourceLength);
    }
    Ok(ConstantData {
        descriptor: float_pass_through_descriptor(8),
        values: source.to_vec(),
        scale_base: Vec::new(),
        scale_factor: Vec::new(),
        offset_base: Vec::new(),
        offset_factor: Vec::new(),
    })
}

fn float_pass_through_descriptor(bit_width: i32) -> ConstantDescriptor {
    ConstantDescriptor {
        numeric_kind: NumericKind::Float,
        bit_width,
        storage_bits: bit_width,
        signedness: Signedness::NotApplicable,
        block_elems: 1,
        group_elems: 1,
        packing: Packing::Unpacked,
        scale_encoding: ScaleEncoding::None,
        offset_encoding: OffsetEncoding::None,
        exactness: Exactness::SourceBytes,
    }
}

/// True when a prepared constant is an 8-bit float that the current iVPU lowering must stub.
pub fn is_f8_constant(descriptor: ConstantDescriptor) -> bool {
    descriptor.numeric_kind == NumericKind::Float && descriptor.bit_width == 8
}

/// Transcode Q4_K into regular logical-order packed u4 plus exact factorized scale/min metadata.
pub fn transcode_q4_k(source: &[u8], k: usize, n: usize) -> Result<ConstantData, PrepareError> {
    let blocks = expected_blocks(k, n)?;
    if source.len() != blocks.checked_mul(Q4_K_BYTES).ok_or(PrepareError::Shape)? {
        return Err(PrepareError::SourceLength);
    }
    let elems = k.checked_mul(n).ok_or(PrepareError::Shape)?;
    let mut values = vec![0u8; packed_len(elems, 4)?];
    let mut scale_base = Vec::with_capacity(blocks * 2);
    let mut scale_factor = Vec::with_capacity(blocks * 8);
    let mut offset_base = Vec::with_capacity(blocks * 2);
    let mut offset_factor = Vec::with_capacity(blocks * 8);
    let kernel = ProductionKernel::detect();
    crate::trace::quant_kernel(kernel.name());

    let (source_blocks, remainder) = source.as_chunks::<Q4_K_BYTES>();
    debug_assert!(remainder.is_empty());
    match kernel {
        ProductionKernel::Scalar(kernel) => transcode_q4_blocks(
            kernel,
            source_blocks,
            &mut values,
            &mut scale_base,
            &mut scale_factor,
            &mut offset_base,
            &mut offset_factor,
        ),
        #[cfg(all(
            feature = "portable-simd",
            any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "arm",
                target_arch = "aarch64"
            )
        ))]
        ProductionKernel::Portable(kernel) => transcode_q4_blocks(
            kernel,
            source_blocks,
            &mut values,
            &mut scale_base,
            &mut scale_factor,
            &mut offset_base,
            &mut offset_factor,
        ),
        #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
        ProductionKernel::Avx2Q6(_) => transcode_q4_blocks(
            super::quant_kernel::PortableSimdKernel,
            source_blocks,
            &mut values,
            &mut scale_base,
            &mut scale_factor,
            &mut offset_base,
            &mut offset_factor,
        ),
    }

    Ok(ConstantData {
        descriptor: native_descriptor(
            4,
            Signedness::Unsigned,
            32,
            ScaleEncoding::BlockF16TimesGroupU8,
            OffsetEncoding::NegativeBlockF16TimesGroupU8,
        ),
        values,
        scale_base,
        scale_factor,
        offset_base,
        offset_factor,
    })
}

/// Transcode Q6_K into a regular logical-order dense 6-bit stream plus exact factorized scales.
pub fn transcode_q6_k(source: &[u8], k: usize, n: usize) -> Result<ConstantData, PrepareError> {
    let blocks = expected_blocks(k, n)?;
    if source.len() != blocks.checked_mul(Q6_K_BYTES).ok_or(PrepareError::Shape)? {
        return Err(PrepareError::SourceLength);
    }
    let elems = k.checked_mul(n).ok_or(PrepareError::Shape)?;
    let mut values = vec![0u8; packed_len(elems, 6)?];
    let mut scale_base = Vec::with_capacity(blocks * 2);
    let mut scale_factor = Vec::with_capacity(blocks * 16);
    let kernel = ProductionKernel::detect();
    crate::trace::quant_kernel(kernel.name());

    let (source_blocks, remainder) = source.as_chunks::<Q6_K_BYTES>();
    debug_assert!(remainder.is_empty());
    match kernel {
        ProductionKernel::Scalar(kernel) => transcode_q6_blocks(
            kernel,
            source_blocks,
            &mut values,
            &mut scale_base,
            &mut scale_factor,
        ),
        #[cfg(all(
            feature = "portable-simd",
            any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "arm",
                target_arch = "aarch64"
            )
        ))]
        ProductionKernel::Portable(kernel) => transcode_q6_blocks(
            kernel,
            source_blocks,
            &mut values,
            &mut scale_base,
            &mut scale_factor,
        ),
        #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
        ProductionKernel::Avx2Q6(kernel) => transcode_q6_blocks(
            kernel,
            source_blocks,
            &mut values,
            &mut scale_base,
            &mut scale_factor,
        ),
    }

    Ok(ConstantData {
        descriptor: native_descriptor(
            6,
            Signedness::Signed,
            16,
            ScaleEncoding::BlockF16TimesGroupI8,
            OffsetEncoding::None,
        ),
        values,
        scale_base,
        scale_factor,
        offset_base: Vec::new(),
        offset_factor: Vec::new(),
    })
}

/// Transcode Q8_0 into unpacked signed i8 quants plus per-group f16 scales (group = 32).
/// No f32 dequant; qs and scales are copied from the source blocks.
pub fn transcode_q8_0(source: &[u8], k: usize, n: usize) -> Result<ConstantData, PrepareError> {
    let blocks = expected_q8_0_blocks(k, n)?;
    if source.len() != blocks.checked_mul(Q8_0_BYTES).ok_or(PrepareError::Shape)? {
        return Err(PrepareError::SourceLength);
    }
    let elems = k.checked_mul(n).ok_or(PrepareError::Shape)?;
    let mut values = Vec::with_capacity(elems);
    let mut scale_base = Vec::with_capacity(blocks * 2);
    for block in source.chunks_exact(Q8_0_BYTES) {
        scale_base.extend_from_slice(&block[0..2]);
        values.extend_from_slice(&block[2..Q8_0_BYTES]);
    }
    Ok(ConstantData {
        descriptor: ConstantDescriptor {
            numeric_kind: NumericKind::Integer,
            bit_width: 8,
            storage_bits: 8,
            signedness: Signedness::Signed,
            block_elems: Q8_0_BLOCK as i32,
            group_elems: Q8_0_BLOCK as i32,
            packing: Packing::Unpacked,
            scale_encoding: ScaleEncoding::PerGroupF16,
            offset_encoding: OffsetEncoding::None,
            exactness: Exactness::SourceNumeric,
        },
        values,
        scale_base,
        scale_factor: Vec::new(),
        offset_base: Vec::new(),
        offset_factor: Vec::new(),
    })
}

/// Transcode Q8_K into unpacked signed i8 quants plus exact per-block f32 scales.
/// Derived `bsums` are dropped (recoverable from qs). No f32 dequant of the weights.
pub fn transcode_q8_k(source: &[u8], k: usize, n: usize) -> Result<ConstantData, PrepareError> {
    let blocks = expected_blocks(k, n)?;
    if source.len() != blocks.checked_mul(Q8_K_BYTES).ok_or(PrepareError::Shape)? {
        return Err(PrepareError::SourceLength);
    }
    let elems = k.checked_mul(n).ok_or(PrepareError::Shape)?;
    let mut values = Vec::with_capacity(elems);
    let mut scale_base = Vec::with_capacity(blocks * 4);
    for block in source.chunks_exact(Q8_K_BYTES) {
        scale_base.extend_from_slice(&block[0..4]);
        values.extend_from_slice(&block[4..4 + SUPER_BLOCK]);
        // Skip bsums at [260..292]; they are derived from qs for vec_dot.
    }
    Ok(ConstantData {
        descriptor: ConstantDescriptor {
            numeric_kind: NumericKind::Integer,
            bit_width: 8,
            storage_bits: 8,
            signedness: Signedness::Signed,
            block_elems: SUPER_BLOCK as i32,
            group_elems: SUPER_BLOCK as i32,
            packing: Packing::Unpacked,
            scale_encoding: ScaleEncoding::PerGroupF32,
            offset_encoding: OffsetEncoding::None,
            exactness: Exactness::SourceNumeric,
        },
        values,
        scale_base,
        scale_factor: Vec::new(),
        offset_base: Vec::new(),
        offset_factor: Vec::new(),
    })
}

fn transcode_q4_blocks(
    kernel: impl QuantKernel,
    source_blocks: &[[u8; Q4_K_BYTES]],
    values: &mut [u8],
    scale_base: &mut Vec<u8>,
    scale_factor: &mut Vec<u8>,
    offset_base: &mut Vec<u8>,
    offset_factor: &mut Vec<u8>,
) {
    for (block, packed) in source_blocks.iter().zip(values.as_chunks_mut::<128>().0) {
        scale_base.extend_from_slice(&block[0..2]);
        offset_base.extend_from_slice(&block[2..4]);
        let metadata = &block[4..16];
        for group in 0..8 {
            let (scale, minimum) = q4_scale_min(metadata, group);
            scale_factor.push(scale);
            offset_factor.push(minimum);
        }
        let quants: &[u8; 128] = block[16..144]
            .try_into()
            .expect("fixed Q4 source block layout");
        kernel.pack_u4_split_nibbles(quants, packed);
    }
}

fn transcode_q6_blocks(
    kernel: impl QuantKernel,
    source_blocks: &[[u8; Q6_K_BYTES]],
    values: &mut [u8],
    scale_base: &mut Vec<u8>,
    scale_factor: &mut Vec<u8>,
) {
    for (block, packed) in source_blocks.iter().zip(values.as_chunks_mut::<192>().0) {
        scale_factor.extend_from_slice(&block[192..208]);
        scale_base.extend_from_slice(&block[208..210]);
        let low: &[u8; 128] = block[0..128].try_into().expect("fixed Q6 low-plane layout");
        let high: &[u8; 64] = block[128..192]
            .try_into()
            .expect("fixed Q6 high-plane layout");
        kernel.pack_i6_quartets(low, high, packed);
    }
}

fn q4_scale_min(metadata: &[u8], group: usize) -> (u8, u8) {
    if group < 4 {
        (metadata[group] & 63, metadata[group + 4] & 63)
    } else {
        (
            (metadata[group + 4] & 0x0f) | ((metadata[group - 4] >> 6) << 4),
            (metadata[group + 4] >> 4) | ((metadata[group] >> 6) << 4),
        )
    }
}

fn packed_len(elems: usize, bits: usize) -> Result<usize, PrepareError> {
    elems
        .checked_mul(bits)
        .and_then(|value| value.checked_add(7))
        .map(|value| value / 8)
        .ok_or(PrepareError::Shape)
}

#[cfg(test)]
fn get_packed(input: &[u8], index: usize, bits: usize) -> u8 {
    let bit = index * bits;
    let byte = bit / 8;
    let shift = bit % 8;
    let mut value = (input[byte] as u16) >> shift;
    if shift + bits > 8 {
        value |= (input[byte + 1] as u16) << (8 - shift);
    }
    (value & ((1u16 << bits) - 1)) as u8
}

#[cfg(test)]
fn read_f16(bytes: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])) as f32
}

#[cfg(test)]
fn dequantize_native(value: &ConstantData, elems: usize) -> Vec<f32> {
    let descriptor = value.descriptor;
    let groups_per_block = descriptor.block_elems as usize / descriptor.group_elems as usize;
    let mut output = vec![0.0; elems];
    for (index, output) in output.iter_mut().enumerate() {
        let block = index / descriptor.block_elems as usize;
        let group = index / descriptor.group_elems as usize;
        let raw = get_packed(&value.values, index, descriptor.storage_bits as usize);
        let quant = if descriptor.signedness == Signedness::Signed {
            let shift = 8 - descriptor.bit_width;
            ((raw << shift) as i8 >> shift) as f32
        } else {
            raw as f32
        };
        let base = read_f16(&value.scale_base[block * 2..block * 2 + 2]);
        let factor = match descriptor.scale_encoding {
            ScaleEncoding::BlockF16TimesGroupI8 => value.scale_factor[group] as i8 as f32,
            ScaleEncoding::BlockF16TimesGroupU8 => value.scale_factor[group] as f32,
            _ => unreachable!(),
        };
        *output = quant * base * factor;
        if descriptor.offset_encoding == OffsetEncoding::NegativeBlockF16TimesGroupU8 {
            let offset_base = read_f16(&value.offset_base[block * 2..block * 2 + 2]);
            *output -= offset_base * value.offset_factor[group] as f32;
        }
        debug_assert_eq!(group / groups_per_block, block);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::requant;
    use alloc::string::ToString;

    fn f16_bytes(value: f32) -> [u8; 2] {
        (value as f16).to_bits().to_le_bytes()
    }

    fn make_q4_block(block_scale: f32, block_min: f32) -> Vec<u8> {
        let scales = [1, 2, 3, 4, 17, 18, 19, 20];
        let minimums = [5, 6, 7, 8, 21, 22, 23, 24];
        let mut block = vec![0u8; Q4_K_BYTES];
        block[0..2].copy_from_slice(&f16_bytes(block_scale));
        block[2..4].copy_from_slice(&f16_bytes(block_min));
        for group in 0..8 {
            set_q4_scale_min(&mut block[4..16], group, scales[group], minimums[group]);
        }
        for chunk in 0..4 {
            for lane in 0..32 {
                let low_index = chunk * 64 + lane;
                let high_index = low_index + 32;
                block[16 + chunk * 32 + lane] =
                    (low_index % 16) as u8 | (((high_index % 16) as u8) << 4);
            }
        }
        block
    }

    fn set_q4_scale_min(metadata: &mut [u8], group: usize, scale: u8, minimum: u8) {
        if group < 4 {
            metadata[group] = (metadata[group] & 0xc0) | (scale & 0x3f);
            metadata[group + 4] = (metadata[group + 4] & 0xc0) | (minimum & 0x3f);
        } else {
            metadata[group + 4] = (scale & 0x0f) | ((minimum & 0x0f) << 4);
            metadata[group - 4] |= (scale >> 4) << 6;
            metadata[group] |= (minimum >> 4) << 6;
        }
    }

    fn q4_reference(block: &[u8]) -> Vec<f32> {
        let d = read_f16(&block[0..2]);
        let dmin = read_f16(&block[2..4]);
        let mut result = vec![0.0; SUPER_BLOCK];
        for chunk in 0..4 {
            let (scale0, min0) = q4_scale_min(&block[4..16], chunk * 2);
            let (scale1, min1) = q4_scale_min(&block[4..16], chunk * 2 + 1);
            for lane in 0..32 {
                let byte = block[16 + chunk * 32 + lane];
                result[chunk * 64 + lane] =
                    d * scale0 as f32 * (byte & 15) as f32 - dmin * min0 as f32;
                result[chunk * 64 + 32 + lane] =
                    d * scale1 as f32 * (byte >> 4) as f32 - dmin * min1 as f32;
            }
        }
        result
    }

    fn make_q6_block(block_scale: f32) -> Vec<u8> {
        let mut block = vec![0u8; Q6_K_BYTES];
        for (index, scale) in (-8i8..8).enumerate() {
            block[192 + index] = scale as u8;
        }
        block[208..210].copy_from_slice(&f16_bytes(block_scale));
        for half in 0..2 {
            for lane in 0..32 {
                let signed = [
                    ((half * 128 + lane) % 64) as i8 - 32,
                    ((half * 128 + 32 + lane) % 64) as i8 - 32,
                    ((half * 128 + 64 + lane) % 64) as i8 - 32,
                    ((half * 128 + 96 + lane) % 64) as i8 - 32,
                ];
                let raw = signed.map(|value| (value as i16 + 32) as u8);
                let low_base = half * 64;
                block[low_base + lane] = (raw[0] & 15) | ((raw[2] & 15) << 4);
                block[low_base + 32 + lane] = (raw[1] & 15) | ((raw[3] & 15) << 4);
                block[128 + half * 32 + lane] = (raw[0] >> 4)
                    | ((raw[1] >> 4) << 2)
                    | ((raw[2] >> 4) << 4)
                    | ((raw[3] >> 4) << 6);
            }
        }
        block
    }

    fn q6_reference(block: &[u8]) -> Vec<f32> {
        let d = read_f16(&block[208..210]);
        let mut result = vec![0.0; SUPER_BLOCK];
        for half in 0..2 {
            for lane in 0..32 {
                let low_base = half * 64;
                let high = block[128 + half * 32 + lane];
                let raw = [
                    (block[low_base + lane] & 15) | ((high & 3) << 4),
                    (block[low_base + 32 + lane] & 15) | (((high >> 2) & 3) << 4),
                    (block[low_base + lane] >> 4) | (((high >> 4) & 3) << 4),
                    (block[low_base + 32 + lane] >> 4) | (((high >> 6) & 3) << 4),
                ];
                for quarter in 0..4 {
                    let index = half * 128 + quarter * 32 + lane;
                    let group = index / 16;
                    let quant = raw[quarter] as i8 - 32;
                    let scale = block[192 + group] as i8;
                    result[index] = d * scale as f32 * quant as f32;
                }
            }
        }
        result
    }

    fn canonical_from(values: &mut [f32], bits: i32) -> Requant {
        let operand = Operand {
            ptr: values.as_mut_ptr().cast(),
            elems: values.len(),
            dtype: GGML_TYPE_F32,
            owned: false,
            shared: None,
};
        requant::requant(&operand, SUPER_BLOCK, 1, bits, true).unwrap()
    }

    fn requant_digest(value: &Requant) -> alloc::string::String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&value.bits.to_le_bytes());
        hasher.update(&value.group.to_le_bytes());
        hasher.update(&value.weight);
        hasher.update(&value.scales);
        hasher.update(&value.zeros);
        hasher.finalize().to_hex().to_string()
    }

    #[test]
    fn q4_k_transcode_matches_authoritative_formula_and_descriptor_golden() {
        let source = make_q4_block(0.5, 0.25);
        let native = transcode_q4_k(&source, SUPER_BLOCK, 1).unwrap();
        assert_eq!(
            native.descriptor,
            ConstantDescriptor {
                numeric_kind: NumericKind::Integer,
                bit_width: 4,
                storage_bits: 4,
                signedness: Signedness::Unsigned,
                block_elems: 256,
                group_elems: 32,
                packing: Packing::DenseLsb,
                scale_encoding: ScaleEncoding::BlockF16TimesGroupU8,
                offset_encoding: OffsetEncoding::NegativeBlockF16TimesGroupU8,
                exactness: Exactness::SourceNumeric,
            }
        );
        assert_eq!(
            &native.values[..8],
            &[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]
        );
        assert_eq!(native.scale_factor, [1, 2, 3, 4, 17, 18, 19, 20]);
        assert_eq!(native.offset_factor, [5, 6, 7, 8, 21, 22, 23, 24]);
        assert_eq!(native.stored_bytes(), 148);

        let reference = q4_reference(&source);
        let actual = dequantize_native(&native, SUPER_BLOCK);
        assert_eq!(actual, reference);
        assert_eq!(actual[0], -1.25);
        assert_eq!(actual[255], 150.0 - 6.0);

        // Dense matmul uses the same accumulation order on both sides; exact source semantics
        // therefore produce a zero-tolerance match before compiler f16 execution is considered.
        let activation: Vec<f32> = (0..SUPER_BLOCK)
            .map(|index| (index as i32 % 9 - 4) as f32 * 0.125)
            .collect();
        let expected: f32 = reference
            .iter()
            .zip(&activation)
            .map(|(weight, activation)| weight * activation)
            .sum();
        let observed: f32 = actual
            .iter()
            .zip(&activation)
            .map(|(weight, activation)| weight * activation)
            .sum();
        assert_eq!(observed, expected);
    }

    #[test]
    fn q6_k_transcode_preserves_edges_groups_and_source_storage_size() {
        let source = make_q6_block(0.125);
        let native = transcode_q6_k(&source, SUPER_BLOCK, 1).unwrap();
        assert_eq!(native.descriptor.bit_width, 6);
        assert_eq!(native.descriptor.storage_bits, 6);
        assert_eq!(native.descriptor.signedness, Signedness::Signed);
        assert_eq!(native.descriptor.group_elems, 16);
        assert_eq!(
            native.descriptor.scale_encoding,
            ScaleEncoding::BlockF16TimesGroupI8
        );
        assert_eq!(native.descriptor.offset_encoding, OffsetEncoding::None);
        assert_eq!(native.values.len(), 192);
        assert_eq!(native.stored_bytes(), Q6_K_BYTES);
        assert_eq!(get_packed(&native.values, 0, 6), 32);
        assert_eq!(get_packed(&native.values, 63, 6), 31);

        let reference = q6_reference(&source);
        let actual = dequantize_native(&native, SUPER_BLOCK);
        assert_eq!(actual, reference);
        assert_eq!(actual[0], 32.0);
        assert_eq!(actual[63], -19.375);
    }

    #[test]
    fn block_tails_rows_and_malformed_inputs_are_checked() {
        let q4 = make_q4_block(0.5, 0.25);
        let source = q4.repeat(4);
        let native = transcode_q4_k(&source, 512, 2).unwrap();
        assert_eq!(native.values.len(), 512);
        assert_eq!(native.scale_factor.len(), 32);
        assert_eq!(
            dequantize_native(&native, 1024),
            q4_reference(&q4).repeat(4)
        );

        assert_eq!(
            transcode_q4_k(&source[..source.len() - 1], 512, 2).unwrap_err(),
            PrepareError::SourceLength
        );
        assert_eq!(
            transcode_q4_k(&source, 257, 1).unwrap_err(),
            PrepareError::Shape
        );
        assert_eq!(
            transcode_q6_k(&vec![0; Q6_K_BYTES - 1], 256, 1).unwrap_err(),
            PrepareError::SourceLength
        );
    }

    #[test]
    fn current_canonical_output_for_quantized_blocks_is_golden() {
        let mut q4 = q4_reference(&make_q4_block(0.5, 0.25));
        let q4_u4 = canonical_from(&mut q4, 4);
        assert_eq!(
            requant_digest(&q4_u4),
            "285b03af698bbdc49c7e876a9082cef58687c7b722250be2e74a956195af7467"
        );

        let mut q6 = q6_reference(&make_q6_block(0.125));
        let q6_u8 = canonical_from(&mut q6, 8);
        assert_eq!(
            requant_digest(&q6_u8),
            "0e06e8dbbeff78e421ea21168073c9acc4ab81a9752ae33d9664f6630b432f2d"
        );
    }

    #[test]
    fn f16_is_byte_pass_through_and_routed_native_is_gated() {
        let source = [0x00, 0x3c, 0x00, 0xc0, 0x55, 0x35];
        let value = pass_through_f16(&source, 3).unwrap();
        assert_eq!(value.values, source);
        assert_eq!(value.descriptor.exactness, Exactness::SourceBytes);
        assert_eq!(value.descriptor.numeric_kind, NumericKind::Float);

        assert!(source_native_supported(GGML_TYPE_Q4_K, ConstantUse::Dense));
        assert!(!source_native_supported(
            GGML_TYPE_Q4_K,
            ConstantUse::RoutedGather
        ));
        assert!(!source_native_supported(GGML_TYPE_F32, ConstantUse::Dense));
    }

    #[test]
    fn bf16_is_byte_pass_through_and_simsimd_roundtrips() {
        // 1.0bf16 = 0x3f80, -2.0bf16 = 0xc000
        let source = [0x80, 0x3f, 0x00, 0xc0];
        let value = pass_through_bf16(&source, 2).unwrap();
        assert_eq!(value.values, source);
        assert_eq!(value.descriptor.bit_width, 16);
        assert_eq!(value.descriptor.exactness, Exactness::SourceBytes);
        assert!(source_native_supported(GGML_TYPE_BF16, ConstantUse::Dense));
        assert!(!source_native_supported(
            GGML_TYPE_BF16,
            ConstantUse::RoutedGather
        ));

        let one = simsimd::bf16::from_f32(1.0);
        let bits = one.0.to_le_bytes();
        assert_eq!(bits, [0x80, 0x3f]);
        assert!((simsimd::bf16(u16::from_le_bytes(bits)).to_f32() - 1.0).abs() < 1e-2);
    }

    #[test]
    fn q8_0_transcode_copies_qs_and_f16_scales_exactly() {
        let mut block = vec![0u8; Q8_0_BYTES];
        block[0..2].copy_from_slice(&f16_bytes(0.5));
        for i in 0..32 {
            block[2 + i] = (i as i8).wrapping_sub(16) as u8;
        }
        let native = transcode_q8_0(&block, Q8_0_BLOCK, 1).unwrap();
        assert_eq!(
            native.descriptor,
            ConstantDescriptor {
                numeric_kind: NumericKind::Integer,
                bit_width: 8,
                storage_bits: 8,
                signedness: Signedness::Signed,
                block_elems: 32,
                group_elems: 32,
                packing: Packing::Unpacked,
                scale_encoding: ScaleEncoding::PerGroupF16,
                offset_encoding: OffsetEncoding::None,
                exactness: Exactness::SourceNumeric,
            }
        );
        assert_eq!(native.values, block[2..]);
        assert_eq!(native.scale_base, block[0..2]);
        assert_eq!(native.stored_bytes(), Q8_0_BYTES);

        assert_eq!(
            transcode_q8_0(&block[..Q8_0_BYTES - 1], 32, 1).unwrap_err(),
            PrepareError::SourceLength
        );
        assert_eq!(
            transcode_q8_0(&block, 33, 1).unwrap_err(),
            PrepareError::Shape
        );
    }

    #[test]
    fn q8_k_transcode_preserves_qs_and_f32_scale_drops_bsums() {
        let mut block = vec![0u8; Q8_K_BYTES];
        block[0..4].copy_from_slice(&0.125f32.to_le_bytes());
        for i in 0..SUPER_BLOCK {
            block[4 + i] = ((i as i16) - 128) as i8 as u8;
        }
        for i in 0..16 {
            let sum = (i as i16) * 10;
            block[260 + i * 2..260 + i * 2 + 2].copy_from_slice(&sum.to_le_bytes());
        }
        let native = transcode_q8_k(&block, SUPER_BLOCK, 1).unwrap();
        assert_eq!(native.descriptor.scale_encoding, ScaleEncoding::PerGroupF32);
        assert_eq!(native.descriptor.bit_width, 8);
        assert_eq!(native.descriptor.signedness, Signedness::Signed);
        assert_eq!(native.values, block[4..260]);
        assert_eq!(native.scale_base, block[0..4]);
        assert_eq!(native.stored_bytes(), SUPER_BLOCK + 4);
        assert!(source_native_supported(GGML_TYPE_Q8_K, ConstantUse::Dense));
        assert_eq!(
            transcode_q8_k(&block[..Q8_K_BYTES - 1], SUPER_BLOCK, 1).unwrap_err(),
            PrepareError::SourceLength
        );
    }

    #[test]
    fn f8_pass_through_sets_float8_descriptor() {
        let source = [0x3c, 0x00, 0x7f];
        let value = pass_through_f8(&source, 3).unwrap();
        assert_eq!(value.values, source);
        assert_eq!(value.descriptor.bit_width, 8);
        assert_eq!(value.descriptor.numeric_kind, NumericKind::Float);
        assert!(is_f8_constant(value.descriptor));
        assert!(!is_f8_constant(pass_through_f16(&[0, 0], 1).unwrap().descriptor));
    }

    #[cfg(feature = "bench")]
    #[test]
    #[ignore = "explicit conversion-latency benchmark"]
    fn benchmark_full_native_conversion() {
        use std::hint::black_box;
        use std::time::Instant;

        const K: usize = 4096;
        const N: usize = 256;
        const BLOCKS: usize = K / SUPER_BLOCK * N;

        fn measure(
            source: &[u8],
            convert: impl Fn(&[u8]) -> Result<ConstantData, PrepareError>,
        ) -> (f64, usize) {
            let mut samples = Vec::with_capacity(31);
            let mut stored_bytes = 0;
            for _ in 0..31 {
                let start = Instant::now();
                let output = convert(black_box(source)).unwrap();
                samples.push(start.elapsed().as_secs_f64() * 1e6);
                stored_bytes = black_box(output.stored_bytes());
            }
            samples.sort_by(f64::total_cmp);
            (samples[samples.len() / 2], stored_bytes)
        }

        let q4 = make_q4_block(0.5, 0.25).repeat(BLOCKS);
        let q6 = make_q6_block(0.5).repeat(BLOCKS);
        let (q4_us, q4_bytes) = measure(&q4, |source| transcode_q4_k(source, K, N));
        let (q6_us, q6_bytes) = measure(&q6, |source| transcode_q6_k(source, K, N));
        std::eprintln!(
            "full native conversion: kernel={} elems={} q4_us={q4_us:.3} q4_bytes={q4_bytes} \
             q6_us={q6_us:.3} q6_bytes={q6_bytes}",
            ProductionKernel::detect().name(),
            K * N
        );
    }
}
