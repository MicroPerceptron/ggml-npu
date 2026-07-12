//! Vendor-neutral descriptions of compile-time constants.
//!
//! The contract describes numerical meaning and storage without naming a source format or a
//! compiler. Source adapters may repack values into a regular logical order, but must state whether
//! source bytes or only source integer/metadata semantics remain exact.

use alloc::vec::Vec;

/// Broad scalar category. Signedness is carried separately for integer constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum NumericKind {
    Float = 0,
    Integer = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Signedness {
    NotApplicable = 0,
    Unsigned = 1,
    Signed = 2,
}

/// How consecutive logical values are represented in `ConstantData::values`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Packing {
    /// One byte for 4/6/8-bit integers, or the natural byte width for floating-point values.
    Unpacked = 0,
    /// A dense little-endian bit stream: logical value `i` starts at bit `i * storage_bits`.
    DenseLsb = 1,
}

/// Scale metadata and the exact graph expression it denotes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ScaleEncoding {
    None = 0,
    /// One little-endian f16 scale for each logical group.
    PerGroupF16 = 1,
    /// One f16 base per block multiplied by one unsigned byte factor per group.
    BlockF16TimesGroupU8 = 2,
    /// One f16 base per block multiplied by one signed byte factor per group.
    BlockF16TimesGroupI8 = 3,
    /// One little-endian f32 scale for each logical group (e.g. Q8_K super-block `d`).
    /// Prefer f16 encodings when the source scale is already f16; use this only when the
    /// source scale is natively f32 and must stay exact.
    PerGroupF32 = 4,
}

/// Offset metadata and the exact graph expression it denotes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum OffsetEncoding {
    None = 0,
    /// Canonical asymmetric form: `(q - per_group_u8_zero) * scale`.
    ZeroPointU8 = 1,
    /// Affine source form: `q * scale - block_f16_min * per_group_u8_factor`.
    NegativeBlockF16TimesGroupU8 = 2,
}

/// What was preserved while preparing the compiler constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Exactness {
    /// Values were dequantized and quantized again.
    Requantized = 0,
    /// Source integers and scale/offset metadata are exact, though values may be regularly repacked.
    SourceNumeric = 1,
    /// Source bytes are passed through unchanged.
    SourceBytes = 2,
}

/// Semantic and physical layout of a compile-time constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ConstantDescriptor {
    pub numeric_kind: NumericKind,
    pub bit_width: i32,
    pub storage_bits: i32,
    pub signedness: Signedness,
    pub block_elems: i32,
    pub group_elems: i32,
    pub packing: Packing,
    pub scale_encoding: ScaleEncoding,
    pub offset_encoding: OffsetEncoding,
    pub exactness: Exactness,
}

/// Owned storage backing a [`ConstantDescriptor`].
///
/// `scale_base`/`offset_base` contain little-endian scale/offset bases as described by
/// [`ScaleEncoding`] / [`OffsetEncoding`] (typically f16; [`ScaleEncoding::PerGroupF32`] stores
/// f32). Factor arrays contain the integer metadata specified by the descriptor. Canonical
/// fallback uses `scale_base` for per-group f16 scales and `offset_base` for zero points.
#[derive(Debug)]
pub struct ConstantData {
    pub descriptor: ConstantDescriptor,
    pub values: Vec<u8>,
    pub scale_base: Vec<u8>,
    pub scale_factor: Vec<u8>,
    pub offset_base: Vec<u8>,
    pub offset_factor: Vec<u8>,
}

impl ConstantData {
    #[cfg(any(test, feature = "bench"))]
    pub fn stored_bytes(&self) -> usize {
        self.values.len()
            + self.scale_base.len()
            + self.scale_factor.len()
            + self.offset_base.len()
            + self.offset_factor.len()
    }
}
