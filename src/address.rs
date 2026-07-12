//! Fixed-width semantic addresses, offsets, lengths, alignments, and checked ranges.
//!
//! This module defines mechanisms and invariants only. Concrete address and offset domains belong
//! to the adapter that assigns their meaning. Live host mappings deliberately remain owned
//! pointers plus host-sized lengths; converting them to integers would discard provenance and
//! lifetime information.

#![allow(dead_code)]

use core::fmt;
use core::marker::PhantomData;
use core::num::NonZeroU64;
use core::ops::Range as HostRange;

/// Marker implemented by adapters to prevent values from unrelated domains being mixed.
pub(crate) trait Domain: Copy + Eq + Ord {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WidthError {
    DoesNotFitUsize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AlignmentError {
    Zero,
    NotPowerOfTwo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RangeError {
    Reversed,
    Overflow,
    Empty,
}

/// A byte length. Zero is valid because serialized objects and half-open ranges may be empty.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ByteLen(u64);

impl ByteLen {
    pub(crate) const ZERO: Self = Self(0);

    pub(crate) const fn new(bytes: u64) -> Self {
        Self(bytes)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }

    pub(crate) const fn is_zero(self) -> bool {
        self.0 == 0
    }

    pub(crate) const fn checked_add(self, rhs: Self) -> Option<Self> {
        match self.0.checked_add(rhs.0) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub(crate) const fn checked_mul(self, factor: u64) -> Option<Self> {
        match self.0.checked_mul(factor) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub(crate) fn try_from_usize(bytes: usize) -> Result<Self, WidthError> {
        u64::try_from(bytes)
            .map(Self)
            .map_err(|_| WidthError::DoesNotFitUsize)
    }

    pub(crate) fn try_usize(self) -> Result<usize, WidthError> {
        usize::try_from(self.0).map_err(|_| WidthError::DoesNotFitUsize)
    }
}

/// A validated nonzero power-of-two byte alignment.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Alignment(NonZeroU64);

impl Alignment {
    pub(crate) const ONE: Self = Self(NonZeroU64::MIN);

    pub(crate) const fn new(bytes: u64) -> Result<Self, AlignmentError> {
        if bytes == 0 {
            return Err(AlignmentError::Zero);
        }
        if !bytes.is_power_of_two() {
            return Err(AlignmentError::NotPowerOfTwo);
        }
        match NonZeroU64::new(bytes) {
            Some(value) => Ok(Self(value)),
            None => Err(AlignmentError::Zero),
        }
    }

    pub(crate) const fn get(self) -> u64 {
        self.0.get()
    }

    pub(crate) const fn is_aligned(self, value: u64) -> bool {
        value & (self.get() - 1) == 0
    }

    pub(crate) const fn align_down(self, value: u64) -> u64 {
        value & !(self.get() - 1)
    }

    pub(crate) const fn checked_align_up(self, value: u64) -> Option<u64> {
        let mask = self.get() - 1;
        match value.checked_add(mask) {
            Some(sum) => Some(sum & !mask),
            None => None,
        }
    }
}

/// A fixed-width address whose domain is supplied by an adapter marker.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Address<D: Domain> {
    raw: u64,
    domain: PhantomData<fn() -> D>,
}

impl<D: Domain> Address<D> {
    pub(crate) const fn new(raw: u64) -> Self {
        Self {
            raw,
            domain: PhantomData,
        }
    }

    pub(crate) const fn get(self) -> u64 {
        self.raw
    }

    pub(crate) const fn checked_add_offset(self, offset: Offset<D>) -> Option<Self> {
        match self.raw.checked_add(offset.get()) {
            Some(raw) => Some(Self::new(raw)),
            None => None,
        }
    }

    pub(crate) const fn checked_add_len(self, len: ByteLen) -> Option<Self> {
        match self.raw.checked_add(len.get()) {
            Some(raw) => Some(Self::new(raw)),
            None => None,
        }
    }

    pub(crate) const fn checked_sub_addr(self, base: Self) -> Option<Offset<D>> {
        match self.raw.checked_sub(base.raw) {
            Some(offset) => Some(Offset::new(offset)),
            None => None,
        }
    }

    pub(crate) const fn is_aligned(self, alignment: Alignment) -> bool {
        alignment.is_aligned(self.raw)
    }

    pub(crate) const fn checked_align_up(self, alignment: Alignment) -> Option<Self> {
        match alignment.checked_align_up(self.raw) {
            Some(raw) => Some(Self::new(raw)),
            None => None,
        }
    }

    pub(crate) fn try_usize(self) -> Result<usize, WidthError> {
        usize::try_from(self.raw).map_err(|_| WidthError::DoesNotFitUsize)
    }
}

impl<D: Domain> fmt::Debug for Address<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Address({:#x})", self.raw)
    }
}

/// A fixed-width relative byte position whose domain is supplied by an adapter marker.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Offset<D: Domain> {
    raw: u64,
    domain: PhantomData<fn() -> D>,
}

impl<D: Domain> Offset<D> {
    pub(crate) const fn new(raw: u64) -> Self {
        Self {
            raw,
            domain: PhantomData,
        }
    }

    pub(crate) const fn get(self) -> u64 {
        self.raw
    }

    pub(crate) const fn checked_add_offset(self, offset: Self) -> Option<Self> {
        match self.raw.checked_add(offset.raw) {
            Some(raw) => Some(Self::new(raw)),
            None => None,
        }
    }

    pub(crate) const fn checked_add_len(self, len: ByteLen) -> Option<Self> {
        match self.raw.checked_add(len.get()) {
            Some(raw) => Some(Self::new(raw)),
            None => None,
        }
    }

    pub(crate) const fn checked_sub(self, base: Self) -> Option<Self> {
        match self.raw.checked_sub(base.raw) {
            Some(raw) => Some(Self::new(raw)),
            None => None,
        }
    }

    pub(crate) const fn is_aligned(self, alignment: Alignment) -> bool {
        alignment.is_aligned(self.raw)
    }

    pub(crate) const fn checked_align_up(self, alignment: Alignment) -> Option<Self> {
        match alignment.checked_align_up(self.raw) {
            Some(raw) => Some(Self::new(raw)),
            None => None,
        }
    }

    pub(crate) fn try_from_usize(raw: usize) -> Result<Self, WidthError> {
        u64::try_from(raw)
            .map(Self::new)
            .map_err(|_| WidthError::DoesNotFitUsize)
    }

    pub(crate) fn try_usize(self) -> Result<usize, WidthError> {
        usize::try_from(self.raw).map_err(|_| WidthError::DoesNotFitUsize)
    }
}

impl<D: Domain> fmt::Debug for Offset<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Offset({:#x})", self.raw)
    }
}

pub(crate) trait Position: Copy + Eq + Ord {
    fn from_raw(raw: u64) -> Self;
    fn raw(self) -> u64;

    fn try_usize(self) -> Result<usize, WidthError> {
        usize::try_from(self.raw()).map_err(|_| WidthError::DoesNotFitUsize)
    }
}

impl<D: Domain> Position for Address<D> {
    fn from_raw(raw: u64) -> Self {
        Self::new(raw)
    }

    fn raw(self) -> u64 {
        self.get()
    }
}

impl<D: Domain> Position for Offset<D> {
    fn from_raw(raw: u64) -> Self {
        Self::new(raw)
    }

    fn raw(self) -> u64 {
        self.get()
    }
}

/// An invariant-preserving half-open byte range `[start, end)`.
///
/// `from_start_len` and `from_bounds` deliberately allow an empty range.
/// `from_start_len_non_empty` makes rejection explicit where a payload is required.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CheckedRange<P: Position> {
    start: P,
    end: P,
}

impl<P: Position> CheckedRange<P> {
    pub(crate) fn from_bounds(start: P, end: P) -> Result<Self, RangeError> {
        if start > end {
            Err(RangeError::Reversed)
        } else {
            Ok(Self { start, end })
        }
    }

    pub(crate) fn from_start_len(start: P, len: ByteLen) -> Result<Self, RangeError> {
        let end = start
            .raw()
            .checked_add(len.get())
            .ok_or(RangeError::Overflow)?;
        Ok(Self {
            start,
            end: P::from_raw(end),
        })
    }

    pub(crate) fn from_start_len_non_empty(start: P, len: ByteLen) -> Result<Self, RangeError> {
        if len.is_zero() {
            return Err(RangeError::Empty);
        }
        Self::from_start_len(start, len)
    }

    pub(crate) const fn start(self) -> P {
        self.start
    }

    pub(crate) const fn end(self) -> P {
        self.end
    }

    pub(crate) fn len(self) -> ByteLen {
        ByteLen::new(self.end.raw() - self.start.raw())
    }

    pub(crate) fn is_empty(self) -> bool {
        self.start == self.end
    }

    pub(crate) fn contains_position(self, position: P) -> bool {
        self.start <= position && position < self.end
    }

    pub(crate) fn contains_range(self, other: Self) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    pub(crate) fn contains_non_empty(self, start: P, len: ByteLen) -> bool {
        Self::from_start_len_non_empty(start, len).is_ok_and(|other| self.contains_range(other))
    }

    pub(crate) fn try_usize_range(self) -> Result<HostRange<usize>, WidthError> {
        Ok(self.start.try_usize()?..self.end.try_usize()?)
    }
}

pub(crate) type AddressRange<D> = CheckedRange<Address<D>>;
pub(crate) type OffsetRange<D> = CheckedRange<Offset<D>>;

#[cfg(test)]
mod tests {
    use core::any::TypeId;

    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum FirstDomain {}
    impl Domain for FirstDomain {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum SecondDomain {}
    impl Domain for SecondDomain {}

    type FirstAddress = Address<FirstDomain>;
    type FirstOffset = Offset<FirstDomain>;
    type FirstRange = AddressRange<FirstDomain>;

    #[test]
    fn checked_arithmetic_rejects_overflow_and_underflow() {
        let last = FirstAddress::new(u64::MAX);
        assert_eq!(last.checked_add_len(ByteLen::new(1)), None);
        assert_eq!(
            FirstRange::from_start_len(last, ByteLen::new(1)),
            Err(RangeError::Overflow)
        );
        assert_eq!(
            FirstAddress::new(0x1000).checked_sub_addr(FirstAddress::new(0x1001)),
            None
        );
    }

    #[test]
    fn empty_range_policy_is_explicit() {
        let empty = FirstRange::from_start_len(FirstAddress::new(0x2000), ByteLen::ZERO).unwrap();
        assert!(empty.is_empty());
        assert!(!empty.contains_position(FirstAddress::new(0x2000)));
        assert_eq!(
            FirstRange::from_start_len_non_empty(FirstAddress::new(0x2000), ByteLen::ZERO),
            Err(RangeError::Empty)
        );

        let outer =
            FirstRange::from_start_len_non_empty(FirstAddress::new(0x1000), ByteLen::new(0x1000))
                .unwrap();
        assert!(outer.contains_range(empty));
    }

    #[test]
    fn half_open_range_checks_boundary_endpoints() {
        let range =
            FirstRange::from_start_len_non_empty(FirstAddress::new(0x1000), ByteLen::new(0x100))
                .unwrap();
        assert!(range.contains_position(FirstAddress::new(0x1000)));
        assert!(range.contains_position(FirstAddress::new(0x10ff)));
        assert!(!range.contains_position(FirstAddress::new(0x1100)));
        assert!(range.contains_non_empty(FirstAddress::new(0x1080), ByteLen::new(0x80)));
        assert!(!range.contains_non_empty(FirstAddress::new(0x1080), ByteLen::new(0x81)));
    }

    #[test]
    fn alignment_validates_and_checks_rounding_overflow() {
        assert_eq!(Alignment::new(0), Err(AlignmentError::Zero));
        assert_eq!(Alignment::new(24), Err(AlignmentError::NotPowerOfTwo));

        let alignment = Alignment::new(0x1000).unwrap();
        assert_eq!(alignment.align_down(0x2345), 0x2000);
        assert_eq!(alignment.checked_align_up(0x2345), Some(0x3000));
        assert_eq!(alignment.checked_align_up(u64::MAX), None);
    }

    #[test]
    fn host_width_conversion_is_fallible() {
        assert_eq!(
            ByteLen::try_from_usize(usize::MAX).unwrap().try_usize(),
            Ok(usize::MAX)
        );

        #[cfg(target_pointer_width = "32")]
        assert_eq!(
            ByteLen::new(u64::from(u32::MAX) + 1).try_usize(),
            Err(WidthError::DoesNotFitUsize)
        );

        #[cfg(target_pointer_width = "64")]
        assert_eq!(ByteLen::new(u64::MAX).try_usize(), Ok(usize::MAX));
    }

    #[test]
    fn domains_and_position_kinds_remain_distinct() {
        assert_ne!(
            TypeId::of::<Address<FirstDomain>>(),
            TypeId::of::<Address<SecondDomain>>()
        );
        assert_ne!(
            TypeId::of::<Address<FirstDomain>>(),
            TypeId::of::<Offset<FirstDomain>>()
        );

        let base = FirstAddress::new(0x1000);
        assert_eq!(
            base.checked_add_offset(FirstOffset::new(0x20)),
            Some(FirstAddress::new(0x1020))
        );
    }

    #[test]
    fn offset_range_conversion_preserves_checked_bounds() {
        let start = FirstOffset::new(4);
        let range = OffsetRange::<FirstDomain>::from_start_len(start, ByteLen::new(8)).unwrap();
        assert_eq!(range.try_usize_range(), Ok(4..12));
        assert_eq!(
            OffsetRange::<FirstDomain>::from_bounds(FirstOffset::new(12), start),
            Err(RangeError::Reversed)
        );
    }
}
