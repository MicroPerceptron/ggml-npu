//! Address and offset domains defined by the IVPU device and compiler adapters.

use crate::address::{Address, AddressRange, Domain, Offset, OffsetRange};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum DeviceVirtualAddressDomain {}
impl Domain for DeviceVirtualAddressDomain {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ElfFileOffsetDomain {}
impl Domain for ElfFileOffsetDomain {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ElfSectionOffsetDomain {}
impl Domain for ElfSectionOffsetDomain {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum BindingOffsetDomain {}
impl Domain for BindingOffsetDomain {}

pub(crate) type IvpuAddr = Address<DeviceVirtualAddressDomain>;
pub(crate) type IvpuRange = AddressRange<DeviceVirtualAddressDomain>;
pub(crate) type IvpuOffset = Offset<DeviceVirtualAddressDomain>;

pub(crate) type FileOffset = Offset<ElfFileOffsetDomain>;
pub(crate) type FileRange = OffsetRange<ElfFileOffsetDomain>;
pub(crate) type SectionOffset = Offset<ElfSectionOffsetDomain>;
pub(crate) type SectionRange = OffsetRange<ElfSectionOffsetDomain>;

pub(crate) type BindingOffset = Offset<BindingOffsetDomain>;
pub(crate) type BindingRange = OffsetRange<BindingOffsetDomain>;
