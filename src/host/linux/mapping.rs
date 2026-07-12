//! Owner-complete Linux mapping variants used by Linux composition adapters.

use core::ptr::NonNull;

use super::super::MemoryMapping;
use super::dmabuf::LinuxDmaBufMapping;
use super::file::LinuxMapping;

pub(crate) enum LinuxOwnedMapping {
    File(LinuxMapping),
    DmaBuf(LinuxDmaBufMapping),
}

impl MemoryMapping for LinuxOwnedMapping {
    fn address(&self) -> NonNull<u8> {
        match self {
            Self::File(mapping) => mapping.address(),
            Self::DmaBuf(mapping) => MemoryMapping::address(mapping),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::File(mapping) => mapping.len(),
            Self::DmaBuf(mapping) => MemoryMapping::len(mapping),
        }
    }
}
