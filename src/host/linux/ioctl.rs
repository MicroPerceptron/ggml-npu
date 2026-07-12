//! Typed Linux ioctl encoding and invocation.

use rustix::fd::BorrowedFd;
use rustix::ioctl;

pub(crate) const IOC_NRBITS: u32 = 8;
pub(crate) const IOC_TYPEBITS: u32 = 8;
pub(crate) const IOC_SIZEBITS: u32 = 14;
pub(crate) const IOC_NRSHIFT: u32 = 0;
pub(crate) const IOC_TYPESHIFT: u32 = IOC_NRSHIFT + IOC_NRBITS;
pub(crate) const IOC_SIZESHIFT: u32 = IOC_TYPESHIFT + IOC_TYPEBITS;
pub(crate) const IOC_DIRSHIFT: u32 = IOC_SIZESHIFT + IOC_SIZEBITS;
pub(crate) const IOC_WRITE: u32 = 1;
pub(crate) const IOC_READ: u32 = 2;

pub(crate) const fn encode(direction: u32, ioctl_type: u32, number: u32, bytes: usize) -> u32 {
    (direction << IOC_DIRSHIFT)
        | (ioctl_type << IOC_TYPESHIFT)
        | (number << IOC_NRSHIFT)
        | ((bytes as u32) << IOC_SIZESHIFT)
}

/// # Safety
///
/// `REQUEST`, direction, and `T`'s ABI layout must match the kernel contract.
pub(crate) unsafe fn update<const REQUEST: u32, T>(
    file: BorrowedFd<'_>,
    value: &mut T,
) -> rustix::io::Result<()> {
    unsafe { ioctl::ioctl(file, ioctl::Updater::<REQUEST, T>::new(value)) }
}

/// # Safety
///
/// `REQUEST`, direction, and `T`'s ABI layout must match the kernel contract.
pub(crate) unsafe fn set<const REQUEST: u32, T>(
    file: BorrowedFd<'_>,
    value: T,
) -> rustix::io::Result<()> {
    unsafe { ioctl::ioctl(file, ioctl::Setter::<REQUEST, T>::new(value)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_ioc_encoding_matches_known_values() {
        assert_eq!(encode(IOC_WRITE, b'd' as u32, 0x09, 8), 0x4008_6409);
        assert_eq!(
            encode(IOC_READ | IOC_WRITE, b'd' as u32, 0x0c, 16),
            0xc010_640c
        );
    }
}
