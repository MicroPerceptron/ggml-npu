//! Linux DRM core, PRIME, and GEM helpers shared by DRM drivers.
//!
//! Source: Linux `drm.h`, SPDX `MIT`. Vendor command structures do not belong here.

use core::mem::{align_of, size_of};

use rustix::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};

use super::file::LinuxFile;
use super::ioctl::{encode, set, update, IOC_READ, IOC_WRITE};

const DRM_IOCTL_BASE: u32 = b'd' as u32;

const fn drm_iow(number: u32, bytes: usize) -> u32 {
    encode(IOC_WRITE, DRM_IOCTL_BASE, number, bytes)
}

const fn drm_iowr(number: u32, bytes: usize) -> u32 {
    encode(IOC_READ | IOC_WRITE, DRM_IOCTL_BASE, number, bytes)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DrmGemClose {
    handle: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DrmGetCap {
    capability: u64,
    value: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DrmPrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}

const DRM_CAP_PRIME: u64 = 5;
const DRM_PRIME_CAP_IMPORT: u64 = 1;
const DRM_PRIME_CAP_EXPORT: u64 = 2;
const DRM_CLOEXEC: u32 = 0x0008_0000;
const DRM_RDWR: u32 = 0x0000_0002;

const DRM_IOCTL_GEM_CLOSE: u32 = drm_iow(0x09, size_of::<DrmGemClose>());
const DRM_IOCTL_GET_CAP: u32 = drm_iowr(0x0c, size_of::<DrmGetCap>());
const DRM_IOCTL_PRIME_HANDLE_TO_FD: u32 = drm_iowr(0x2d, size_of::<DrmPrimeHandle>());
const DRM_IOCTL_PRIME_FD_TO_HANDLE: u32 = drm_iowr(0x2e, size_of::<DrmPrimeHandle>());

const _: [(); 8] = [(); size_of::<DrmGemClose>()];
const _: [(); 4] = [(); align_of::<DrmGemClose>()];
const _: [(); 16] = [(); size_of::<DrmGetCap>()];
const _: [(); 8] = [(); align_of::<DrmGetCap>()];
const _: [(); 12] = [(); size_of::<DrmPrimeHandle>()];
const _: [(); 4] = [(); align_of::<DrmPrimeHandle>()];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrmError {
    Os(rustix::io::Errno),
    InvalidResponse,
    InvalidHandle,
}

impl From<rustix::io::Errno> for DrmError {
    fn from(value: rustix::io::Errno) -> Self {
        Self::Os(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrimeCapabilities {
    pub import: bool,
    pub export: bool,
}

fn validate_import_handle(handle: u32) -> Result<u32, DrmError> {
    (handle != 0)
        .then_some(handle)
        .ok_or(DrmError::InvalidResponse)
}

fn validate_export_fd(fd: i32) -> Result<i32, DrmError> {
    (fd >= 0).then_some(fd).ok_or(DrmError::InvalidResponse)
}

pub(crate) fn prime_capabilities(file: &LinuxFile) -> Result<PrimeCapabilities, DrmError> {
    let mut value = DrmGetCap {
        capability: DRM_CAP_PRIME,
        value: 0,
    };
    unsafe { update::<DRM_IOCTL_GET_CAP, _>(file.as_fd(), &mut value) }?;
    Ok(PrimeCapabilities {
        import: value.value & DRM_PRIME_CAP_IMPORT != 0,
        export: value.value & DRM_PRIME_CAP_EXPORT != 0,
    })
}

pub(crate) fn prime_fd_to_handle(
    file: &LinuxFile,
    shared_fd: BorrowedFd<'_>,
) -> Result<u32, DrmError> {
    let mut value = DrmPrimeHandle {
        handle: 0,
        flags: 0,
        fd: shared_fd.as_raw_fd(),
    };
    unsafe { update::<DRM_IOCTL_PRIME_FD_TO_HANDLE, _>(file.as_fd(), &mut value) }?;
    validate_import_handle(value.handle)
}

pub(crate) fn prime_handle_to_fd(file: &LinuxFile, handle: u32) -> Result<OwnedFd, DrmError> {
    if handle == 0 {
        return Err(DrmError::InvalidHandle);
    }
    let mut value = DrmPrimeHandle {
        handle,
        flags: DRM_CLOEXEC | DRM_RDWR,
        fd: -1,
    };
    unsafe { update::<DRM_IOCTL_PRIME_HANDLE_TO_FD, _>(file.as_fd(), &mut value) }?;
    let fd = validate_export_fd(value.fd)?;
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

pub(crate) fn close_gem(file: &LinuxFile, handle: u32) -> Result<(), DrmError> {
    if handle == 0 {
        return Err(DrmError::InvalidHandle);
    }
    unsafe { set::<DRM_IOCTL_GEM_CLOSE, _>(file.as_fd(), DrmGemClose { handle, pad: 0 }) }
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drm_core_layouts_and_requests_match_uapi() {
        assert_eq!(DRM_IOCTL_GEM_CLOSE, 0x4008_6409);
        assert_eq!(DRM_IOCTL_GET_CAP, 0xc010_640c);
        assert_eq!(DRM_IOCTL_PRIME_HANDLE_TO_FD, 0xc00c_642d);
        assert_eq!(DRM_IOCTL_PRIME_FD_TO_HANDLE, 0xc00c_642e);
        assert_eq!(
            (size_of::<DrmGemClose>(), align_of::<DrmGemClose>()),
            (8, 4)
        );
        assert_eq!((size_of::<DrmGetCap>(), align_of::<DrmGetCap>()), (16, 8));
        assert_eq!(
            (size_of::<DrmPrimeHandle>(), align_of::<DrmPrimeHandle>()),
            (12, 4)
        );
    }

    #[test]
    fn prime_response_validation_rejects_invalid_values() {
        assert_eq!(validate_import_handle(0), Err(DrmError::InvalidResponse));
        assert_eq!(validate_import_handle(7), Ok(7));
        assert_eq!(validate_export_fd(-1), Err(DrmError::InvalidResponse));
        assert_eq!(validate_export_fd(3), Ok(3));
    }
}
