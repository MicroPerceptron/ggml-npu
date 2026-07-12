//! Linux DMA-BUF heap allocation, CPU mapping, and cache-maintenance ownership.

use core::mem::{align_of, size_of};

use rustix::fd::{AsFd, FromRawFd, OwnedFd};
use rustix::fs::{fstat, open, Mode, OFlags};

use crate::interop::{AllocationIdentity, SharedMapping};
use crate::runtime::AccessMode;

use super::super::{FileService, MapOptions, MappingOwner, MemoryMapping};
use super::file::{LinuxFile, LinuxFiles, LinuxMapping};
use super::ioctl::{encode, set, update, IOC_READ, IOC_WRITE};

const DMA_HEAP_PATH: &core::ffi::CStr = c"/dev/dma_heap/system";
const PROVIDER_LINUX_DMABUF: u64 = u64::from_be_bytes(*b"LNXDMABU");

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct DmaHeapAllocationData {
    len: u64,
    fd: u32,
    fd_flags: u32,
    heap_flags: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DmaBufSync {
    flags: u64,
}

const DMA_HEAP_IOCTL_ALLOC: u32 = encode(
    IOC_READ | IOC_WRITE,
    b'H' as u32,
    0,
    size_of::<DmaHeapAllocationData>(),
);
const DMA_BUF_IOCTL_SYNC: u32 = encode(IOC_WRITE, b'b' as u32, 0, size_of::<DmaBufSync>());
const DMA_BUF_SYNC_READ: u64 = 1;
const DMA_BUF_SYNC_WRITE: u64 = 2;
const DMA_BUF_SYNC_END: u64 = 4;

const _: [(); 24] = [(); size_of::<DmaHeapAllocationData>()];
const _: [(); 8] = [(); align_of::<DmaHeapAllocationData>()];
const _: [(); 8] = [(); size_of::<DmaBufSync>()];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinuxDmaBufError {
    InvalidArgument,
    ResourceLimit,
    OutOfMemory,
    DeviceLost,
    InvalidResponse,
    Os(rustix::io::Errno),
    System(i32),
}

fn ioctl_error(error: rustix::io::Errno) -> LinuxDmaBufError {
    match error {
        rustix::io::Errno::NOMEM => LinuxDmaBufError::OutOfMemory,
        rustix::io::Errno::NODEV | rustix::io::Errno::BADFD => LinuxDmaBufError::DeviceLost,
        _ => LinuxDmaBufError::Os(error),
    }
}

fn access_flags(access: AccessMode) -> u64 {
    match access {
        AccessMode::Read => DMA_BUF_SYNC_READ,
        AccessMode::Write => DMA_BUF_SYNC_WRITE,
        AccessMode::ReadWrite => DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LinuxDmaBufInfo {
    pub identity: AllocationIdentity,
    pub bytes: u64,
    pub alignment: u64,
}

pub(crate) struct LinuxDmaBuf {
    file: LinuxFile,
    info: LinuxDmaBufInfo,
}

impl LinuxDmaBuf {
    pub(crate) fn allocate_system(bytes: u64, alignment: u64) -> Result<Self, LinuxDmaBufError> {
        let page = rustix::param::page_size() as u64;
        if bytes == 0
            || bytes > usize::MAX as u64
            || alignment == 0
            || !alignment.is_power_of_two()
            || alignment > page
        {
            return Err(LinuxDmaBufError::InvalidArgument);
        }
        let allocated = bytes
            .checked_add(page - 1)
            .map(|value| value & !(page - 1))
            .ok_or(LinuxDmaBufError::ResourceLimit)?;
        let heap = open(DMA_HEAP_PATH, OFlags::RDWR | OFlags::CLOEXEC, Mode::empty())
            .map_err(ioctl_error)?;
        let mut request = DmaHeapAllocationData {
            len: allocated,
            fd: 0,
            fd_flags: (OFlags::RDWR | OFlags::CLOEXEC).bits(),
            heap_flags: 0,
        };
        // SAFETY: request number and layout match Linux dma-heap UAPI.
        unsafe { update::<DMA_HEAP_IOCTL_ALLOC, _>(heap.as_fd(), &mut request) }
            .map_err(ioctl_error)?;
        if request.fd == 0 || request.len != allocated {
            return Err(LinuxDmaBufError::InvalidResponse);
        }
        // SAFETY: successful allocation returns one fresh owned descriptor.
        let fd = unsafe { OwnedFd::from_raw_fd(request.fd as i32) };
        Self::from_owned_fd(fd, allocated, page)
    }

    pub(crate) fn from_owned_fd(
        fd: OwnedFd,
        bytes: u64,
        alignment: u64,
    ) -> Result<Self, LinuxDmaBufError> {
        if bytes == 0 || bytes > usize::MAX as u64 || alignment == 0 {
            return Err(LinuxDmaBufError::InvalidArgument);
        }
        let stat = fstat(&fd).map_err(ioctl_error)?;
        Ok(Self {
            file: LinuxFile { fd },
            info: LinuxDmaBufInfo {
                identity: AllocationIdentity {
                    provider: PROVIDER_LINUX_DMABUF,
                    object: stat.st_ino,
                    generation: stat.st_dev,
                },
                bytes,
                alignment,
            },
        })
    }

    pub(crate) fn info(&self) -> LinuxDmaBufInfo {
        self.info
    }

    pub(crate) fn as_fd(&self) -> rustix::fd::BorrowedFd<'_> {
        self.file.as_fd()
    }

    pub(crate) fn duplicate(&self) -> Result<Self, LinuxDmaBufError> {
        let fd = rustix::io::fcntl_dupfd_cloexec(&self.file.fd, 3).map_err(ioctl_error)?;
        Self::from_owned_fd(fd, self.info.bytes, self.info.alignment)
    }

    pub(crate) fn begin_cpu_access(&self, access: AccessMode) -> Result<(), LinuxDmaBufError> {
        self.sync(access_flags(access))
    }

    pub(crate) fn end_cpu_access(&self, access: AccessMode) -> Result<(), LinuxDmaBufError> {
        self.sync(access_flags(access) | DMA_BUF_SYNC_END)
    }

    pub(crate) fn map(&self, access: AccessMode) -> Result<LinuxDmaBufMapping, LinuxDmaBufError> {
        let sync_file = self.duplicate()?;
        sync_file.begin_cpu_access(access)?;
        let mapping = match LinuxFiles.map_file(
            &sync_file.file,
            MapOptions {
                offset: 0,
                bytes: self.info.bytes as usize,
                readable: access.reads(),
                writable: access.writes(),
                shared: true,
            },
        ) {
            Ok(mapping) => mapping,
            Err(error) => {
                let _ = sync_file.end_cpu_access(access);
                return Err(LinuxDmaBufError::System(error.0));
            }
        };
        Ok(LinuxDmaBufMapping {
            inner: MappingOwner::new(mapping, sync_file),
            access,
        })
    }

    fn sync(&self, flags: u64) -> Result<(), LinuxDmaBufError> {
        // SAFETY: request number and 8-byte layout match Linux dma-buf UAPI.
        unsafe { set::<DMA_BUF_IOCTL_SYNC, _>(self.file.as_fd(), DmaBufSync { flags }) }
            .map_err(ioctl_error)
    }
}

pub(crate) struct LinuxDmaBufMapping {
    inner: MappingOwner<LinuxMapping, LinuxDmaBuf>,
    access: AccessMode,
}

impl MemoryMapping for LinuxDmaBufMapping {
    fn address(&self) -> core::ptr::NonNull<u8> {
        self.inner.address()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl SharedMapping for LinuxDmaBufMapping {
    fn address(&self) -> core::ptr::NonNull<u8> {
        self.inner.address()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn access(&self) -> AccessMode {
        self.access
    }
}

impl Drop for LinuxDmaBufMapping {
    fn drop(&mut self) {
        let _ = self.inner.owner().end_cpu_access(self.access);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dma_buf_uapi_numbers_match_headers() {
        assert_eq!(DMA_HEAP_IOCTL_ALLOC, 0xc018_4800);
        assert_eq!(DMA_BUF_IOCTL_SYNC, 0x4008_6200);
    }
}
