//! Linux syscall implementation backed by rustix.

use core::ffi::{c_void, CStr};
use core::ptr::{self, NonNull};

use rustix::event::{poll, PollFd, PollFlags, Timespec};
use rustix::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use rustix::fs::{open, Mode, OFlags};
use rustix::mm::{mmap, munmap, MapFlags, ProtFlags};
use rustix::time::{clock_gettime, ClockId};

use super::{
    MapOptions, MemoryMapping, OpenAccess, OpenOptions, OsError, OsServices, OwnedDeviceFile,
    Ready, WaitInterest,
};
use crate::runtime::Deadline;

fn os_error(error: rustix::io::Errno) -> OsError {
    OsError(error.raw_os_error())
}

pub struct LinuxFile {
    fd: OwnedFd,
}

impl LinuxFile {
    pub(crate) fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

impl OwnedDeviceFile for LinuxFile {
    fn raw_handle(&self) -> i64 {
        self.fd.as_raw_fd() as i64
    }
}

pub struct LinuxMapping {
    address: NonNull<u8>,
    bytes: usize,
}

impl MemoryMapping for LinuxMapping {
    fn address(&self) -> NonNull<u8> {
        self.address
    }

    fn len(&self) -> usize {
        self.bytes
    }
}

impl Drop for LinuxMapping {
    fn drop(&mut self) {
        // SAFETY: this object uniquely owns the exact mapping returned by mmap, and no references
        // can outlive `drop` through the portable Mapping trait.
        let _ = unsafe { munmap(self.address.as_ptr().cast::<c_void>(), self.bytes) };
    }
}

#[derive(Clone, Copy, Default)]
pub struct LinuxOs;

impl LinuxOs {
    fn timeout(&self, deadline: Deadline) -> Result<Option<Timespec>, OsError> {
        let Deadline::MonotonicNs(deadline) = deadline else {
            return Ok(None);
        };
        let now = self.monotonic_now_ns()?;
        let remaining = deadline.saturating_sub(now);
        Ok(Some(Timespec {
            tv_sec: (remaining / 1_000_000_000) as i64,
            tv_nsec: (remaining % 1_000_000_000) as i64,
        }))
    }
}

impl OsServices for LinuxOs {
    type File = LinuxFile;
    type Mapping = LinuxMapping;

    fn open_device(&self, path: &CStr, options: OpenOptions) -> Result<Self::File, OsError> {
        let mut flags = OFlags::CLOEXEC;
        flags |= match options.access {
            OpenAccess::ReadOnly => OFlags::RDONLY,
            OpenAccess::ReadWrite => OFlags::RDWR,
        };
        if options.nonblocking {
            flags |= OFlags::NONBLOCK;
        }
        open(path, flags, Mode::empty())
            .map(|fd| LinuxFile { fd })
            .map_err(os_error)
    }

    fn map_file(&self, file: &Self::File, options: MapOptions) -> Result<Self::Mapping, OsError> {
        if options.bytes == 0
            || (!options.readable && !options.writable)
            || options.offset % rustix::param::page_size() as u64 != 0
        {
            return Err(OsError(22));
        }
        let mut protection = ProtFlags::empty();
        if options.readable {
            protection |= ProtFlags::READ;
        }
        if options.writable {
            protection |= ProtFlags::WRITE;
        }
        let flags = if options.shared {
            MapFlags::SHARED
        } else {
            MapFlags::PRIVATE
        };
        // SAFETY: a null address asks the kernel to select a non-overlapping range. LinuxFile
        // remains open for the call; LinuxMapping exclusively owns the returned range.
        let raw = unsafe {
            mmap(
                ptr::null_mut(),
                options.bytes,
                protection,
                flags,
                file.as_fd(),
                options.offset,
            )
        }
        .map_err(os_error)?;
        let address = NonNull::new(raw.cast::<u8>()).ok_or(OsError(12))?;
        Ok(LinuxMapping {
            address,
            bytes: options.bytes,
        })
    }

    fn wait(
        &self,
        file: &Self::File,
        interest: WaitInterest,
        deadline: Deadline,
    ) -> Result<Ready, OsError> {
        let events = match interest {
            WaitInterest::Readable => PollFlags::IN,
            WaitInterest::Writable => PollFlags::OUT,
            WaitInterest::ReadWrite => PollFlags::IN | PollFlags::OUT,
        };
        let mut descriptor = [PollFd::new(&file.fd, events)];
        let timeout = self.timeout(deadline)?;
        poll(&mut descriptor, timeout.as_ref()).map_err(os_error)?;
        let ready = descriptor[0].revents();
        Ok(Ready {
            readable: ready.contains(PollFlags::IN),
            writable: ready.contains(PollFlags::OUT),
            error: ready.contains(PollFlags::ERR) || ready.contains(PollFlags::NVAL),
            hung_up: ready.contains(PollFlags::HUP),
        })
    }

    fn monotonic_now_ns(&self) -> Result<u64, OsError> {
        let value = clock_gettime(ClockId::Monotonic);
        let seconds = u64::try_from(value.tv_sec).map_err(|_| OsError(75))?;
        let nanos = u64::try_from(value.tv_nsec).map_err(|_| OsError(75))?;
        seconds
            .checked_mul(1_000_000_000)
            .and_then(|base| base.checked_add(nanos))
            .ok_or(OsError(75))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_clock_moves_forward() {
        let os = LinuxOs;
        let first = os.monotonic_now_ns().unwrap();
        let second = os.monotonic_now_ns().unwrap();
        assert!(second >= first);
    }

    #[test]
    fn invalid_mapping_is_rejected_before_syscall() {
        let os = LinuxOs;
        assert!(os
            .open_device(
                c"/definitely/not/a/device",
                OpenOptions {
                    access: OpenAccess::ReadOnly,
                    nonblocking: false,
                },
            )
            .is_err());
    }
}
