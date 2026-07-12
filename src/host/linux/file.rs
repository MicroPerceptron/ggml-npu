//! Linux owned files, mappings, polling, and monotonic deadline conversion.

use core::ffi::{c_void, CStr};
use core::ptr::{self, NonNull};

use rustix::event::{poll, PollFd, PollFlags, Timespec};
use rustix::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use rustix::fs::{open, Mode, OFlags};
use rustix::mm::{mmap, munmap, MapFlags, ProtFlags};
use rustix::time::{clock_gettime, ClockId};

use super::super::{
    FileError, FileService, MapOptions, MemoryMapping, OpenAccess, OpenOptions, OwnedFile, Ready,
    WaitInterest,
};
use crate::runtime::Deadline;

fn file_error(error: rustix::io::Errno) -> FileError {
    FileError(error.raw_os_error())
}

pub(crate) struct LinuxFile {
    pub(super) fd: OwnedFd,
}

impl LinuxFile {
    pub(crate) fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

impl OwnedFile for LinuxFile {
    fn diagnostic_id(&self) -> i64 {
        i64::from(self.fd.as_raw_fd())
    }
}

pub(crate) struct LinuxMapping {
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
        // SAFETY: this object uniquely owns the exact mapping returned by mmap.
        let _ = unsafe { munmap(self.address.as_ptr().cast::<c_void>(), self.bytes) };
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxFiles;

impl LinuxFiles {
    fn timeout(&self, deadline: Deadline) -> Result<Option<Timespec>, FileError> {
        let Deadline::MonotonicNs(deadline) = deadline else {
            return Ok(None);
        };
        let value = clock_gettime(ClockId::Monotonic);
        let seconds = u64::try_from(value.tv_sec).map_err(|_| FileError(75))?;
        let nanos = u64::try_from(value.tv_nsec).map_err(|_| FileError(75))?;
        let now = seconds
            .checked_mul(1_000_000_000)
            .and_then(|base| base.checked_add(nanos))
            .ok_or(FileError(75))?;
        let remaining = deadline.saturating_sub(now);
        Ok(Some(Timespec {
            tv_sec: (remaining / 1_000_000_000) as i64,
            tv_nsec: (remaining % 1_000_000_000) as i64,
        }))
    }
}

impl FileService for LinuxFiles {
    type File = LinuxFile;
    type Mapping = LinuxMapping;

    fn open_device(&self, path: &CStr, options: OpenOptions) -> Result<Self::File, FileError> {
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
            .map_err(file_error)
    }

    fn map_file(&self, file: &Self::File, options: MapOptions) -> Result<Self::Mapping, FileError> {
        if options.bytes == 0
            || (!options.readable && !options.writable)
            || !options
                .offset
                .is_multiple_of(rustix::param::page_size() as u64)
        {
            return Err(FileError(22));
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
        // SAFETY: null lets the kernel select a range; the returned owner retains exact length.
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
        .map_err(file_error)?;
        let address = NonNull::new(raw.cast::<u8>()).ok_or(FileError(12))?;
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
    ) -> Result<Ready, FileError> {
        let events = match interest {
            WaitInterest::Readable => PollFlags::IN,
            WaitInterest::Writable => PollFlags::OUT,
            WaitInterest::ReadWrite => PollFlags::IN | PollFlags::OUT,
        };
        let mut descriptor = [PollFd::new(&file.fd, events)];
        let timeout = self.timeout(deadline)?;
        poll(&mut descriptor, timeout.as_ref()).map_err(file_error)?;
        let ready = descriptor[0].revents();
        Ok(Ready {
            readable: ready.contains(PollFlags::IN),
            writable: ready.contains(PollFlags::OUT),
            error: ready.contains(PollFlags::ERR) || ready.contains(PollFlags::NVAL),
            hung_up: ready.contains(PollFlags::HUP),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_device_open_is_reported() {
        assert!(LinuxFiles
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
