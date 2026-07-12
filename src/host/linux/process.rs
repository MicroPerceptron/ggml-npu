//! Linux process, loader, clock, memory-status, math, and artifact services.

use alloc::vec::Vec;
use core::alloc::Layout;
use core::ffi::{c_char, c_int, c_void, CStr};
use core::ptr;

use rustix::fd::BorrowedFd;
use rustix::fs::{fstat, open, Mode, OFlags};
use rustix::io::read;
use rustix::time::{clock_gettime, ClockId};

use super::super::{
    AllocationService, ArtifactService, ClockService, ConfigService, DiagnosticService,
    DynamicLibraryService, ExitCallback, HostError, LibraryVisibility, MathService,
    MemoryStatusService,
};

unsafe extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn free(pointer: *mut c_void);
    fn posix_memalign(result: *mut *mut c_void, alignment: usize, size: usize) -> c_int;
    fn abort() -> !;
    fn getenv(name: *const c_char) -> *const c_char;
    fn __cxa_atexit(
        callback: ExitCallback,
        argument: *mut c_void,
        dso_handle: *mut c_void,
    ) -> c_int;
    static mut __dso_handle: *mut c_void;
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn roundf(value: f32) -> f32;
}

const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;
const RTLD_GLOBAL: c_int = 0x100;

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxClock;

impl ClockService for LinuxClock {
    fn monotonic_now_ns(&self) -> Result<u64, HostError> {
        let value = clock_gettime(ClockId::Monotonic);
        let seconds = u64::try_from(value.tv_sec).map_err(|_| HostError(75))?;
        let nanos = u64::try_from(value.tv_nsec).map_err(|_| HostError(75))?;
        seconds
            .checked_mul(1_000_000_000)
            .and_then(|base| base.checked_add(nanos))
            .ok_or(HostError(75))
    }
}

impl crate::runtime::async_runtime::MonotonicClock for LinuxClock {
    fn now_ns(&self) -> Result<u64, crate::runtime::async_runtime::WaitError> {
        self.monotonic_now_ns().map_err(|error| {
            crate::runtime::async_runtime::WaitError::HostFailure {
                domain: super::ERROR_DOMAIN,
                code: error.0,
            }
        })
    }
}

impl crate::artifact_cache::CacheClock for LinuxClock {
    fn now_ns(&self) -> u64 {
        self.monotonic_now_ns().unwrap_or(0)
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxMemoryStatus;

fn read_decimal_file(path: &CStr) -> Result<Option<u64>, HostError> {
    let file = open(path, OFlags::RDONLY | OFlags::CLOEXEC, Mode::empty())
        .map_err(|error| HostError(i64::from(error.raw_os_error())))?;
    let mut buffer = [0u8; 64];
    let length =
        read(&file, &mut buffer).map_err(|error| HostError(i64::from(error.raw_os_error())))?;
    let bytes = buffer[..length]
        .split(|byte| byte.is_ascii_whitespace())
        .next()
        .ok_or(HostError(22))?;
    if bytes == b"max" {
        return Ok(None);
    }
    let text = core::str::from_utf8(bytes).map_err(|_| HostError(84))?;
    text.parse::<u64>().map(Some).map_err(|_| HostError(22))
}

impl MemoryStatusService for LinuxMemoryStatus {
    fn available_memory_bytes(&self) -> Result<u64, HostError> {
        let file = open(
            c"/proc/meminfo",
            OFlags::RDONLY | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|error| HostError(i64::from(error.raw_os_error())))?;
        let mut buffer = [0u8; 4096];
        let length =
            read(&file, &mut buffer).map_err(|error| HostError(i64::from(error.raw_os_error())))?;
        let text = core::str::from_utf8(&buffer[..length]).map_err(|_| HostError(84))?;
        let line = text
            .split('\n')
            .find(|line| line.starts_with("MemAvailable:"))
            .ok_or(HostError(2))?;
        let kib = line
            .split_ascii_whitespace()
            .nth(1)
            .ok_or(HostError(22))?
            .parse::<u64>()
            .map_err(|_| HostError(22))?;
        kib.checked_mul(1024).ok_or(HostError(75))
    }

    fn constrained_available_memory_bytes(&self) -> Result<u64, HostError> {
        let available = self.available_memory_bytes()?;
        let limit = match read_decimal_file(c"/sys/fs/cgroup/memory.max") {
            Ok(Some(limit)) => limit,
            Ok(None) | Err(_) => return Ok(available),
        };
        let current = read_decimal_file(c"/sys/fs/cgroup/memory.current")
            .ok()
            .flatten()
            .unwrap_or(0);
        Ok(available.min(limit.saturating_sub(current)))
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxMath;

impl MathService for LinuxMath {
    fn round_f32(&self, value: f32) -> f32 {
        // SAFETY: roundf has no pointer or ownership preconditions.
        unsafe { roundf(value) }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxArtifacts;

impl ArtifactService for LinuxArtifacts {
    fn load(&self, location: &CStr, max_bytes: usize) -> Result<Vec<u8>, HostError> {
        let file = open(location, OFlags::RDONLY | OFlags::CLOEXEC, Mode::empty())
            .map_err(|error| HostError(i64::from(error.raw_os_error())))?;
        let metadata = fstat(&file).map_err(|error| HostError(i64::from(error.raw_os_error())))?;
        let length = usize::try_from(metadata.st_size).map_err(|_| HostError(75))?;
        if length > max_bytes {
            return Err(HostError(27));
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(|_| HostError(12))?;
        bytes.resize(length, 0);
        let mut filled = 0;
        while filled < length {
            let read_bytes = read(&file, &mut bytes[filled..])
                .map_err(|error| HostError(i64::from(error.raw_os_error())))?;
            if read_bytes == 0 {
                return Err(HostError(5));
            }
            filled += read_bytes;
        }
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxConfig;

impl ConfigService for LinuxConfig {
    fn value(&self, name: &CStr) -> Option<&'static CStr> {
        // SAFETY: this backend never mutates the process environment after startup.
        let pointer = unsafe { getenv(name.as_ptr()) };
        (!pointer.is_null()).then(|| unsafe { CStr::from_ptr(pointer) })
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxDiagnostics;

impl DiagnosticService for LinuxDiagnostics {
    fn write_stderr(&self, bytes: &[u8]) {
        // SAFETY: descriptor 2 is borrowed only for this synchronous write.
        let stderr = unsafe { BorrowedFd::borrow_raw(2) };
        let _ = rustix::io::write(stderr, bytes);
    }

    fn register_process_exit(&self, callback: ExitCallback, argument: *mut c_void) -> bool {
        // SAFETY: tie the callback to this shared object so dlclose runs it before unmapping code.
        unsafe {
            __cxa_atexit(
                callback,
                argument,
                core::ptr::addr_of_mut!(__dso_handle).cast(),
            ) == 0
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxAllocator;

impl AllocationService for LinuxAllocator {
    unsafe fn allocate(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= 16 {
            return unsafe { malloc(layout.size()) }.cast();
        }
        let mut pointer = ptr::null_mut();
        if unsafe { posix_memalign(&mut pointer, layout.align(), layout.size()) } == 0 {
            pointer.cast()
        } else {
            ptr::null_mut()
        }
    }

    unsafe fn deallocate(&self, pointer: *mut u8, _layout: Layout) {
        unsafe { free(pointer.cast()) };
    }

    fn abort(&self) -> ! {
        unsafe { abort() }
    }
}

pub(crate) struct LinuxLibrary {
    handle: *mut c_void,
}

unsafe impl Send for LinuxLibrary {}
unsafe impl Sync for LinuxLibrary {}

impl Drop for LinuxLibrary {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = unsafe { dlclose(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxLibraries;

impl DynamicLibraryService for LinuxLibraries {
    type Library = LinuxLibrary;

    fn open(&self, path: &CStr, visibility: LibraryVisibility) -> Result<Self::Library, HostError> {
        let visibility = match visibility {
            LibraryVisibility::Local => RTLD_LOCAL,
            LibraryVisibility::Process => RTLD_GLOBAL,
        };
        unsafe { dlerror() };
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | visibility) };
        if handle.is_null() {
            Err(HostError(2))
        } else {
            Ok(LinuxLibrary { handle })
        }
    }

    unsafe fn symbol(
        &self,
        library: &Self::Library,
        name: &CStr,
    ) -> Result<*mut c_void, HostError> {
        unsafe { dlerror() };
        let pointer = unsafe { dlsym(library.handle, name.as_ptr()) };
        if pointer.is_null() {
            Err(HostError(2))
        } else {
            Ok(pointer)
        }
    }

    fn last_error(&self) -> Option<&'static CStr> {
        let pointer = unsafe { dlerror() };
        (!pointer.is_null()).then(|| unsafe { CStr::from_ptr(pointer) })
    }
}
