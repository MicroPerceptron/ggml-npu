//! In-memory host provider for host-independent runtime tests.

use alloc::vec::Vec;
use core::alloc::Layout;
use core::ffi::{c_void, CStr};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use std::alloc::{GlobalAlloc, System};

use super::{
    async_services::{CallerDrivenBlockingFactory, ReactorFactory},
    AllocationService, ArtifactService, ClockService, ConfigService, DiagnosticService,
    DynamicLibraryService, ExitCallback, FileError, FileService, HostEnvironment, HostError,
    LibraryVisibility, MapOptions, MappingOwner, MathService, MemoryMapping, MemoryStatusService,
    OpenOptions, OwnedFile, Ready, WaitInterest,
};
use crate::runtime::async_runtime::{AtomicWait, NoProgress, WaitChange, WaitError, WakeCount};
use crate::runtime::Deadline;

pub(crate) struct MockFile(i64);

impl OwnedFile for MockFile {
    fn diagnostic_id(&self) -> i64 {
        self.0
    }
}

pub(crate) struct MockMapping([u8; 64]);

impl MemoryMapping for MockMapping {
    fn address(&self) -> NonNull<u8> {
        NonNull::from(&self.0).cast()
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct MockFiles;

impl FileService for MockFiles {
    type File = MockFile;
    type Mapping = MockMapping;

    fn open_device(&self, path: &CStr, _options: OpenOptions) -> Result<Self::File, FileError> {
        (!path.to_bytes().is_empty())
            .then_some(MockFile(7))
            .ok_or(FileError(22))
    }

    fn map_file(
        &self,
        _file: &Self::File,
        options: MapOptions,
    ) -> Result<Self::Mapping, FileError> {
        (options.bytes <= 64)
            .then_some(MockMapping([0; 64]))
            .ok_or(FileError(12))
    }

    fn wait(
        &self,
        _file: &Self::File,
        _interest: WaitInterest,
        deadline: Deadline,
    ) -> Result<Ready, FileError> {
        Ok(Ready {
            readable: !deadline.expired_at(10),
            writable: false,
            error: false,
            hung_up: false,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MockClock(pub u64);

impl ClockService for MockClock {
    fn monotonic_now_ns(&self) -> Result<u64, HostError> {
        Ok(self.0)
    }
}

impl crate::runtime::async_runtime::MonotonicClock for MockClock {
    fn now_ns(&self) -> Result<u64, WaitError> {
        Ok(self.0)
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct MockAtomicWait;

impl AtomicWait for MockAtomicWait {
    fn wait_u32(
        &self,
        state: &core::sync::atomic::AtomicU32,
        expected: u32,
        deadline: Deadline,
    ) -> Result<WaitChange, WaitError> {
        if state.load(Ordering::Acquire) != expected {
            Ok(WaitChange::Changed)
        } else if matches!(deadline, Deadline::MonotonicNs(_)) {
            Ok(WaitChange::TimedOut)
        } else {
            Ok(WaitChange::Spurious)
        }
    }

    fn wake_u32(
        &self,
        _state: &core::sync::atomic::AtomicU32,
        _count: WakeCount,
    ) -> Result<(), WaitError> {
        Ok(())
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct MockReactorFactory;

impl ReactorFactory for MockReactorFactory {
    type Reactor = NoProgress;

    fn create(&self) -> Result<Self::Reactor, WaitError> {
        Ok(NoProgress)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MockMemoryStatus(pub u64);

impl MemoryStatusService for MockMemoryStatus {
    fn available_memory_bytes(&self) -> Result<u64, HostError> {
        Ok(self.0)
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct MockMath;

impl MathService for MockMath {
    fn round_f32(&self, value: f32) -> f32 {
        value.round()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MockArtifacts(pub &'static [u8]);

impl ArtifactService for MockArtifacts {
    fn load(&self, _location: &CStr, max_bytes: usize) -> Result<Vec<u8>, HostError> {
        if self.0.len() > max_bytes {
            return Err(HostError(27));
        }
        Ok(self.0.to_vec())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MockConfig {
    pub name: &'static CStr,
    pub value: &'static CStr,
}

impl ConfigService for MockConfig {
    fn value(&self, name: &CStr) -> Option<&'static CStr> {
        (name == self.name).then_some(self.value)
    }
}

#[derive(Default)]
pub(crate) struct MockDiagnostics {
    pub writes: AtomicUsize,
    pub registrations: AtomicUsize,
}

impl DiagnosticService for MockDiagnostics {
    fn write_stderr(&self, bytes: &[u8]) {
        self.writes.fetch_add(bytes.len(), Ordering::Relaxed);
    }

    fn register_process_exit(&self, _callback: ExitCallback, _argument: *mut c_void) -> bool {
        self.registrations.fetch_add(1, Ordering::Relaxed);
        true
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct MockAllocator;

impl AllocationService for MockAllocator {
    unsafe fn allocate(&self, layout: Layout) -> *mut u8 {
        unsafe { System.alloc(layout) }
    }

    unsafe fn deallocate(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    fn abort(&self) -> ! {
        panic!("mock host abort")
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct MockLibraries;

impl DynamicLibraryService for MockLibraries {
    type Library = ();

    fn open(
        &self,
        _path: &CStr,
        _visibility: LibraryVisibility,
    ) -> Result<Self::Library, HostError> {
        Err(HostError(2))
    }

    unsafe fn symbol(
        &self,
        _library: &Self::Library,
        _name: &CStr,
    ) -> Result<*mut c_void, HostError> {
        Err(HostError(2))
    }

    fn last_error(&self) -> Option<&'static CStr> {
        Some(c"mock dynamic loading unavailable")
    }
}

pub(crate) struct MockHost {
    pub files: MockFiles,
    pub clock: MockClock,
    pub memory: MockMemoryStatus,
    pub math: MockMath,
    pub artifacts: MockArtifacts,
    pub config: MockConfig,
    pub diagnostics: MockDiagnostics,
    pub allocator: MockAllocator,
    pub libraries: MockLibraries,
    pub atomic_wait: MockAtomicWait,
    pub reactor_factory: MockReactorFactory,
    pub blocking_executor_factory: CallerDrivenBlockingFactory,
}

impl HostEnvironment for MockHost {
    type Files = MockFiles;
    type Clock = MockClock;
    type Memory = MockMemoryStatus;
    type Math = MockMath;
    type Artifacts = MockArtifacts;
    type Config = MockConfig;
    type Diagnostics = MockDiagnostics;
    type Allocator = MockAllocator;
    type Libraries = MockLibraries;
    type AtomicWait = MockAtomicWait;
    type ReactorFactory = MockReactorFactory;
    type BlockingExecutorFactory = CallerDrivenBlockingFactory;

    fn files(&self) -> &Self::Files {
        &self.files
    }

    fn clock(&self) -> &Self::Clock {
        &self.clock
    }

    fn memory(&self) -> &Self::Memory {
        &self.memory
    }

    fn math(&self) -> &Self::Math {
        &self.math
    }

    fn artifacts(&self) -> &Self::Artifacts {
        &self.artifacts
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn diagnostics(&self) -> &Self::Diagnostics {
        &self.diagnostics
    }

    fn allocator(&self) -> &Self::Allocator {
        &self.allocator
    }

    fn libraries(&self) -> &Self::Libraries {
        &self.libraries
    }

    fn atomic_wait(&self) -> &Self::AtomicWait {
        &self.atomic_wait
    }

    fn reactor_factory(&self) -> &Self::ReactorFactory {
        &self.reactor_factory
    }

    fn blocking_executor_factory(&self) -> &Self::BlockingExecutorFactory {
        &self.blocking_executor_factory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static OWNER_DROPS: AtomicUsize = AtomicUsize::new(0);

    struct DropOwner;
    impl Drop for DropOwner {
        fn drop(&mut self) {
            OWNER_DROPS.fetch_add(1, Ordering::Relaxed);
        }
    }

    extern "C" fn exit_callback(_argument: *mut c_void) {}

    #[test]
    fn mock_host_composes_focused_services() {
        let host = MockHost {
            files: MockFiles,
            clock: MockClock(42),
            memory: MockMemoryStatus(1024),
            math: MockMath,
            artifacts: MockArtifacts(b"artifact"),
            config: MockConfig {
                name: c"SETTING",
                value: c"value",
            },
            diagnostics: MockDiagnostics::default(),
            allocator: MockAllocator,
            libraries: MockLibraries,
            atomic_wait: MockAtomicWait,
            reactor_factory: MockReactorFactory,
            blocking_executor_factory: CallerDrivenBlockingFactory,
        };

        assert_eq!(host.clock().monotonic_now_ns(), Ok(42));
        assert_eq!(host.memory().available_memory_bytes(), Ok(1024));
        assert_eq!(host.math().round_f32(1.6), 2.0);
        assert_eq!(host.artifacts().load(c"ignored", 8).unwrap(), b"artifact");
        assert!(host.config().is_nonempty(c"SETTING"));
        assert!(!host.config().is_set(c"MISSING"));
        host.diagnostics().write_stderr(b"abc");
        assert!(host
            .diagnostics()
            .register_process_exit(exit_callback, core::ptr::null_mut()));
        assert_eq!(host.diagnostics.writes.load(Ordering::Relaxed), 3);
        assert_eq!(host.diagnostics.registrations.load(Ordering::Relaxed), 1);
        assert!(host
            .libraries()
            .open(c"missing", LibraryVisibility::Local)
            .is_err());
    }

    #[test]
    fn mapping_owner_retains_resource_until_mapping_drop() {
        OWNER_DROPS.store(0, Ordering::Relaxed);
        let mapping = MappingOwner::new(MockMapping([0; 64]), DropOwner);
        assert_eq!(mapping.len(), 64);
        assert_eq!(OWNER_DROPS.load(Ordering::Relaxed), 0);
        drop(mapping);
        assert_eq!(OWNER_DROPS.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn generic_modules_do_not_bypass_host_services() {
        let generic_sources = [
            ("address", include_str!("../address.rs")),
            ("hal", include_str!("../hal.rs")),
            ("interop", include_str!("../interop.rs")),
            ("metrics", include_str!("../metrics.rs")),
            ("identity", include_str!("../identity.rs")),
            ("runtime", include_str!("../runtime/mod.rs")),
            ("artifact", include_str!("../runtime/artifact.rs")),
            ("graph", include_str!("../graph.rs")),
        ];
        let forbidden = [
            "target_os",
            "Linux",
            "RawFd",
            "OwnedFd",
            "BorrowedFd",
            "getenv",
            "dlopen",
            "dlsym",
            "posix_memalign",
            "ioctl",
            "/proc/",
            "/dev/",
        ];
        for (module, source) in generic_sources {
            for term in forbidden {
                assert!(
                    !source.contains(term),
                    "generic module {module} bypasses host boundary with {term}"
                );
            }
        }
    }
}
