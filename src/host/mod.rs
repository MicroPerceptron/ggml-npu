//! Host service composition and centralized built-in platform selection.
//!
//! Focused traits keep process, clock, dynamic-loader, and file/mapping semantics independent.
//! Runtime code may accept any [`HostEnvironment`]; production uses [`builtin`].

#![allow(dead_code)]

use alloc::vec::Vec;
use core::alloc::Layout;
use core::ffi::{c_void, CStr};
use core::ptr::NonNull;

use crate::artifact_cache::{ArtifactStore, CompressionProvider, WaitProvider};
use crate::runtime::Deadline;

pub(crate) mod artifact_cache;

pub(crate) const ERROR_DOMAIN: u32 = u32::from_be_bytes(*b"HOST");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostError(pub i64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FileError(pub i32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpenAccess {
    ReadOnly,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OpenOptions {
    pub access: OpenAccess,
    pub nonblocking: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MapOptions {
    pub offset: u64,
    pub bytes: usize,
    pub readable: bool,
    pub writable: bool,
    pub shared: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WaitInterest {
    Readable,
    Writable,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ready {
    pub readable: bool,
    pub writable: bool,
    pub error: bool,
    pub hung_up: bool,
}

pub(crate) trait OwnedFile {
    fn diagnostic_id(&self) -> i64;
}

pub(crate) trait MemoryMapping {
    fn address(&self) -> NonNull<u8>;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Couples a mapping with an arbitrary resource that must outlive it.
pub(crate) struct MappingOwner<M, O> {
    mapping: M,
    owner: O,
}

impl<M, O> MappingOwner<M, O> {
    pub const fn new(mapping: M, owner: O) -> Self {
        Self { mapping, owner }
    }

    pub const fn mapping(&self) -> &M {
        &self.mapping
    }

    pub const fn owner(&self) -> &O {
        &self.owner
    }
}

impl<M: MemoryMapping, O> MemoryMapping for MappingOwner<M, O> {
    fn address(&self) -> NonNull<u8> {
        self.mapping.address()
    }

    fn len(&self) -> usize {
        self.mapping.len()
    }
}

pub(crate) trait FileService {
    type File: OwnedFile;
    type Mapping: MemoryMapping;

    fn open_device(&self, path: &CStr, options: OpenOptions) -> Result<Self::File, FileError>;
    fn map_file(&self, file: &Self::File, options: MapOptions) -> Result<Self::Mapping, FileError>;
    fn wait(
        &self,
        file: &Self::File,
        interest: WaitInterest,
        deadline: Deadline,
    ) -> Result<Ready, FileError>;
}

pub(crate) trait ClockService {
    fn monotonic_now_ns(&self) -> Result<u64, HostError>;
}

pub(crate) trait MemoryStatusService {
    fn available_memory_bytes(&self) -> Result<u64, HostError>;

    fn constrained_available_memory_bytes(&self) -> Result<u64, HostError> {
        self.available_memory_bytes()
    }
}

pub(crate) trait MathService {
    fn round_f32(&self, value: f32) -> f32;
}

pub(crate) trait ArtifactService {
    fn load(&self, location: &CStr, max_bytes: usize) -> Result<Vec<u8>, HostError>;
}

pub(crate) trait ConfigService {
    fn value(&self, name: &CStr) -> Option<&'static CStr>;

    fn is_set(&self, name: &CStr) -> bool {
        self.value(name).is_some()
    }

    fn is_nonempty(&self, name: &CStr) -> bool {
        self.value(name)
            .is_some_and(|value| !value.to_bytes().is_empty())
    }
}

pub(crate) type ExitCallback = extern "C" fn(*mut c_void);

pub(crate) trait DiagnosticService {
    fn write_stderr(&self, bytes: &[u8]);
    fn register_process_exit(&self, callback: ExitCallback, argument: *mut c_void) -> bool;
}

pub(crate) trait AllocationService {
    /// Allocate according to `layout`, returning null on failure.
    ///
    /// # Safety
    ///
    /// The caller must deallocate a successful result through the same service and layout.
    unsafe fn allocate(&self, layout: Layout) -> *mut u8;

    /// # Safety
    ///
    /// `pointer` must have been returned by this service for `layout` and not yet deallocated.
    unsafe fn deallocate(&self, pointer: *mut u8, layout: Layout);

    fn abort(&self) -> !;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LibraryVisibility {
    Local,
    Process,
}

pub(crate) trait DynamicLibraryService {
    type Library;

    fn open(&self, path: &CStr, visibility: LibraryVisibility) -> Result<Self::Library, HostError>;

    /// Returns a raw symbol pointer whose lifetime is bounded by `library`.
    ///
    /// # Safety
    ///
    /// The caller must validate the symbol's function/data type before using the pointer.
    unsafe fn symbol(&self, library: &Self::Library, name: &CStr)
        -> Result<*mut c_void, HostError>;

    fn last_error(&self) -> Option<&'static CStr>;
}

pub(crate) trait HostEnvironment {
    type Files: FileService;
    type Clock: ClockService;
    type Memory: MemoryStatusService;
    type Math: MathService;
    type Artifacts: ArtifactService;
    type Config: ConfigService;
    type Diagnostics: DiagnosticService;
    type Allocator: AllocationService;
    type Libraries: DynamicLibraryService;
    type Compression: CompressionProvider;
    type ArtifactCache: ArtifactStore;
    type Wait: WaitProvider;

    fn files(&self) -> &Self::Files;
    fn clock(&self) -> &Self::Clock;
    fn memory(&self) -> &Self::Memory;
    fn math(&self) -> &Self::Math;
    fn artifacts(&self) -> &Self::Artifacts;
    fn config(&self) -> &Self::Config;
    fn diagnostics(&self) -> &Self::Diagnostics;
    fn allocator(&self) -> &Self::Allocator;
    fn libraries(&self) -> &Self::Libraries;
    fn compression(&self) -> &Self::Compression;
    fn artifact_cache(&self) -> &Self::ArtifactCache;
    fn wait_provider(&self) -> &Self::Wait;
}

#[cfg(target_os = "linux")]
pub(crate) mod linux;
#[cfg(target_os = "linux")]
pub(crate) use linux::{builtin, BuiltinLibrary};

#[cfg(not(target_os = "linux"))]
compile_error!(
    "ggml-npu has no built-in host provider for this target; add and select a host implementation"
);

#[cfg(test)]
pub(crate) mod mock;
