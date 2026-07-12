//! Linux host adapters for compression, blocking single-flight waits and durable artifact records.

use alloc::boxed::Box;
use alloc::ffi::CString;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};

use rustix::fd::OwnedFd;
use rustix::fs::{
    flock, fstat, fsync, futimens, mkdir, open, openat, renameat, statat, unlinkat, AtFlags, Dir,
    FlockOperation, Mode, OFlags, Timespec, Timestamps, UTIME_NOW, UTIME_OMIT,
};
use rustix::io::{read, write};
use rustix::thread::futex;

use crate::artifact_cache::{
    ArtifactKey, ArtifactStore, CacheError, CodecDescriptor, CompressionProvider, StoreReport,
    WaitProvider,
};
use crate::host::{ConfigService, DynamicLibraryService, HostEnvironment, LibraryVisibility};

use super::process::{LinuxLibraries, LinuxLibrary};

type CompressBound = unsafe extern "C" fn(usize) -> usize;
type Compress = unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize, i32) -> usize;
type Decompress = unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize) -> usize;
type IsError = unsafe extern "C" fn(usize) -> u32;
type VersionNumber = unsafe extern "C" fn() -> u32;

struct CodecApi {
    _library: LinuxLibrary,
    compress_bound: CompressBound,
    compress: Compress,
    decompress: Decompress,
    is_error: IsError,
    version: u32,
}

static CODEC: AtomicPtr<CodecApi> = AtomicPtr::new(ptr::null_mut());
const CODEC_FAILED: *mut CodecApi = ptr::dangling_mut();
const MIN_ZSTD_VERSION: u32 = 10_500;
const MAX_ZSTD_VERSION_EXCLUSIVE: u32 = 20_000;

fn supported_zstd_version(version: u32) -> bool {
    (MIN_ZSTD_VERSION..MAX_ZSTD_VERSION_EXCLUSIVE).contains(&version)
}

unsafe fn symbol<T: Copy>(
    libraries: &LinuxLibraries,
    library: &LinuxLibrary,
    name: &core::ffi::CStr,
) -> Option<T> {
    let pointer = unsafe { libraries.symbol(library, name) }.ok()?;
    // SAFETY: each caller supplies the exact ABI declared by the codec's stable C API.
    Some(unsafe { mem::transmute_copy::<*mut c_void, T>(&pointer) })
}

fn load_codec() -> Option<CodecApi> {
    let libraries = LinuxLibraries;
    let library = libraries
        .open(c"libzstd.so.1", LibraryVisibility::Local)
        .or_else(|_| libraries.open(c"libzstd.so", LibraryVisibility::Local))
        .ok()?;
    let compress_bound = unsafe { symbol(&libraries, &library, c"ZSTD_compressBound") }?;
    let compress = unsafe { symbol(&libraries, &library, c"ZSTD_compress") }?;
    let decompress = unsafe { symbol(&libraries, &library, c"ZSTD_decompress") }?;
    let is_error = unsafe { symbol(&libraries, &library, c"ZSTD_isError") }?;
    let version_number: VersionNumber =
        unsafe { symbol(&libraries, &library, c"ZSTD_versionNumber") }?;
    let version = unsafe { version_number() };
    if !supported_zstd_version(version) {
        return None;
    }
    Some(CodecApi {
        _library: library,
        compress_bound,
        compress,
        decompress,
        is_error,
        version,
    })
}

fn codec() -> Option<&'static CodecApi> {
    let current = CODEC.load(Ordering::Acquire);
    if current == CODEC_FAILED {
        return None;
    }
    if !current.is_null() {
        // SAFETY: the published allocation is retained for process lifetime.
        return Some(unsafe { &*current });
    }
    let Some(candidate) = load_codec() else {
        let _ = CODEC.compare_exchange(
            ptr::null_mut(),
            CODEC_FAILED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        return None;
    };
    let candidate = Box::into_raw(Box::new(candidate));
    match CODEC.compare_exchange(
        ptr::null_mut(),
        candidate,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => Some(unsafe { &*candidate }),
        Err(winner) => {
            // SAFETY: this allocation was never published.
            drop(unsafe { Box::from_raw(candidate) });
            if winner == CODEC_FAILED {
                None
            } else {
                // SAFETY: the winner remains process-resident.
                Some(unsafe { &*winner })
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxCompression;

impl LinuxCompression {
    pub(crate) fn available(&self) -> bool {
        codec().is_some()
    }
}

impl CompressionProvider for LinuxCompression {
    fn descriptor(&self) -> CodecDescriptor {
        CodecDescriptor {
            id: u32::from_le_bytes(*b"ZSTD"),
            version: codec().map_or(0, |api| api.version),
            options: 1,
        }
    }

    fn compress(&self, source: &[u8]) -> Result<Vec<u8>, CacheError> {
        let api = codec().ok_or(CacheError::Compression)?;
        let bound = unsafe { (api.compress_bound)(source.len()) };
        if bound == 0 || bound > crate::artifact_cache::MAX_RECORD_BYTES {
            return Err(CacheError::RecordTooLarge);
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(bound)
            .map_err(|_| CacheError::Allocation)?;
        output.resize(bound, 0);
        let written = unsafe {
            (api.compress)(
                output.as_mut_ptr().cast(),
                output.len(),
                source.as_ptr().cast(),
                source.len(),
                1,
            )
        };
        if unsafe { (api.is_error)(written) } != 0 || written > output.len() {
            return Err(CacheError::Compression);
        }
        output.truncate(written);
        Ok(output)
    }

    fn decompress(&self, source: &[u8], uncompressed_bytes: usize) -> Result<Vec<u8>, CacheError> {
        let api = codec().ok_or(CacheError::Compression)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(uncompressed_bytes)
            .map_err(|_| CacheError::Allocation)?;
        output.resize(uncompressed_bytes, 0);
        let written = unsafe {
            (api.decompress)(
                output.as_mut_ptr().cast(),
                output.len(),
                source.as_ptr().cast(),
                source.len(),
            )
        };
        if unsafe { (api.is_error)(written) } != 0 || written != uncompressed_bytes {
            return Err(CacheError::Compression);
        }
        Ok(output)
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxWait;

impl WaitProvider for LinuxWait {
    fn wait_while(&self, state: &AtomicU32, expected: u32) {
        while state.load(Ordering::Acquire) == expected {
            let _ = futex::wait(state, futex::Flags::PRIVATE, expected, None);
        }
    }

    fn wake_all(&self, state: &AtomicU32) {
        let _ = futex::wake(state, futex::Flags::PRIVATE, u32::MAX);
    }
}

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const DIRECTORY_MODE: Mode = Mode::RWXU
    .union(Mode::RGRP)
    .union(Mode::XGRP)
    .union(Mode::ROTH)
    .union(Mode::XOTH);
const FILE_MODE: Mode = Mode::RUSR.union(Mode::WUSR);

pub(crate) struct LinuxArtifactStore {
    root: Option<CString>,
}

impl LinuxArtifactStore {
    pub(crate) const fn from_environment() -> Self {
        Self { root: None }
    }

    #[cfg(test)]
    fn at(root: CString) -> Self {
        Self { root: Some(root) }
    }

    fn configured_root(&self) -> Result<CString, CacheError> {
        if let Some(root) = &self.root {
            return Ok(root.clone());
        }
        let config = super::builtin().config();
        let base = if let Some(path) = config.value(c"GGML_NPU_CACHE") {
            path.to_bytes()
        } else if let Some(path) = config.value(c"XDG_CACHE_HOME") {
            path.to_bytes()
        } else if let Some(path) = config.value(c"HOME") {
            let mut home = path.to_bytes().to_vec();
            home.extend_from_slice(b"/.cache");
            let base = CString::new(home).map_err(|_| CacheError::Store)?;
            let _ = mkdir(base.as_c_str(), DIRECTORY_MODE);
            return Self::append_component(&base, b"ggml-npu/artifacts-v1");
        } else {
            return Err(CacheError::Store);
        };
        let base = CString::new(base).map_err(|_| CacheError::Store)?;
        Self::append_component(&base, b"artifacts-v1")
    }

    fn append_component(base: &core::ffi::CStr, component: &[u8]) -> Result<CString, CacheError> {
        let mut bytes = base.to_bytes().to_vec();
        if !bytes.ends_with(b"/") {
            bytes.push(b'/');
        }
        bytes.extend_from_slice(component);
        CString::new(bytes).map_err(|_| CacheError::Store)
    }

    fn open_root(&self) -> Result<OwnedFd, CacheError> {
        let root = self.configured_root()?;
        if let Some(parent_end) = root.to_bytes().iter().rposition(|byte| *byte == b'/') {
            let parent =
                CString::new(&root.to_bytes()[..parent_end]).map_err(|_| CacheError::Store)?;
            let _ = mkdir(parent.as_c_str(), DIRECTORY_MODE);
        }
        let _ = mkdir(root.as_c_str(), DIRECTORY_MODE);
        open(
            root.as_c_str(),
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|_| CacheError::Store)
    }

    fn file_name(key: ArtifactKey) -> Result<CString, CacheError> {
        let mut bytes = key.0.hex().into_bytes();
        bytes.extend_from_slice(b".ggac");
        CString::new(bytes).map_err(|_| CacheError::Store)
    }

    fn lock_name(key: ArtifactKey) -> Result<CString, CacheError> {
        let mut bytes = key.0.hex().into_bytes();
        bytes.extend_from_slice(b".lock");
        CString::new(bytes).map_err(|_| CacheError::Store)
    }

    fn lock_named(root: &OwnedFd, name: &core::ffi::CStr) -> Result<OwnedFd, CacheError> {
        let lock = openat(
            root,
            name,
            OFlags::CREATE | OFlags::RDWR | OFlags::CLOEXEC,
            FILE_MODE,
        )
        .map_err(|_| CacheError::Store)?;
        flock(&lock, FlockOperation::LockExclusive).map_err(|_| CacheError::Store)?;
        Ok(lock)
    }

    fn key_lock(root: &OwnedFd, key: ArtifactKey) -> Result<OwnedFd, CacheError> {
        let name = Self::lock_name(key)?;
        Self::lock_named(root, name.as_c_str())
    }

    fn read_all(file: &OwnedFd, max_bytes: usize) -> Result<Vec<u8>, CacheError> {
        let stat = fstat(file).map_err(|_| CacheError::Store)?;
        let length = usize::try_from(stat.st_size).map_err(|_| CacheError::Overflow)?;
        if length > max_bytes {
            return Err(CacheError::RecordTooLarge);
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|_| CacheError::Allocation)?;
        output.resize(length, 0);
        let mut filled = 0;
        while filled < length {
            let bytes = read(file, &mut output[filled..]).map_err(|_| CacheError::Store)?;
            if bytes == 0 {
                return Err(CacheError::LengthMismatch);
            }
            filled += bytes;
        }
        Ok(output)
    }

    fn write_all(file: &OwnedFd, bytes: &[u8]) -> Result<(), CacheError> {
        let mut written = 0;
        while written < bytes.len() {
            let count = write(file, &bytes[written..]).map_err(|_| CacheError::Store)?;
            if count == 0 {
                return Err(CacheError::Store);
            }
            written += count;
        }
        Ok(())
    }

    fn reserve_space(
        root: &OwnedFd,
        target: &core::ffi::CStr,
        incoming: u64,
        budget: u64,
        low_water: u64,
    ) -> Result<StoreReport, CacheError> {
        if budget == 0 || incoming > budget {
            return Err(CacheError::Store);
        }
        struct Candidate {
            name: CString,
            bytes: u64,
            modified_seconds: i64,
            modified_nanos: i64,
        }
        let mut total = 0u64;
        let mut candidates = Vec::new();
        let mut directory = Dir::read_from(root).map_err(|_| CacheError::Store)?;
        for entry in &mut directory {
            let entry = entry.map_err(|_| CacheError::Store)?;
            let name = entry.file_name();
            if name == target
                || !name.to_bytes().ends_with(b".ggac")
                || name.to_bytes().len() != 64 + b".ggac".len()
            {
                continue;
            }
            let stat =
                statat(root, name, AtFlags::SYMLINK_NOFOLLOW).map_err(|_| CacheError::Store)?;
            let bytes = u64::try_from(stat.st_size).map_err(|_| CacheError::Overflow)?;
            total = total.saturating_add(bytes);
            candidates
                .try_reserve(1)
                .map_err(|_| CacheError::Allocation)?;
            candidates.push(Candidate {
                name: CString::new(name.to_bytes()).map_err(|_| CacheError::Store)?,
                bytes,
                modified_seconds: stat.st_mtime as i64,
                modified_nanos: stat.st_mtime_nsec as i64,
            });
        }
        if total.saturating_add(incoming) <= budget {
            return Ok(StoreReport::default());
        }
        candidates.sort_by(|left, right| {
            (
                left.modified_seconds,
                left.modified_nanos,
                left.name.to_bytes(),
            )
                .cmp(&(
                    right.modified_seconds,
                    right.modified_nanos,
                    right.name.to_bytes(),
                ))
        });
        let target_total = low_water.min(budget).saturating_sub(incoming);
        let mut evicted_bytes = 0u64;
        for candidate in candidates {
            if total <= target_total {
                break;
            }
            if unlinkat(root, candidate.name.as_c_str(), AtFlags::empty()).is_ok() {
                total = total.saturating_sub(candidate.bytes);
                evicted_bytes = evicted_bytes.saturating_add(candidate.bytes);
            }
        }
        if total.saturating_add(incoming) > budget {
            Err(CacheError::Store)
        } else {
            Ok(StoreReport { evicted_bytes })
        }
    }
}

impl ArtifactStore for LinuxArtifactStore {
    fn load(&self, key: ArtifactKey, max_bytes: usize) -> Result<Option<Vec<u8>>, CacheError> {
        let root = self.open_root()?;
        let name = Self::file_name(key)?;
        let file = match openat(
            &root,
            name.as_c_str(),
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        ) {
            Ok(file) => file,
            Err(error) if error == rustix::io::Errno::NOENT => return Ok(None),
            Err(_) => return Err(CacheError::Store),
        };
        let bytes = Self::read_all(&file, max_bytes)?;
        let _ = futimens(
            &file,
            &Timestamps {
                last_access: Timespec {
                    tv_sec: 0,
                    tv_nsec: UTIME_OMIT,
                },
                last_modification: Timespec {
                    tv_sec: 0,
                    tv_nsec: UTIME_NOW,
                },
            },
        );
        Ok(Some(bytes))
    }

    fn store(
        &self,
        key: ArtifactKey,
        record: &[u8],
        budget_bytes: u64,
        low_water_bytes: u64,
    ) -> Result<StoreReport, CacheError> {
        let root = self.open_root()?;
        let _key_lock = Self::key_lock(&root, key)?;
        let target = Self::file_name(key)?;
        if openat(
            &root,
            target.as_c_str(),
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        )
        .is_ok()
        {
            return Ok(StoreReport::default());
        }
        // A per-key lock prevents duplicate publication. This directory-wide lock additionally
        // serializes accounting and eviction for distinct keys across processes.
        let _budget_lock = Self::lock_named(&root, c".budget.lock")?;
        let report = Self::reserve_space(
            &root,
            &target,
            record.len() as u64,
            budget_bytes,
            low_water_bytes,
        )?;
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let pid = rustix::process::getpid().as_raw_nonzero().get();
        let temp = CString::new(alloc::format!(".tmp-{pid}-{sequence:016x}"))
            .map_err(|_| CacheError::Store)?;
        let file = openat(
            &root,
            temp.as_c_str(),
            OFlags::CREATE | OFlags::EXCL | OFlags::WRONLY | OFlags::CLOEXEC,
            FILE_MODE,
        )
        .map_err(|_| CacheError::Store)?;
        if let Err(error) = Self::write_all(&file, record).and_then(|()| {
            fsync(&file).map_err(|_| CacheError::Store)?;
            renameat(&root, temp.as_c_str(), &root, target.as_c_str())
                .map_err(|_| CacheError::Store)?;
            fsync(&root).map_err(|_| CacheError::Store)
        }) {
            let _ = unlinkat(&root, temp.as_c_str(), AtFlags::empty());
            return Err(error);
        }
        Ok(report)
    }

    fn quarantine(&self, key: ArtifactKey) -> Result<(), CacheError> {
        let root = self.open_root()?;
        let _lock = Self::key_lock(&root, key)?;
        let target = Self::file_name(key)?;
        match unlinkat(&root, target.as_c_str(), AtFlags::empty()) {
            Ok(()) => {
                fsync(&root).map_err(|_| CacheError::Store)?;
                Ok(())
            }
            Err(error) if error == rustix::io::Errno::NOENT => Ok(()),
            Err(_) => Err(CacheError::Store),
        }
    }
}

// SAFETY: the optional root is immutable after construction and all methods use independent FDs.
unsafe impl Sync for LinuxArtifactStore {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::SemanticDigest;
    use alloc::format;
    use alloc::vec;
    use std::fs;
    use std::os::unix::ffi::OsStrExt;
    use std::process::Command;
    use std::sync::Arc;
    use std::thread;

    fn temporary_store(label: &str) -> (LinuxArtifactStore, CString) {
        let root = std::env::temp_dir().join(format!(
            "ggml-npu-cache-test-{}-{}-{label}",
            rustix::process::getpid().as_raw_nonzero().get(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let root = CString::new(root.as_os_str().as_bytes()).unwrap();
        mkdir(root.as_c_str(), Mode::RWXU).unwrap();
        (LinuxArtifactStore::at(root.clone()), root)
    }

    #[test]
    fn codec_round_trips_and_reports_runtime_version() {
        let codec = LinuxCompression;
        if !codec.available() {
            return;
        }
        let source = vec![42; 128 * 1024];
        let compressed = codec.compress(&source).unwrap();
        assert!(compressed.len() < source.len());
        assert_eq!(codec.decompress(&compressed, source.len()).unwrap(), source);
        assert!(codec.descriptor().version >= 10_500);
    }

    #[test]
    fn codec_version_policy_rejects_old_and_new_abis() {
        assert!(!supported_zstd_version(10_499));
        assert!(supported_zstd_version(10_500));
        assert!(supported_zstd_version(19_999));
        assert!(!supported_zstd_version(20_000));
    }

    #[test]
    fn concurrent_writers_publish_one_complete_record() {
        let (store, root) = temporary_store("writers");
        let store = Arc::new(store);
        let key = ArtifactKey(SemanticDigest([9; 32]));
        let record = Arc::new(vec![7; 64 * 1024]);
        let mut writers = Vec::new();
        for _ in 0..8 {
            let store = Arc::clone(&store);
            let record = Arc::clone(&record);
            writers.push(thread::spawn(move || {
                store.store(key, &record, 1024 * 1024, 900 * 1024)
            }));
        }
        for writer in writers {
            writer.join().unwrap().unwrap();
        }
        assert_eq!(store.load(key, record.len()).unwrap().unwrap(), *record);
        fs::remove_dir_all(root.to_string_lossy().as_ref()).unwrap();
    }

    #[test]
    fn concurrent_distinct_writers_preserve_global_disk_budget() {
        let (store, root) = temporary_store("concurrent-budget");
        let store = Arc::new(store);
        let mut writers = Vec::new();
        for byte in 1..=8 {
            let store = Arc::clone(&store);
            writers.push(thread::spawn(move || {
                store.store(
                    ArtifactKey(SemanticDigest([byte; 32])),
                    &[byte; 400],
                    1000,
                    800,
                )
            }));
        }
        for writer in writers {
            writer.join().unwrap().unwrap();
        }
        let retained = (1..=8)
            .filter(|byte| {
                store
                    .load(ArtifactKey(SemanticDigest([*byte; 32])), 1000)
                    .unwrap()
                    .is_some()
            })
            .count();
        assert_eq!(retained, 2);
        fs::remove_dir_all(root.to_string_lossy().as_ref()).unwrap();
    }

    #[test]
    #[ignore = "helper invoked by cross_process_writers_publish_atomically"]
    fn cross_process_writer_child() {
        let Ok(root) = std::env::var("GGML_NPU_CACHE_TEST_ROOT") else {
            return;
        };
        let store = LinuxArtifactStore::at(CString::new(root).unwrap());
        let key = ArtifactKey(SemanticDigest([11; 32]));
        store
            .store(key, &[13; 32 * 1024], 1024 * 1024, 900 * 1024)
            .unwrap();
    }

    #[test]
    fn cross_process_writers_publish_atomically() {
        let (store, root) = temporary_store("process-writers");
        let executable = std::env::current_exe().unwrap();
        let root_text = root.to_string_lossy();
        let mut children = Vec::new();
        for _ in 0..4 {
            children.push(
                Command::new(&executable)
                    .args([
                        "--ignored",
                        "--exact",
                        "host::linux::cache::tests::cross_process_writer_child",
                    ])
                    .env("GGML_NPU_CACHE_TEST_ROOT", root_text.as_ref())
                    .spawn()
                    .unwrap(),
            );
        }
        for mut child in children {
            assert!(child.wait().unwrap().success());
        }
        let key = ArtifactKey(SemanticDigest([11; 32]));
        assert_eq!(
            store.load(key, 64 * 1024).unwrap().unwrap(),
            vec![13; 32 * 1024]
        );
        fs::remove_dir_all(root_text.as_ref()).unwrap();
    }

    #[test]
    fn disk_budget_evicts_oldest_deterministically_and_bypasses_oversized() {
        let (store, root) = temporary_store("budget");
        let first = ArtifactKey(SemanticDigest([1; 32]));
        let second = ArtifactKey(SemanticDigest([2; 32]));
        let third = ArtifactKey(SemanticDigest([3; 32]));
        store.store(first, &[1; 400], 1000, 800).unwrap();
        store.store(second, &[2; 400], 1000, 800).unwrap();
        store.store(third, &[3; 400], 1000, 800).unwrap();
        assert!(store.load(first, 1000).unwrap().is_none());
        assert!(store.load(second, 1000).unwrap().is_some());
        assert!(store.load(third, 1000).unwrap().is_some());
        assert_eq!(
            store.store(first, &[0; 1001], 1000, 800),
            Err(CacheError::Store)
        );
        fs::remove_dir_all(root.to_string_lossy().as_ref()).unwrap();
    }
}
