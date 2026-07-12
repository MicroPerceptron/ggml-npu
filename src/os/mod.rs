//! Host/OS service seam. Dynamic-library loading intentionally remains outside this trait.

#![allow(dead_code)]

use core::ffi::CStr;
use core::ptr::NonNull;

use crate::runtime::Deadline;

#[cfg(target_os = "linux")]
pub mod linux;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OsError(pub i32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenAccess {
    ReadOnly,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenOptions {
    pub access: OpenAccess,
    pub nonblocking: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapOptions {
    pub offset: u64,
    pub bytes: usize,
    pub readable: bool,
    pub writable: bool,
    pub shared: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitInterest {
    Readable,
    Writable,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ready {
    pub readable: bool,
    pub writable: bool,
    pub error: bool,
    pub hung_up: bool,
}

pub trait OwnedDeviceFile {
    /// Stable process-local diagnostic identity; never serialized into an artifact.
    fn raw_handle(&self) -> i64;
}

pub trait MemoryMapping {
    fn address(&self) -> NonNull<u8>;
    fn len(&self) -> usize;
}

pub trait OsServices {
    type File: OwnedDeviceFile;
    type Mapping: MemoryMapping;

    fn open_device(&self, path: &CStr, options: OpenOptions) -> Result<Self::File, OsError>;
    fn map_file(&self, file: &Self::File, options: MapOptions) -> Result<Self::Mapping, OsError>;
    fn wait(
        &self,
        file: &Self::File,
        interest: WaitInterest,
        deadline: Deadline,
    ) -> Result<Ready, OsError>;
    fn monotonic_now_ns(&self) -> Result<u64, OsError>;
}

/// Optional provider-specific asynchronous command transport. Implement only when a device's
/// `file_operations` exposes `uring_cmd`; ordinary ioctl cannot be wrapped by this generically.
pub trait UringCommandServices: OsServices {
    type Completion;

    fn submit_uring_command(
        &self,
        file: &Self::File,
        provider_opcode: u32,
        command: &[u8],
    ) -> Result<Self::Completion, OsError>;
    fn wait_uring_completion(
        &self,
        completion: &Self::Completion,
        deadline: Deadline,
    ) -> Result<i32, OsError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeFile(i64);
    impl OwnedDeviceFile for FakeFile {
        fn raw_handle(&self) -> i64 {
            self.0
        }
    }

    struct FakeMap([u8; 16]);
    impl MemoryMapping for FakeMap {
        fn address(&self) -> NonNull<u8> {
            NonNull::from(&self.0).cast()
        }
        fn len(&self) -> usize {
            self.0.len()
        }
    }

    struct FakeOs;
    impl OsServices for FakeOs {
        type File = FakeFile;
        type Mapping = FakeMap;

        fn open_device(&self, path: &CStr, _options: OpenOptions) -> Result<Self::File, OsError> {
            (!path.to_bytes().is_empty())
                .then_some(FakeFile(7))
                .ok_or(OsError(22))
        }
        fn map_file(
            &self,
            _file: &Self::File,
            options: MapOptions,
        ) -> Result<Self::Mapping, OsError> {
            (options.bytes <= 16)
                .then_some(FakeMap([0; 16]))
                .ok_or(OsError(12))
        }
        fn wait(
            &self,
            _file: &Self::File,
            _interest: WaitInterest,
            deadline: Deadline,
        ) -> Result<Ready, OsError> {
            if deadline.expired_at(10) {
                return Ok(Ready {
                    readable: false,
                    writable: false,
                    error: false,
                    hung_up: false,
                });
            }
            Ok(Ready {
                readable: true,
                writable: false,
                error: false,
                hung_up: false,
            })
        }
        fn monotonic_now_ns(&self) -> Result<u64, OsError> {
            Ok(10)
        }
    }

    #[test]
    fn fake_os_conforms_to_ownership_mapping_and_deadlines() {
        let os = FakeOs;
        let file = os
            .open_device(
                c"/dev/fake",
                OpenOptions {
                    access: OpenAccess::ReadWrite,
                    nonblocking: false,
                },
            )
            .unwrap();
        assert_eq!(file.raw_handle(), 7);
        let mapping = os
            .map_file(
                &file,
                MapOptions {
                    offset: 0,
                    bytes: 16,
                    readable: true,
                    writable: true,
                    shared: true,
                },
            )
            .unwrap();
        assert_eq!(mapping.len(), 16);
        assert!(
            !os.wait(&file, WaitInterest::Readable, Deadline::MonotonicNs(10))
                .unwrap()
                .readable
        );
    }
}
