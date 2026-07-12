//! IVPU-on-Linux composition adapter.
//!
//! Device paths, IVPU request semantics, and IVPU-specific errno translation remain here. Typed
//! ioctl mechanics, DRM PRIME/GEM, DMA-BUF ownership, and mapping composition come from
//! `host::linux`.

use alloc::boxed::Box;
use core::ffi::{c_void, CStr};
use core::sync::atomic::{AtomicU8, Ordering};

use rustix::fd::AsRawFd;

use crate::host::linux::dmabuf::{LinuxDmaBuf, LinuxDmaBufError, LinuxDmaBufInfo};
use crate::host::linux::drm::{self, DrmError, PrimeCapabilities};
use crate::host::linux::file::LinuxFile;
use crate::host::linux::ioctl::{set, update};
use crate::host::linux::mapping::LinuxOwnedMapping;
use crate::host::{
    builtin, FileError, FileService, HostEnvironment, MapOptions, MemoryMapping, OpenAccess,
    OpenOptions, OwnedFile,
};
use crate::interop::{
    AllocationIdentity, AllocationOwnership, AllocationToken, OpaqueExternalHandle, OwnershipError,
    SharedAllocationOwner,
};
use crate::runtime::{AccessMode, RuntimeError};

use super::uapi::*;

const DEVICE_PATHS: [&CStr; 8] = [
    c"/dev/accel/accel0",
    c"/dev/accel/accel1",
    c"/dev/accel/accel2",
    c"/dev/accel/accel3",
    c"/dev/accel/accel4",
    c"/dev/accel/accel5",
    c"/dev/accel/accel6",
    c"/dev/accel/accel7",
];

fn file_error(error: FileError) -> RuntimeError {
    RuntimeError::External {
        domain: crate::host::linux::ERROR_DOMAIN,
        code: i64::from(error.0),
    }
}

fn ioctl_error(error: rustix::io::Errno) -> RuntimeError {
    match error {
        rustix::io::Errno::NOMEM => RuntimeError::OutOfMemory,
        rustix::io::Errno::NODEV | rustix::io::Errno::IO | rustix::io::Errno::BADFD => {
            RuntimeError::DeviceLost
        }
        rustix::io::Errno::TIME | rustix::io::Errno::TIMEDOUT => RuntimeError::DeadlineExpired,
        _ => RuntimeError::External {
            domain: crate::host::linux::ERROR_DOMAIN,
            code: i64::from(error.raw_os_error()),
        },
    }
}

fn drm_error(error: DrmError) -> RuntimeError {
    match error {
        DrmError::Os(error) => ioctl_error(error),
        DrmError::InvalidResponse => RuntimeError::DeviceLost,
        DrmError::InvalidHandle => RuntimeError::InvalidArgument,
    }
}

fn dmabuf_error(error: LinuxDmaBufError) -> RuntimeError {
    match error {
        LinuxDmaBufError::InvalidArgument => RuntimeError::InvalidArgument,
        LinuxDmaBufError::ResourceLimit => RuntimeError::ResourceLimit,
        LinuxDmaBufError::OutOfMemory => RuntimeError::OutOfMemory,
        LinuxDmaBufError::DeviceLost | LinuxDmaBufError::InvalidResponse => {
            RuntimeError::DeviceLost
        }
        LinuxDmaBufError::Os(error) => ioctl_error(error),
        LinuxDmaBufError::System(code) => RuntimeError::External {
            domain: crate::host::linux::ERROR_DOMAIN,
            code: i64::from(code),
        },
    }
}

pub(super) struct LinuxIvpuDeviceFile(LinuxFile);

impl LinuxIvpuDeviceFile {
    pub(super) fn open_index(index: usize) -> Result<Self, RuntimeError> {
        let path = DEVICE_PATHS
            .get(index)
            .copied()
            .ok_or(RuntimeError::InvalidArgument)?;
        builtin()
            .files()
            .open_device(
                path,
                OpenOptions {
                    access: OpenAccess::ReadWrite,
                    nonblocking: true,
                },
            )
            .map(Self)
            .map_err(file_error)
    }

    pub(super) fn path(index: usize) -> Option<&'static CStr> {
        DEVICE_PATHS.get(index).copied()
    }

    pub(super) const fn device_count() -> usize {
        DEVICE_PATHS.len()
    }

    pub(super) fn diagnostic_id(&self) -> i64 {
        self.0.diagnostic_id()
    }

    pub(super) fn map(&self, options: MapOptions) -> Result<LinuxIvpuMapping, RuntimeError> {
        builtin()
            .files()
            .map_file(&self.0, options)
            .map(LinuxOwnedMapping::File)
            .map(LinuxIvpuMapping)
            .map_err(file_error)
    }

    pub(super) fn prime_capabilities(&self) -> Result<PrimeCapabilities, RuntimeError> {
        drm::prime_capabilities(&self.0).map_err(drm_error)
    }

    pub(super) fn get_param(&self, value: &mut DrmIvpuParam) -> Result<(), RuntimeError> {
        // SAFETY: IVPU request number and value layout are asserted in `uapi` tests.
        unsafe { update::<DRM_IOCTL_IVPU_GET_PARAM, _>(self.0.as_fd(), value) }.map_err(ioctl_error)
    }

    pub(super) fn create_bo(&self, value: &mut DrmIvpuBoCreate) -> Result<(), RuntimeError> {
        // SAFETY: IVPU request number and value layout are asserted in `uapi` tests.
        unsafe { update::<DRM_IOCTL_IVPU_BO_CREATE, _>(self.0.as_fd(), value) }.map_err(ioctl_error)
    }

    pub(super) fn bo_info(&self, value: &mut DrmIvpuBoInfo) -> Result<(), RuntimeError> {
        // SAFETY: IVPU request number and value layout are asserted in `uapi` tests.
        unsafe { update::<DRM_IOCTL_IVPU_BO_INFO, _>(self.0.as_fd(), value) }.map_err(ioctl_error)
    }

    pub(super) fn wait_bo(&self, value: &mut DrmIvpuBoWait) -> Result<(), RuntimeError> {
        // SAFETY: IVPU request number and value layout are asserted in `uapi` tests.
        unsafe { update::<DRM_IOCTL_IVPU_BO_WAIT, _>(self.0.as_fd(), value) }.map_err(ioctl_error)
    }

    pub(super) fn create_queue(&self, value: &mut DrmIvpuCmdqCreate) -> Result<(), RuntimeError> {
        // SAFETY: IVPU request number and value layout are asserted in `uapi` tests.
        unsafe { update::<DRM_IOCTL_IVPU_CMDQ_CREATE, _>(self.0.as_fd(), value) }
            .map_err(ioctl_error)
    }

    pub(super) fn destroy_queue(&self, id: u32) -> Result<(), RuntimeError> {
        let value = DrmIvpuCmdqDestroy { cmdq_id: id };
        // SAFETY: IVPU request number and value layout are asserted in `uapi` tests.
        unsafe { set::<DRM_IOCTL_IVPU_CMDQ_DESTROY, _>(self.0.as_fd(), value) }.map_err(ioctl_error)
    }

    pub(super) fn submit_queue(&self, value: DrmIvpuCmdqSubmit) -> Result<(), RuntimeError> {
        // SAFETY: the validated submit plan retains its handle slice for the synchronous ioctl.
        unsafe { set::<DRM_IOCTL_IVPU_CMDQ_SUBMIT, _>(self.0.as_fd(), value) }.map_err(ioctl_error)
    }

    pub(super) fn close_bo(&self, handle: u32) -> Result<(), RuntimeError> {
        drm::close_gem(&self.0, handle).map_err(drm_error)
    }

    pub(super) fn import_shared(
        &self,
        allocation: &LinuxIvpuSharedAllocation,
    ) -> Result<u32, RuntimeError> {
        drm::prime_fd_to_handle(&self.0, allocation.0.as_fd()).map_err(drm_error)
    }

    pub(super) fn export_shared(
        &self,
        handle: u32,
        bytes: u64,
        alignment: u64,
    ) -> Result<LinuxIvpuSharedAllocation, RuntimeError> {
        let fd = drm::prime_handle_to_fd(&self.0, handle).map_err(drm_error)?;
        LinuxDmaBuf::from_owned_fd(fd, bytes, alignment)
            .map(LinuxIvpuSharedAllocation)
            .map_err(dmabuf_error)
    }
}

pub(crate) struct LinuxIvpuSharedAllocation(LinuxDmaBuf);

impl LinuxIvpuSharedAllocation {
    pub(super) fn allocate(bytes: u64, alignment: u64) -> Result<Self, RuntimeError> {
        LinuxDmaBuf::allocate_system(bytes, alignment)
            .map(Self)
            .map_err(dmabuf_error)
    }

    pub(super) fn info(&self) -> LinuxDmaBufInfo {
        self.0.info()
    }

    pub(super) fn identity(&self) -> AllocationIdentity {
        self.0.info().identity
    }

    pub(super) fn duplicate(&self) -> Result<Self, RuntimeError> {
        self.0.duplicate().map(Self).map_err(dmabuf_error)
    }

    pub(super) fn map(&self, access: AccessMode) -> Result<LinuxIvpuMapping, RuntimeError> {
        self.0
            .map(access)
            .map(LinuxOwnedMapping::DmaBuf)
            .map(LinuxIvpuMapping)
            .map_err(dmabuf_error)
    }

    pub(super) fn raw_fd(&self) -> i32 {
        self.0.as_fd().as_raw_fd()
    }

    #[cfg(all(test, feature = "bench-hardware"))]
    pub(super) fn linux_fd_for_external_api_test(&self) -> i32 {
        self.raw_fd()
    }
}

const DIRECT_SHARED_PROVIDER: u64 = u64::from_be_bytes(*b"IVPUSHAR");
const OWNER_BACKEND: u8 = 0;
const OWNER_EXTERNAL: u8 = 1;
const OWNER_UNCERTAIN: u8 = 2;
const OWNER_RELEASED: u8 = 3;
static NEXT_SHARED_GENERATION: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(1);

pub(in crate::vendor::ivpu) struct DirectSharedBuffer {
    allocation: LinuxIvpuSharedAllocation,
    mapping: LinuxIvpuMapping,
    identity: AllocationIdentity,
    external_fd: i32,
    ownership: AtomicU8,
}

impl DirectSharedBuffer {
    pub(in crate::vendor::ivpu) fn allocate(
        bytes: usize,
    ) -> Result<Box<dyn SharedAllocationOwner>, RuntimeError> {
        let allocation = LinuxIvpuSharedAllocation::allocate(bytes.max(1) as u64, 4096)?;
        let mapping = allocation.map(AccessMode::ReadWrite)?;
        let external_fd = allocation.raw_fd();
        let mut identity = allocation.identity();
        identity.generation = NEXT_SHARED_GENERATION.fetch_add(1, Ordering::AcqRel);
        Ok(Box::new(Self {
            allocation,
            mapping,
            identity,
            external_fd,
            ownership: AtomicU8::new(OWNER_BACKEND),
        }))
    }

    pub(super) fn duplicate_allocation(&self) -> Result<LinuxIvpuSharedAllocation, RuntimeError> {
        self.allocation.duplicate()
    }

    pub(super) fn from_token(token: AllocationToken) -> Option<&'static Self> {
        if token.provider != DIRECT_SHARED_PROVIDER {
            return None;
        }
        // SAFETY: tokens are issued only from a live boxed DirectSharedBuffer and callers retain
        // the owning ggml buffer through direct submission completion.
        Some(unsafe { &*token.owner.as_ptr().cast::<Self>() })
    }
}

impl SharedAllocationOwner for DirectSharedBuffer {
    fn address(&self) -> core::ptr::NonNull<u8> {
        self.mapping.address()
    }

    fn len(&self) -> usize {
        self.mapping.len()
    }

    fn alignment(&self) -> usize {
        4096
    }

    fn identity(&self) -> AllocationIdentity {
        self.identity
    }

    fn token(&self) -> AllocationToken {
        AllocationToken {
            provider: DIRECT_SHARED_PROVIDER,
            // SAFETY: self is boxed before this method is exposed through the backend buffer.
            owner: unsafe {
                core::ptr::NonNull::new_unchecked((self as *const Self).cast_mut().cast::<c_void>())
            },
        }
    }

    fn external_handle(&self) -> OpaqueExternalHandle<'_> {
        OpaqueExternalHandle {
            type_name: c"ggml.platform.linux.shared-memory-fd.v1",
            data: (&self.external_fd as *const i32).cast::<c_void>(),
            size: core::mem::size_of::<i32>(),
            _owner: core::marker::PhantomData,
        }
    }

    fn ownership(&self) -> AllocationOwnership {
        match self.ownership.load(Ordering::Acquire) {
            OWNER_BACKEND => AllocationOwnership::Backend,
            OWNER_EXTERNAL => AllocationOwnership::External,
            OWNER_UNCERTAIN => AllocationOwnership::Uncertain,
            _ => AllocationOwnership::Released,
        }
    }

    fn acquire_backend(&self) -> Result<(), OwnershipError> {
        self.ownership
            .compare_exchange(
                OWNER_EXTERNAL,
                OWNER_BACKEND,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|state| match state {
                OWNER_UNCERTAIN => OwnershipError::Uncertain,
                OWNER_RELEASED => OwnershipError::Released,
                _ => OwnershipError::WrongOwner,
            })
    }

    fn release_external(&self) -> Result<(), OwnershipError> {
        self.ownership
            .compare_exchange(
                OWNER_BACKEND,
                OWNER_EXTERNAL,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|state| match state {
                OWNER_UNCERTAIN => OwnershipError::Uncertain,
                OWNER_RELEASED => OwnershipError::Released,
                _ => OwnershipError::WrongOwner,
            })
    }

    fn mark_uncertain(&self) {
        self.ownership.store(OWNER_UNCERTAIN, Ordering::Release);
    }
}

pub(super) struct LinuxIvpuMapping(LinuxOwnedMapping);

impl MemoryMapping for LinuxIvpuMapping {
    fn address(&self) -> core::ptr::NonNull<u8> {
        self.0.address()
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}
