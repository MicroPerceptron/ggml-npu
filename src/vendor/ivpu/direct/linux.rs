//! IVPU-on-Linux composition adapter.
//!
//! Device paths, IVPU request semantics, and IVPU-specific errno translation remain here. Typed
//! ioctl mechanics, DRM PRIME/GEM, DMA-BUF ownership, and mapping composition come from
//! `host::linux`.

use core::ffi::CStr;

#[cfg(feature = "bench-hardware")]
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
use crate::interop::AllocationIdentity;
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

    #[cfg(feature = "bench-hardware")]
    pub(super) fn linux_fd_for_external_api_test(&self) -> i32 {
        self.0.as_fd().as_raw_fd()
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
