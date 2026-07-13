//! Direct Linux `intel_vpu` foundation.
//!
//! Command submission is restricted to validated, opt-in tiny hardware tests.

#![allow(dead_code)]

pub mod budget;
pub mod corpus;
pub mod elf;
mod firmware;
pub mod health;
pub mod hpi;
pub mod link;
pub(super) mod linux;
pub mod load;
pub mod oracle;
pub mod prepare;
pub mod transport;
mod uapi;
pub mod validate;

use core::marker::PhantomData;

use crate::host::{MapOptions, MemoryMapping};
use crate::interop::AllocationIdentity;
use crate::runtime::{
    artifact::ArtifactEnvelope, AccessMode, Deadline, DeviceId, MemoryRequest, RuntimeError,
    RuntimeHal, SubmissionBindings,
};

use uapi::*;

use self::linux::{LinuxIvpuDeviceFile, LinuxIvpuMapping, LinuxIvpuSharedAllocation};
use self::transport::{AccelTransport, IvpuTransport};

const IVPU_ERROR_DOMAIN: u32 = u32::from_be_bytes(*b"IVPU");

#[cfg(test)]
pub(crate) fn test_artifact_location() -> &'static core::ffi::CStr {
    use crate::host::{ConfigService, HostEnvironment};

    crate::host::builtin()
        .config()
        .value(c"GGML_NPU_TEST_ARTIFACT")
        .expect("set GGML_NPU_TEST_ARTIFACT to the bounded Intel NPU test artifact")
}

#[cfg(test)]
pub(crate) fn load_test_artifact(max_bytes: usize) -> alloc::vec::Vec<u8> {
    use crate::host::{ArtifactService, HostEnvironment};

    crate::host::builtin()
        .artifacts()
        .load(test_artifact_location(), max_bytes)
        .expect("load bounded Intel NPU test artifact")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceParameters {
    pub device_id: u32,
    pub revision: u32,
    pub platform_type: u64,
    pub core_clock_rate: u64,
    pub context_id: u64,
    pub context_base_address: u64,
    pub firmware_api: [u64; 8],
    pub tile_config: u64,
    pub sku: u64,
    pub capability_bits: u64,
    pub max_contexts: u64,
    pub preempt_buffer_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimeCapabilities {
    pub import: bool,
    pub export: bool,
}

pub struct IvpuDevice {
    file: LinuxIvpuDeviceFile,
    path_index: usize,
}

impl IvpuDevice {
    pub fn open_first() -> Result<Self, RuntimeError> {
        let mut last = RuntimeError::Unsupported;
        for index in 0..LinuxIvpuDeviceFile::device_count() {
            match Self::open_index(index) {
                Ok(device) => return Ok(device),
                Err(error) => last = error,
            }
        }
        Err(last)
    }

    pub fn open_index(index: usize) -> Result<Self, RuntimeError> {
        LinuxIvpuDeviceFile::open_index(index).map(|file| Self {
            file,
            path_index: index,
        })
    }

    pub fn device_path(&self) -> &'static core::ffi::CStr {
        LinuxIvpuDeviceFile::path(self.path_index).expect("validated device path index")
    }

    pub fn diagnostic_id(&self) -> i64 {
        self.file.diagnostic_id()
    }

    /// Read-only kernel parameter query.
    pub fn get_param(&self, param: u32, index: u32) -> Result<u64, RuntimeError> {
        let mut value = DrmIvpuParam {
            param,
            index,
            value: 0,
        };
        self.file.get_param(&mut value)?;
        Ok(value.value)
    }

    /// Bounded snapshot using only DRM_IOCTL_IVPU_GET_PARAM.
    pub fn parameters(&self) -> Result<DeviceParameters, RuntimeError> {
        let mut firmware_api = [0u64; 8];
        for (index, version) in firmware_api.iter_mut().enumerate() {
            *version = self.get_param(DRM_IVPU_PARAM_FW_API_VERSION, index as u32)?;
        }
        let mut capability_bits = 0u64;
        for capability in 1..=4u32 {
            if self.get_param(DRM_IVPU_PARAM_CAPABILITIES, capability)? != 0 {
                capability_bits |= 1u64 << capability;
            }
        }
        Ok(DeviceParameters {
            device_id: self.get_param(DRM_IVPU_PARAM_DEVICE_ID, 0)? as u32,
            revision: self.get_param(DRM_IVPU_PARAM_DEVICE_REVISION, 0)? as u32,
            platform_type: self.get_param(DRM_IVPU_PARAM_PLATFORM_TYPE, 0)?,
            core_clock_rate: self.get_param(DRM_IVPU_PARAM_CORE_CLOCK_RATE, 0)?,
            context_id: self.get_param(DRM_IVPU_PARAM_CONTEXT_ID, 0)?,
            context_base_address: self.get_param(DRM_IVPU_PARAM_CONTEXT_BASE_ADDRESS, 0)?,
            firmware_api,
            tile_config: self.get_param(DRM_IVPU_PARAM_TILE_CONFIG, 0)?,
            sku: self.get_param(DRM_IVPU_PARAM_SKU, 0)?,
            capability_bits,
            max_contexts: self.get_param(DRM_IVPU_PARAM_NUM_CONTEXTS, 0)?,
            preempt_buffer_bytes: self.get_param(DRM_IVPU_PARAM_PREEMPT_BUFFER_SIZE, 0)?,
        })
    }

    pub fn prime_capabilities(&self) -> Result<PrimeCapabilities, RuntimeError> {
        let capabilities = self.file.prime_capabilities()?;
        Ok(PrimeCapabilities {
            import: capabilities.import,
            export: capabilities.export,
        })
    }

    pub fn create_queue(
        &self,
        priority: u32,
        flags: u32,
    ) -> Result<CommandQueue<'_>, RuntimeError> {
        if priority > DRM_IVPU_JOB_PRIORITY_REALTIME || flags & !DRM_IVPU_CMDQ_FLAG_TURBO != 0 {
            return Err(RuntimeError::InvalidArgument);
        }
        if self.get_param(DRM_IVPU_PARAM_CAPABILITIES, DRM_IVPU_CAP_MANAGE_CMDQ)? == 0 {
            return Err(RuntimeError::Unsupported);
        }
        let mut request = DrmIvpuCmdqCreate {
            cmdq_id: 0,
            priority,
            flags,
        };
        self.file.create_queue(&mut request)?;
        if request.cmdq_id == 0 {
            return Err(RuntimeError::External {
                domain: IVPU_ERROR_DOMAIN,
                code: -3,
            });
        }
        Ok(CommandQueue {
            device: self,
            id: request.cmdq_id,
        })
    }

    /// Create a small mappable GEM BO. This wrapper is implemented for ABI validation but is not
    /// used by backend execution until the direct command protocol is specified.
    pub fn create_bo(&self, bytes: u64, flags: u32) -> Result<BufferObject<'_>, RuntimeError> {
        if bytes == 0 || bytes > usize::MAX as u64 || flags & !DRM_IVPU_BO_FLAGS != 0 {
            return Err(RuntimeError::InvalidArgument);
        }
        let mut request = DrmIvpuBoCreate {
            size: bytes,
            flags,
            handle: 0,
            vpu_addr: 0,
        };
        self.file.create_bo(&mut request)?;
        if request.handle == 0 || request.vpu_addr == 0 {
            return Err(RuntimeError::External {
                domain: IVPU_ERROR_DOMAIN,
                code: -1,
            });
        }
        Ok(BufferObject {
            device: self,
            handle: request.handle,
            requested_bytes: bytes,
            vpu_address: request.vpu_addr,
            backing: BufferBacking::IvpuGem,
            shared: None,
        })
    }

    /// Import one host-owned transferable allocation into this context.
    pub(super) fn import_shared(
        &self,
        allocation: LinuxIvpuSharedAllocation,
    ) -> Result<BufferObject<'_>, RuntimeError> {
        self.import_shared_impl(allocation, true)
    }

    pub(super) fn import_shared_unretained(
        &self,
        allocation: LinuxIvpuSharedAllocation,
    ) -> Result<BufferObject<'_>, RuntimeError> {
        self.import_shared_impl(allocation, false)
    }

    fn import_shared_impl(
        &self,
        allocation: LinuxIvpuSharedAllocation,
        retain_allocation: bool,
    ) -> Result<BufferObject<'_>, RuntimeError> {
        if !self.prime_capabilities()?.import {
            return Err(RuntimeError::Unsupported);
        }
        let descriptor = allocation.info();
        let handle = self.file.import_shared(&allocation)?;
        let mut info = DrmIvpuBoInfo {
            handle,
            ..DrmIvpuBoInfo::default()
        };
        if let Err(error) = self.file.bo_info(&mut info) {
            let _ = self.file.close_bo(handle);
            return Err(error);
        }
        if info.vpu_addr == 0 || info.size < descriptor.bytes {
            let _ = self.file.close_bo(handle);
            return Err(RuntimeError::Incompatible);
        }
        Ok(BufferObject {
            device: self,
            handle,
            requested_bytes: descriptor.bytes,
            vpu_address: info.vpu_addr,
            backing: BufferBacking::ImportedShared,
            shared: retain_allocation.then_some(allocation),
        })
    }
}

pub struct DirectIvpuRuntime<T = AccelTransport> {
    transport: T,
}

impl<T> DirectIvpuRuntime<T> {
    pub const fn new(transport: T) -> Self {
        Self { transport }
    }
}

impl Default for DirectIvpuRuntime<AccelTransport> {
    fn default() -> Self {
        Self::new(AccelTransport)
    }
}
pub enum DirectMemory {}
pub enum DirectProgram {}
pub enum DirectQueue {}
pub enum DirectEvent {}

/// Discovery-only runtime adapter. Resource helpers remain explicit methods on [`IvpuDevice`]
/// until the native executable and command protocol are specified.
impl RuntimeHal for DirectIvpuRuntime<AccelTransport> {
    type Device = IvpuDevice;
    type Memory = DirectMemory;
    type Program = DirectProgram;
    type Queue = DirectQueue;
    type Event = DirectEvent;

    fn open_device(&self, id: DeviceId) -> Result<Self::Device, RuntimeError> {
        let index = id
            .get()
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(RuntimeError::InvalidArgument)?;
        self.transport.open(index)
    }

    fn allocate(
        &self,
        _device: &Self::Device,
        _request: MemoryRequest,
    ) -> Result<Self::Memory, RuntimeError> {
        Err(RuntimeError::Unsupported)
    }

    fn load_program(
        &self,
        _device: &Self::Device,
        _artifact: &ArtifactEnvelope<'_>,
    ) -> Result<Self::Program, RuntimeError> {
        Err(RuntimeError::Unsupported)
    }

    fn create_queue(&self, _device: &Self::Device) -> Result<Self::Queue, RuntimeError> {
        Err(RuntimeError::Unsupported)
    }

    fn submit(
        &self,
        _queue: &Self::Queue,
        _program: &Self::Program,
        _bindings: &SubmissionBindings<'_>,
        _deadline: Deadline,
    ) -> Result<Self::Event, RuntimeError> {
        Err(RuntimeError::Unsupported)
    }

    fn wait(&self, _event: &Self::Event, _deadline: Deadline) -> Result<(), RuntimeError> {
        Err(RuntimeError::Unsupported)
    }

    fn map<'a>(&self, memory: &'a mut Self::Memory) -> Result<&'a mut [u8], RuntimeError> {
        match *memory {}
    }

    fn unmap(&self, memory: &mut Self::Memory) -> Result<(), RuntimeError> {
        match *memory {}
    }

    fn destroy_event(&self, event: Self::Event) -> Result<(), RuntimeError> {
        match event {}
    }

    fn destroy_queue(&self, queue: Self::Queue) -> Result<(), RuntimeError> {
        match queue {}
    }

    fn unload_program(&self, program: Self::Program) -> Result<(), RuntimeError> {
        match program {}
    }

    fn free_memory(&self, memory: Self::Memory) -> Result<(), RuntimeError> {
        match memory {}
    }

    fn close_device(&self, _device: Self::Device) -> Result<(), RuntimeError> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferInfo {
    pub flags: u32,
    pub vpu_address: u64,
    pub mmap_offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferBacking {
    IvpuGem,
    ImportedShared,
}

pub struct BufferObject<'device> {
    device: &'device IvpuDevice,
    handle: u32,
    requested_bytes: u64,
    vpu_address: u64,
    backing: BufferBacking,
    shared: Option<LinuxIvpuSharedAllocation>,
}

impl BufferObject<'_> {
    pub const fn handle(&self) -> u32 {
        self.handle
    }

    pub const fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }

    pub const fn vpu_address(&self) -> u64 {
        self.vpu_address
    }

    pub const fn backing(&self) -> BufferBacking {
        self.backing
    }

    pub(super) fn shared_identity(&self) -> Option<AllocationIdentity> {
        self.shared
            .as_ref()
            .map(LinuxIvpuSharedAllocation::identity)
    }

    /// Duplicate a transferable allocation for this exact backing object.
    pub(super) fn export_shared(&self) -> Result<LinuxIvpuSharedAllocation, RuntimeError> {
        if let Some(shared) = &self.shared {
            return shared.duplicate();
        }
        if !self.device.prime_capabilities()?.export {
            return Err(RuntimeError::Unsupported);
        }
        let info = self.info()?;
        self.device
            .file
            .export_shared(self.handle, info.bytes, rustix::param::page_size() as u64)
    }

    pub fn info(&self) -> Result<BufferInfo, RuntimeError> {
        let mut info = DrmIvpuBoInfo {
            handle: self.handle,
            ..DrmIvpuBoInfo::default()
        };
        self.device.file.bo_info(&mut info)?;
        if info.handle != self.handle || info.size < self.requested_bytes {
            return Err(RuntimeError::External {
                domain: IVPU_ERROR_DOMAIN,
                code: -2,
            });
        }
        Ok(BufferInfo {
            flags: info.flags,
            vpu_address: info.vpu_addr,
            mmap_offset: info.mmap_offset,
            bytes: info.size,
        })
    }

    pub fn map(&self, writable: bool) -> Result<BoMapping<'_>, RuntimeError> {
        if let Some(shared) = &self.shared {
            let access = if writable {
                AccessMode::ReadWrite
            } else {
                AccessMode::Read
            };
            return shared.map(access).map(|mapping| BoMapping {
                mapping,
                _owner: PhantomData,
            });
        }
        let info = self.info()?;
        if info.mmap_offset == 0 || info.bytes == 0 || info.bytes > usize::MAX as u64 {
            return Err(RuntimeError::InvalidArgument);
        }
        let mapping = self.device.file.map(MapOptions {
            offset: info.mmap_offset,
            bytes: info.bytes as usize,
            readable: true,
            writable,
            shared: true,
        })?;
        Ok(BoMapping {
            mapping,
            _owner: PhantomData,
        })
    }

    pub fn destroy(mut self) -> Result<(), RuntimeError> {
        let handle = self.handle;
        self.handle = 0;
        if self.backing == BufferBacking::ImportedShared {
            // PRIME_FD_TO_HANDLE may return an existing per-file handle without adding a userspace
            // close reference. Keep imported handles until the device file closes to avoid
            // invalidating another prepared program that shares the same allocation.
            Ok(())
        } else {
            self.device.file.close_bo(handle)
        }
    }

    /// Wait for this BO to become idle. This is meaningful for a command BO only after a future
    /// validated submission; the wrapper itself is stable UAPI and uses an absolute deadline.
    pub fn wait_idle(&self, deadline: Deadline) -> Result<u32, RuntimeError> {
        let timeout_ns = match deadline {
            Deadline::Infinite => i64::MAX,
            Deadline::MonotonicNs(value) => {
                i64::try_from(value).map_err(|_| RuntimeError::InvalidArgument)?
            }
        };
        let mut request = DrmIvpuBoWait {
            handle: self.handle,
            flags: 0,
            timeout_ns,
            job_status: u32::MAX,
            pad: 0,
        };
        self.device.file.wait_bo(&mut request)?;
        Ok(request.job_status)
    }
}

impl Drop for BufferObject<'_> {
    fn drop(&mut self) {
        if self.handle != 0 {
            if self.backing != BufferBacking::ImportedShared {
                let _ = self.device.file.close_bo(self.handle);
            }
            self.handle = 0;
        }
    }
}

pub struct CommandQueue<'device> {
    device: &'device IvpuDevice,
    id: u32,
}

impl CommandQueue<'_> {
    pub const fn id(&self) -> u32 {
        self.id
    }

    /// Submit one already-validated command image using the stable explicit-CMDQ UAPI.
    ///
    /// A zero preemption index requests the kernel-managed preemption allocation. Callers must
    /// retain every BO in `handles` until the command BO reports completion.
    pub fn submit(
        &self,
        handles: &[u32],
        commands_offset: u32,
        preempt_buffer_index: u32,
    ) -> Result<(), RuntimeError> {
        if self.id == 0
            || handles.is_empty()
            || handles.len() > validate::MAX_SUBMISSION_BUFFERS
            || handles.iter().any(|handle| *handle == 0)
            || commands_offset % 8 != 0
            || (preempt_buffer_index != 0 && preempt_buffer_index as usize >= handles.len())
        {
            return Err(RuntimeError::InvalidArgument);
        }
        let buffer_count =
            u32::try_from(handles.len()).map_err(|_| RuntimeError::InvalidArgument)?;
        let value = DrmIvpuCmdqSubmit {
            buffers_ptr: handles.as_ptr() as u64,
            buffer_count,
            cmdq_id: self.id,
            flags: 0,
            commands_offset,
            preempt_buffer_index,
            reserved: 0,
        };
        self.device.file.submit_queue(value)
    }

    pub fn destroy(mut self) -> Result<(), RuntimeError> {
        let id = self.id;
        self.id = 0;
        self.device.file.destroy_queue(id)
    }
}

impl Drop for CommandQueue<'_> {
    fn drop(&mut self) {
        if self.id != 0 {
            let _ = self.device.file.destroy_queue(self.id);
            self.id = 0;
        }
    }
}

pub struct BoMapping<'bo> {
    mapping: LinuxIvpuMapping,
    _owner: PhantomData<&'bo BufferObject<'bo>>,
}

impl MemoryMapping for BoMapping<'_> {
    fn address(&self) -> core::ptr::NonNull<u8> {
        self.mapping.address()
    }

    fn len(&self) -> usize {
        self.mapping.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "bench-hardware")]
    use crate::host::linux::resource_snapshot as resources;
    #[cfg(feature = "bench-hardware")]
    use crate::host::{ClockService, HostEnvironment};

    #[test]
    fn direct_bo_requests_are_strictly_bounded_without_hardware() {
        assert!(usize::MAX as u64 >= 64 * 1024 * 1024);
        assert_eq!(DRM_IVPU_BO_FLAGS & 0x8000_0000, 0);
    }

    #[test]
    #[cfg(feature = "bench-hardware")]
    #[ignore = "explicit read-only hardware query; never run in the default test gate"]
    fn read_only_device_parameter_query() {
        let device = IvpuDevice::open_first().unwrap();
        let parameters = device.parameters().unwrap();
        std::println!(
            "device={} fd={} device_id={:#x} revision={:#x} platform={} clock_hz={} context_id={} context_base={:#x} fw_api={:?} tile_config={:#x} sku={:#x} capabilities={:#x} max_contexts={} preempt_buffer_bytes={}",
            device.device_path().to_string_lossy(),
            device.diagnostic_id(),
            parameters.device_id,
            parameters.revision,
            parameters.platform_type,
            parameters.core_clock_rate,
            parameters.context_id,
            parameters.context_base_address,
            parameters.firmware_api,
            parameters.tile_config,
            parameters.sku,
            parameters.capability_bits,
            parameters.max_contexts,
            parameters.preempt_buffer_bytes,
        );
        assert_ne!(parameters.device_id, 0);
        assert_ne!(parameters.context_id, 0);
    }

    #[test]
    #[cfg(feature = "bench-hardware")]
    #[ignore = "explicit single-page GEM lifecycle; requires successful read-only query first"]
    fn bounded_bo_lifecycle() {
        const REQUEST_BYTES: u64 = 4096;
        const PATTERN_BYTES: usize = 64;

        let before = resources();
        let observed = {
            let device = IvpuDevice::open_first().unwrap();
            let bo = device
                .create_bo(REQUEST_BYTES, DRM_IVPU_BO_MAPPABLE)
                .unwrap();
            let info = bo.info().unwrap();
            assert_eq!(info.flags & DRM_IVPU_BO_MAPPABLE, DRM_IVPU_BO_MAPPABLE);
            assert!(info.bytes >= REQUEST_BYTES);
            assert!(info.bytes <= 64 * 1024);
            assert_eq!(info.mmap_offset % rustix::param::page_size() as u64, 0);

            let mapping = bo.map(true).unwrap();
            assert_eq!(mapping.len() as u64, info.bytes);
            assert!(mapping.len() >= PATTERN_BYTES);
            let address = mapping.address().as_ptr();
            for index in 0..PATTERN_BYTES {
                // SAFETY: the mapped range is at least PATTERN_BYTES long, uniquely owned by this
                // test, and remains mapped for the entire volatile access loop.
                unsafe { core::ptr::write_volatile(address.add(index), (index as u8) ^ 0xa5) };
            }
            for index in 0..PATTERN_BYTES {
                // SAFETY: same bounded live mapping established above.
                let actual = unsafe { core::ptr::read_volatile(address.add(index)) };
                assert_eq!(actual, (index as u8) ^ 0xa5);
            }
            let observed = (
                device.device_path().to_string_lossy().into_owned(),
                bo.handle(),
                info,
            );
            drop(mapping);
            bo.destroy().unwrap();
            drop(device);
            observed
        };
        let after = resources();
        let (path, handle, info) = observed;
        std::println!(
            "device={path} handle={handle} requested={REQUEST_BYTES} returned_size={} flags={:#x} vpu_addr={:#x} mmap_offset={:#x} pattern_bytes={PATTERN_BYTES} before={before:?} after={after:?}",
            info.bytes,
            info.flags,
            info.vpu_address,
            info.mmap_offset,
        );
        assert_eq!(after.open_fds, before.open_fds);
        if let (Some(before_rss), Some(after_rss)) = (before.rss_kib, after.rss_kib) {
            assert!(after_rss <= before_rss.saturating_add(2048));
        }
    }

    #[test]
    #[cfg(feature = "bench-hardware")]
    #[ignore = "explicit dma-heap/PRIME import-export lifecycle; no command submission"]
    fn bounded_shared_dmabuf_lifecycle() {
        const BYTES: u64 = 4096;
        let before = resources();
        let (capabilities, identity, exported_identity, vpu_address, handle) = {
            let allocation =
                LinuxIvpuSharedAllocation::allocate(BYTES, rustix::param::page_size() as u64)
                    .unwrap();
            let identity = allocation.info().identity;
            {
                let mapping = allocation.map(AccessMode::ReadWrite).unwrap();
                for index in 0..64 {
                    // SAFETY: mapping is writable and at least one page.
                    unsafe {
                        core::ptr::write_volatile(
                            mapping.address().as_ptr().add(index),
                            (index as u8) ^ 0x5a,
                        )
                    };
                }
            }
            let device = IvpuDevice::open_first().unwrap();
            let capabilities = device.prime_capabilities().unwrap();
            assert!(capabilities.import && capabilities.export);
            let bo = device.import_shared(allocation).unwrap();
            assert_eq!(bo.backing(), BufferBacking::ImportedShared);
            assert_ne!(bo.vpu_address(), 0);
            let handle = bo.handle();
            let vpu_address = bo.vpu_address();
            let exported = bo.export_shared().unwrap();
            let exported_identity = exported.info().identity;
            assert_eq!(identity, exported_identity);
            {
                let mapping = exported.map(AccessMode::Read).unwrap();
                for index in 0..64 {
                    // SAFETY: mapping is readable and at least one page.
                    let actual =
                        unsafe { core::ptr::read_volatile(mapping.address().as_ptr().add(index)) };
                    assert_eq!(actual, (index as u8) ^ 0x5a);
                }
            }
            drop(exported);
            bo.destroy().unwrap();
            drop(device);
            (
                capabilities,
                identity,
                exported_identity,
                vpu_address,
                handle,
            )
        };
        let after = resources();
        std::println!(
            "prime={capabilities:?} identity={identity:?} exported_identity={exported_identity:?} ivpu_handle={handle} vpu_address={vpu_address:#x} bytes={BYTES} before={before:?} after={after:?}"
        );
        assert_eq!(before.open_fds, after.open_fds);
    }

    #[test]
    #[cfg(feature = "bench-hardware")]
    #[ignore = "explicit stable-UAPI queue create/destroy; no command submission"]
    fn bounded_queue_lifecycle() {
        let before = resources();
        let (path, queue_id, preempt_bytes) = {
            let device = IvpuDevice::open_first().unwrap();
            let parameters = device.parameters().unwrap();
            assert_ne!(
                parameters.capability_bits & (1 << DRM_IVPU_CAP_MANAGE_CMDQ),
                0
            );
            assert!(parameters.preempt_buffer_bytes <= 32 * 1024 * 1024);
            let queue = device
                .create_queue(DRM_IVPU_JOB_PRIORITY_DEFAULT, 0)
                .unwrap();
            let observed = (
                device.device_path().to_string_lossy().into_owned(),
                queue.id(),
                parameters.preempt_buffer_bytes,
            );
            queue.destroy().unwrap();
            drop(device);
            observed
        };
        let after = resources();
        std::println!(
            "device={path} queue_id={queue_id} preempt_buffer_bytes={preempt_bytes} before={before:?} after={after:?}"
        );
        assert_eq!(after.open_fds, before.open_fds);
        if let (Some(before_rss), Some(after_rss)) = (before.rss_kib, after.rss_kib) {
            assert!(after_rss <= before_rss.saturating_add(2048));
        }
    }

    #[test]
    #[cfg(feature = "bench-hardware")]
    #[ignore = "enumeration-only Level Zero differential; no context, queue, graph, or execution"]
    fn level_zero_discovery_differential() {
        let device = IvpuDevice::open_first().unwrap();
        let parameters = device.parameters().unwrap();
        let level_zero = super::super::ze::probe();
        assert_eq!(level_zero.len(), 1);
        std::println!(
            "direct_device={} direct_id={:#x} direct_revision={:#x} level_zero_devices={} level_zero_name={}",
            device.device_path().to_string_lossy(),
            parameters.device_id,
            parameters.revision,
            level_zero.len(),
            level_zero[0].name,
        );
    }

    #[test]
    #[cfg(feature = "bench-hardware")]
    #[ignore = "explicit tiny direct-IVPU execution; performs two submits only after the first succeeds"]
    fn tiny_known_answer_direct_submission() {
        use std::time::Instant;

        use super::elf::parse_intel_elf;
        use super::hpi::inspect_hpi_requirements;
        use super::prepare::{prepare_inference, PreparationBudget};
        use super::transport::release_host_writes;

        const ARTIFACT_LIMIT: usize = 1024 * 1024;
        const ELEMENTS: usize = 32;
        const OUTPUT_BYTES: usize = ELEMENTS * core::mem::size_of::<f32>();
        const MAX_ABS_ERROR: f32 = 1.0e-5;
        const MAX_REL_ERROR: f32 = 1.0e-5;
        const SUBMISSIONS: usize = 2;

        let artifact = super::load_test_artifact(ARTIFACT_LIMIT);
        let length = artifact.len();
        let artifact_location = super::test_artifact_location().to_string_lossy();

        let input0 = core::array::from_fn::<_, ELEMENTS, _>(|index| index as f32 * 0.125 - 1.5);
        let input1 =
            core::array::from_fn::<_, ELEMENTS, _>(|index| ((index * 7) % 17) as f32 * 0.25 - 2.0);
        let mut oracle = [0.0f32; ELEMENTS];
        let level_zero = super::super::ze::probe();
        assert_eq!(level_zero.len(), 1);
        let oracle_count = super::super::graph::run_native_blob(
            &level_zero[0],
            &artifact,
            &[&input0, &input1],
            &mut oracle,
        )
        .unwrap();
        assert_eq!(oracle_count, ELEMENTS);
        assert!(oracle.iter().all(|value| value.is_finite()));
        drop(level_zero);

        let input0_bytes =
            unsafe { core::slice::from_raw_parts(input0.as_ptr().cast::<u8>(), OUTPUT_BYTES) };
        let input1_bytes =
            unsafe { core::slice::from_raw_parts(input1.as_ptr().cast::<u8>(), OUTPUT_BYTES) };

        let before = resources();
        let mut observed_outputs = [[0.0f32; ELEMENTS]; SUBMISSIONS];
        let mut elapsed = [std::time::Duration::ZERO; SUBMISSIONS];
        let (
            path,
            queue_id,
            user_bytes,
            preempt_bytes,
            accounted_bytes,
            firmware_api,
            output_identity,
        ) = {
            let elf = parse_intel_elf(&artifact).unwrap();
            let device = IvpuDevice::open_first().unwrap();
            let parameters = device.parameters().unwrap();
            super::validate::validate_device(&parameters).unwrap();
            assert_eq!(
                parameters.capability_bits & (1 << DRM_IVPU_CAP_MANAGE_CMDQ),
                1 << DRM_IVPU_CAP_MANAGE_CMDQ
            );
            let hpi = inspect_hpi_requirements(&elf, &parameters).unwrap();
            let prepared = prepare_inference(
                &device,
                &parameters,
                &elf,
                &hpi,
                &[input0_bytes, input1_bytes],
                PreparationBudget {
                    max_user_bo_bytes: 64 * 1024 * 1024,
                    max_total_accounted_bytes: 64 * 1024 * 1024,
                },
            )
            .unwrap();
            assert_eq!(prepared.command_count, 2);
            assert!(prepared
                .bindings
                .iter()
                .all(|binding| binding.backing == prepare::BindingBacking::ImportedShared));
            assert!(prepared.outputs_are_zero().unwrap());
            assert!(prepared.total_accounted_bytes <= 64 * 1024 * 1024);
            let queue = device
                .create_queue(DRM_IVPU_JOB_PRIORITY_DEFAULT, 0)
                .unwrap();

            for submission in 0..SUBMISSIONS {
                if submission != 0 {
                    prepared.zero_outputs().unwrap();
                    assert!(prepared.outputs_are_zero().unwrap());
                }
                release_host_writes();
                let started = Instant::now();
                queue
                    .submit(&prepared.handles, prepared.commands_offset(), 0)
                    .unwrap();
                let now = crate::host::builtin().clock().monotonic_now_ns().unwrap();
                let deadline = now.checked_add(2_000_000_000).unwrap();
                let status = prepared
                    .command
                    .wait_idle(Deadline::MonotonicNs(deadline))
                    .unwrap();
                elapsed[submission] = started.elapsed();
                assert_eq!(status, DRM_IVPU_JOB_STATUS_SUCCESS);

                let mut bytes = [0u8; OUTPUT_BYTES];
                prepared.copy_output(0, &mut bytes).unwrap();
                for (index, chunk) in bytes.chunks_exact(4).enumerate() {
                    observed_outputs[submission][index] =
                        f32::from_le_bytes(chunk.try_into().unwrap());
                }
                assert!(observed_outputs[submission]
                    .iter()
                    .all(|value| value.is_finite()));
                for (actual, expected) in observed_outputs[submission].iter().zip(oracle) {
                    let absolute = (actual - expected).abs();
                    let relative = absolute / expected.abs().max(1.0);
                    assert!(
                        absolute <= MAX_ABS_ERROR || relative <= MAX_REL_ERROR,
                        "submission={submission} actual={actual} expected={expected} absolute={absolute} relative={relative}"
                    );
                }
            }
            let (output_shared, output_logical_bytes, output_backing) =
                prepared.output_shared_allocation(0).unwrap();
            assert_eq!(output_backing, prepare::BindingBacking::ImportedShared);
            assert_eq!(output_logical_bytes, OUTPUT_BYTES);
            let output_identity = output_shared.info().identity;
            {
                let output_mapping = output_shared.map(AccessMode::Read).unwrap();
                let shared_output = unsafe {
                    core::slice::from_raw_parts(
                        output_mapping.address().as_ptr().cast::<f32>(),
                        ELEMENTS,
                    )
                };
                let shared_bits =
                    core::array::from_fn::<_, ELEMENTS, _>(|index| shared_output[index].to_bits());
                assert_eq!(
                    shared_bits,
                    observed_outputs[SUBMISSIONS - 1].map(f32::to_bits)
                );
            }
            drop(output_shared);
            assert_eq!(
                observed_outputs[0].map(f32::to_bits),
                observed_outputs[1].map(f32::to_bits)
            );
            let observed = (
                device.device_path().to_string_lossy().into_owned(),
                queue.id(),
                prepared.total_user_bo_bytes,
                parameters.preempt_buffer_bytes,
                prepared.total_accounted_bytes,
                parameters.firmware_api,
                output_identity,
            );
            queue.destroy().unwrap();
            prepared.destroy();
            drop(device);
            observed
        };
        let after = resources();
        let diff = super::oracle::compare_f32(
            &observed_outputs[0],
            &oracle,
            MAX_ABS_ERROR,
            MAX_REL_ERROR,
        );
        let max_absolute = diff.max_absolute;
        let max_relative = diff.max_relative;
        std::println!(
            "device={path} queue_id={queue_id} artifact={} submissions={SUBMISSIONS} user_bo_bytes={user_bytes} preempt_buffer_bytes={preempt_bytes} accounted_bytes={accounted_bytes} firmware_api={firmware_api:?} output_shared_identity={output_identity:?} oracle={oracle:?} direct={:?} max_absolute={max_absolute} max_relative={max_relative} tolerance_abs={MAX_ABS_ERROR} tolerance_rel={MAX_REL_ERROR} elapsed={elapsed:?} before={before:?} after={after:?}",
            artifact_location,
            observed_outputs[0],
        );
        assert!(
            diff.within(MAX_ABS_ERROR, MAX_REL_ERROR),
            "direct vs Level Zero oracle mismatch: {diff:?}"
        );
        assert_eq!(after.open_fds, before.open_fds);
        if let (Some(before_rss), Some(after_rss)) = (before.rss_kib, after.rss_kib) {
            assert!(after_rss <= before_rss.saturating_add(4096));
        }
        if let (Some(before_swap), Some(after_swap)) = (before.swap_kib, after.swap_kib) {
            assert!(after_swap <= before_swap);
        }
    }

    #[test]
    #[cfg(feature = "bench-hardware")]
    #[ignore = "explicit CPU -> direct IVPU -> Vulkan -> CPU same-DMA-BUF hardware KAT"]
    fn shared_allocation_ivpu_vulkan_known_answer() {
        use std::vec;

        use super::elf::parse_intel_elf;
        use super::hpi::inspect_hpi_requirements;
        use super::prepare::{prepare_inference, BindingBacking, PreparationBudget};
        use super::transport::release_host_writes;
        use crate::graph::{
            Dim, GraphIr, Node, OpRegistry, OperationContract, TensorDesc, TensorRole,
            OP_PARAM_BYTES,
        };
        use crate::host::MemoryMapping;
        use crate::host::{
            ClockService, ConfigService, DynamicLibraryService, HostEnvironment, LibraryVisibility,
        };

        const ELEMENTS: usize = 32;
        const LOGICAL_BYTES: usize = ELEMENTS * core::mem::size_of::<f32>();
        const VULKAN_DMABUF_MEMORY: u32 = 1;
        const VULKAN_DMABUF_SYNC_FD: u32 = 2;
        const VULKAN_DMABUF_FOREIGN: u32 = 4;

        type Capabilities = unsafe extern "C" fn(usize) -> u32;
        type Import = unsafe extern "C" fn(usize, i32, usize) -> crate::ffi::ggml_backend_buffer_t;
        type Ownership = unsafe extern "C" fn(crate::ffi::ggml_backend_buffer_t) -> bool;
        type Free = unsafe extern "C" fn(crate::ffi::ggml_backend_buffer_t);

        unsafe fn symbol<T: Copy>(
            library: &crate::host::BuiltinLibrary,
            name: &core::ffi::CStr,
        ) -> T {
            let pointer = unsafe {
                crate::host::builtin()
                    .libraries()
                    .symbol(library, name)
                    .expect("missing external API symbol")
            };
            assert_eq!(core::mem::size_of::<T>(), core::mem::size_of_val(&pointer));
            unsafe { core::mem::transmute_copy(&pointer) }
        }

        fn tensor(role: TensorRole) -> TensorDesc {
            TensorDesc {
                dtype: crate::ffi::GGML_TYPE_F32,
                dims: [
                    Dim::Static(ELEMENTS as i64),
                    Dim::Static(1),
                    Dim::Static(1),
                    Dim::Static(1),
                ],
                strides: [4, LOGICAL_BYTES, LOGICAL_BYTES, LOGICAL_BYTES],
                role,
                data: 0,
                owned: false,
                shared: None,
            }
        }

        let mut add = Node {
            op: crate::ffi::GGML_OP_ADD,
            spec: Err(crate::graph::OpReject::UnsupportedOp),
            inputs: vec![0, 1],
            output: 2,
            op_params: [0; OP_PARAM_BYTES],
        };
        add.spec = OpRegistry::decode_node(&add);
        let graph = GraphIr {
            generation: 1,
            values: vec![
                tensor(TensorRole::RuntimeInput),
                tensor(TensorRole::RuntimeInput),
                tensor(TensorRole::Output),
            ],
            nodes: vec![add],
            inputs: vec![0, 1],
            outputs: vec![2],
        };
        let artifact = super::super::compile::compile_graph_blob(&graph)
            .expect("OpenVINO shim failed to compile vector-add KAT");
        assert!(artifact.len() <= 1024 * 1024);

        let left = core::array::from_fn::<_, ELEMENTS, _>(|index| index as f32 * 0.125 - 1.5);
        let right =
            core::array::from_fn::<_, ELEMENTS, _>(|index| ((index * 7) % 17) as f32 * 0.25 - 2.0);
        let expected = core::array::from_fn::<_, ELEMENTS, _>(|index| left[index] + right[index]);
        let left_bytes =
            unsafe { core::slice::from_raw_parts(left.as_ptr().cast::<u8>(), LOGICAL_BYTES) };
        let right_bytes =
            unsafe { core::slice::from_raw_parts(right.as_ptr().cast::<u8>(), LOGICAL_BYTES) };

        let device = IvpuDevice::open_first().unwrap();
        let parameters = device.parameters().unwrap();
        let elf = parse_intel_elf(&artifact).unwrap();
        let hpi = inspect_hpi_requirements(&elf, &parameters).unwrap();
        let prepared = prepare_inference(
            &device,
            &parameters,
            &elf,
            &hpi,
            &[left_bytes, right_bytes],
            PreparationBudget {
                max_user_bo_bytes: 64 * 1024 * 1024,
                max_total_accounted_bytes: 64 * 1024 * 1024,
            },
        )
        .unwrap();
        assert!(prepared
            .bindings
            .iter()
            .all(|binding| binding.backing == BindingBacking::ImportedShared));
        let queue = device
            .create_queue(DRM_IVPU_JOB_PRIORITY_DEFAULT, 0)
            .unwrap();
        prepared.zero_outputs().unwrap();
        release_host_writes();
        queue
            .submit(&prepared.handles, prepared.commands_offset(), 0)
            .unwrap();
        let deadline = crate::host::builtin()
            .clock()
            .monotonic_now_ns()
            .unwrap()
            .saturating_add(2_000_000_000);
        assert_eq!(
            prepared
                .command
                .wait_idle(Deadline::MonotonicNs(deadline))
                .unwrap(),
            DRM_IVPU_JOB_STATUS_SUCCESS
        );

        let (shared, logical_bytes, backing) = prepared.output_shared_allocation(0).unwrap();
        assert_eq!(logical_bytes, LOGICAL_BYTES);
        assert_eq!(backing, BindingBacking::ImportedShared);
        let descriptor = shared.info();
        let allocation_bytes = descriptor.bytes as usize;
        assert!(allocation_bytes >= LOGICAL_BYTES + 4);

        let host = crate::host::builtin();
        let base_path = host
            .config()
            .value(c"GGML_VULKAN_BASE_LIBRARY")
            .expect("set GGML_VULKAN_BASE_LIBRARY to libggml-base.so");
        let vulkan_path = host
            .config()
            .value(c"GGML_VULKAN_LIBRARY")
            .expect("set GGML_VULKAN_LIBRARY to libggml-vulkan.so");
        let base_library = host
            .libraries()
            .open(base_path, LibraryVisibility::Process)
            .unwrap();
        let vulkan_library = host
            .libraries()
            .open(vulkan_path, LibraryVisibility::Process)
            .unwrap();
        let capabilities: Capabilities =
            unsafe { symbol(&vulkan_library, c"ggml_backend_vk_dmabuf_capabilities") };
        let import: Import =
            unsafe { symbol(&vulkan_library, c"ggml_backend_vk_buffer_from_dmabuf") };
        let acquire: Ownership =
            unsafe { symbol(&vulkan_library, c"ggml_backend_vk_buffer_dmabuf_acquire") };
        let release: Ownership =
            unsafe { symbol(&vulkan_library, c"ggml_backend_vk_buffer_dmabuf_release") };
        let free_buffer: Free = unsafe { symbol(&base_library, c"ggml_backend_buffer_free") };

        let caps = unsafe { capabilities(0) };
        assert_eq!(
            caps & (VULKAN_DMABUF_MEMORY | VULKAN_DMABUF_SYNC_FD | VULKAN_DMABUF_FOREIGN),
            VULKAN_DMABUF_MEMORY | VULKAN_DMABUF_SYNC_FD | VULKAN_DMABUF_FOREIGN
        );
        let vk_buffer =
            unsafe { import(0, shared.linux_fd_for_external_api_test(), allocation_bytes) };
        assert!(!vk_buffer.is_null());
        assert!(unsafe { acquire(vk_buffer) });

        // Preserve the IVPU-written logical result and have Vulkan fill only the page tail. This is
        // a device command on the imported allocation, not a host staging copy.
        let mut tail_tensor: crate::ffi::ggml_tensor = unsafe { core::mem::zeroed() };
        let get_base = unsafe { (*vk_buffer).iface.get_base.unwrap() };
        let base = unsafe { get_base(vk_buffer) }.cast::<u8>();
        tail_tensor.buffer = vk_buffer;
        tail_tensor.data = unsafe { base.add(LOGICAL_BYTES) }.cast();
        let memset = unsafe { (*vk_buffer).iface.memset_tensor.unwrap() };
        unsafe {
            memset(
                vk_buffer,
                &mut tail_tensor,
                0xa5,
                0,
                allocation_bytes - LOGICAL_BYTES,
            )
        };
        assert!(unsafe { release(vk_buffer) });
        unsafe { free_buffer(vk_buffer) };

        let mapping = shared.map(AccessMode::Read).unwrap();
        let output = unsafe {
            core::slice::from_raw_parts(mapping.address().as_ptr().cast::<f32>(), ELEMENTS)
        };
        assert_eq!(
            core::array::from_fn::<_, ELEMENTS, _>(|index| output[index].to_bits()),
            expected.map(f32::to_bits)
        );
        let tail = unsafe {
            core::slice::from_raw_parts(
                mapping.address().as_ptr().add(LOGICAL_BYTES),
                allocation_bytes - LOGICAL_BYTES,
            )
        };
        assert!(tail.iter().all(|byte| *byte == 0xa5));
        drop(mapping);

        std::println!(
            "same-allocation interop identity={descriptor:?} logical_bytes={logical_bytes} allocation_bytes={allocation_bytes} ivpu_vpu_address={:#x} vulkan_caps={caps:#x} path=CPU-map->IVPU-submit/wait->Vulkan-import/acquire/fill/release->CPU-map staged_bytes=0",
            prepared
                .bindings
                .iter()
                .find(|binding| binding.class == super::link::DeviceMemoryClass::Output)
                .unwrap()
                .bo
                .vpu_address()
        );

        drop(vulkan_library);
        drop(base_library);
        queue.destroy().unwrap();
        prepared.destroy();
        drop(device);
    }
}
