//! Direct Linux `intel_vpu` foundation.
//!
//! Command submission is restricted to validated, opt-in tiny hardware tests.

#![allow(dead_code)]

pub mod budget;
pub mod elf;
mod firmware;
pub mod hpi;
pub mod link;
pub mod load;
pub mod prepare;
pub mod transport;
mod uapi;
pub mod validate;

use core::marker::PhantomData;

use rustix::ioctl;

use crate::os::linux::{LinuxFile, LinuxMapping, LinuxOs};
use crate::os::{MapOptions, MemoryMapping, OpenAccess, OpenOptions, OsServices, OwnedDeviceFile};
use crate::runtime::{
    artifact::ArtifactEnvelope, Deadline, DeviceId, MemoryRequest, RuntimeError, RuntimeHal,
    SubmissionBindings,
};

use uapi::*;

use self::transport::{AccelTransport, IvpuTransport};

const DEVICE_PATHS: [&core::ffi::CStr; 8] = [
    c"/dev/accel/accel0",
    c"/dev/accel/accel1",
    c"/dev/accel/accel2",
    c"/dev/accel/accel3",
    c"/dev/accel/accel4",
    c"/dev/accel/accel5",
    c"/dev/accel/accel6",
    c"/dev/accel/accel7",
];

fn runtime_error(error: crate::os::OsError) -> RuntimeError {
    RuntimeError::Os(error.0)
}

fn ioctl_error(error: rustix::io::Errno) -> RuntimeError {
    match error {
        rustix::io::Errno::NOMEM => RuntimeError::OutOfMemory,
        rustix::io::Errno::NODEV | rustix::io::Errno::IO | rustix::io::Errno::BADFD => {
            RuntimeError::DeviceLost
        }
        rustix::io::Errno::TIME | rustix::io::Errno::TIMEDOUT => RuntimeError::DeadlineExpired,
        _ => RuntimeError::Os(error.raw_os_error()),
    }
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

pub struct IvpuDevice {
    file: LinuxFile,
    path_index: usize,
}

impl IvpuDevice {
    pub fn open_first() -> Result<Self, RuntimeError> {
        let mut last = RuntimeError::Unsupported;
        for index in 0..DEVICE_PATHS.len() {
            match Self::open_index(index) {
                Ok(device) => return Ok(device),
                Err(error) => last = error,
            }
        }
        Err(last)
    }

    pub fn open_index(index: usize) -> Result<Self, RuntimeError> {
        let path = DEVICE_PATHS
            .get(index)
            .copied()
            .ok_or(RuntimeError::InvalidArgument)?;
        let os = LinuxOs;
        os.open_device(
            path,
            OpenOptions {
                access: OpenAccess::ReadWrite,
                nonblocking: true,
            },
        )
        .map(|file| Self {
            file,
            path_index: index,
        })
        .map_err(runtime_error)
    }

    pub fn device_path(&self) -> &'static core::ffi::CStr {
        DEVICE_PATHS[self.path_index]
    }

    pub fn raw_handle(&self) -> i64 {
        self.file.raw_handle()
    }

    /// Read-only kernel parameter query.
    pub fn get_param(&self, param: u32, index: u32) -> Result<u64, RuntimeError> {
        let mut value = DrmIvpuParam {
            param,
            index,
            value: 0,
        };
        ioctl_get_param(&self.file, &mut value)?;
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
        ioctl_cmdq_create(&self.file, &mut request)?;
        if request.cmdq_id == 0 {
            return Err(RuntimeError::Vendor {
                domain: 0x4956_5055,
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
        ioctl_bo_create(&self.file, &mut request)?;
        if request.handle == 0 || request.vpu_addr == 0 {
            return Err(RuntimeError::Vendor {
                domain: 0x4956_5055,
                code: -1,
            });
        }
        Ok(BufferObject {
            device: self,
            handle: request.handle,
            requested_bytes: bytes,
            vpu_address: request.vpu_addr,
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

pub struct BufferObject<'device> {
    device: &'device IvpuDevice,
    handle: u32,
    requested_bytes: u64,
    vpu_address: u64,
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

    pub fn info(&self) -> Result<BufferInfo, RuntimeError> {
        let mut info = DrmIvpuBoInfo {
            handle: self.handle,
            ..DrmIvpuBoInfo::default()
        };
        ioctl_bo_info(&self.device.file, &mut info)?;
        if info.handle != self.handle || info.size < self.requested_bytes {
            return Err(RuntimeError::Vendor {
                domain: 0x4956_5055,
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
        let info = self.info()?;
        if info.mmap_offset == 0 || info.bytes == 0 || info.bytes > usize::MAX as u64 {
            return Err(RuntimeError::InvalidArgument);
        }
        let mapping = LinuxOs
            .map_file(
                &self.device.file,
                MapOptions {
                    offset: info.mmap_offset,
                    bytes: info.bytes as usize,
                    readable: true,
                    writable,
                    shared: true,
                },
            )
            .map_err(runtime_error)?;
        Ok(BoMapping {
            mapping,
            _owner: PhantomData,
        })
    }

    pub fn destroy(mut self) -> Result<(), RuntimeError> {
        let handle = self.handle;
        self.handle = 0;
        ioctl_gem_close(&self.device.file, handle)
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
        ioctl_bo_wait(&self.device.file, &mut request)?;
        Ok(request.job_status)
    }
}

impl Drop for BufferObject<'_> {
    fn drop(&mut self) {
        if self.handle != 0 {
            let _ = ioctl_gem_close(&self.device.file, self.handle);
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
        ioctl_cmdq_submit(&self.device.file, value)
    }

    pub fn destroy(mut self) -> Result<(), RuntimeError> {
        let id = self.id;
        self.id = 0;
        ioctl_cmdq_destroy(&self.device.file, id)
    }
}

impl Drop for CommandQueue<'_> {
    fn drop(&mut self) {
        if self.id != 0 {
            let _ = ioctl_cmdq_destroy(&self.device.file, self.id);
            self.id = 0;
        }
    }
}

pub struct BoMapping<'bo> {
    mapping: LinuxMapping,
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

/// SAFETY boundary for DRM_IOCTL_IVPU_GET_PARAM (`_IOWR`, 16-byte repr(C) value).
fn ioctl_get_param(file: &LinuxFile, value: &mut DrmIvpuParam) -> Result<(), RuntimeError> {
    // SAFETY: request number, direction, type size, and alignment are asserted in uapi tests.
    unsafe {
        ioctl::ioctl(
            file.as_fd(),
            ioctl::Updater::<{ DRM_IOCTL_IVPU_GET_PARAM }, DrmIvpuParam>::new(value),
        )
    }
    .map_err(ioctl_error)
}

/// SAFETY boundary for DRM_IOCTL_IVPU_BO_CREATE (`_IOWR`, 24-byte repr(C) value).
fn ioctl_bo_create(file: &LinuxFile, value: &mut DrmIvpuBoCreate) -> Result<(), RuntimeError> {
    // SAFETY: request number, direction, type size, and alignment are asserted in uapi tests.
    unsafe {
        ioctl::ioctl(
            file.as_fd(),
            ioctl::Updater::<{ DRM_IOCTL_IVPU_BO_CREATE }, DrmIvpuBoCreate>::new(value),
        )
    }
    .map_err(ioctl_error)
}

/// SAFETY boundary for DRM_IOCTL_IVPU_BO_INFO (`_IOWR`, 32-byte repr(C) value).
fn ioctl_bo_info(file: &LinuxFile, value: &mut DrmIvpuBoInfo) -> Result<(), RuntimeError> {
    // SAFETY: request number, direction, type size, and alignment are asserted in uapi tests.
    unsafe {
        ioctl::ioctl(
            file.as_fd(),
            ioctl::Updater::<{ DRM_IOCTL_IVPU_BO_INFO }, DrmIvpuBoInfo>::new(value),
        )
    }
    .map_err(ioctl_error)
}

/// SAFETY boundary for DRM_IOCTL_IVPU_BO_WAIT (`_IOWR`, 24-byte repr(C) value).
fn ioctl_bo_wait(file: &LinuxFile, value: &mut DrmIvpuBoWait) -> Result<(), RuntimeError> {
    // SAFETY: request number, direction, type size, and alignment are asserted in uapi tests.
    unsafe {
        ioctl::ioctl(
            file.as_fd(),
            ioctl::Updater::<{ DRM_IOCTL_IVPU_BO_WAIT }, DrmIvpuBoWait>::new(value),
        )
    }
    .map_err(ioctl_error)
}

/// SAFETY boundary for DRM_IOCTL_IVPU_CMDQ_CREATE (`_IOWR`, 12-byte repr(C) value).
fn ioctl_cmdq_create(file: &LinuxFile, value: &mut DrmIvpuCmdqCreate) -> Result<(), RuntimeError> {
    // SAFETY: request number, direction, type size, and alignment are asserted in uapi tests.
    unsafe {
        ioctl::ioctl(
            file.as_fd(),
            ioctl::Updater::<{ DRM_IOCTL_IVPU_CMDQ_CREATE }, DrmIvpuCmdqCreate>::new(value),
        )
    }
    .map_err(ioctl_error)
}

/// SAFETY boundary for DRM_IOCTL_IVPU_CMDQ_DESTROY (`_IOW`, 4-byte repr(C) value).
fn ioctl_cmdq_destroy(file: &LinuxFile, id: u32) -> Result<(), RuntimeError> {
    let value = DrmIvpuCmdqDestroy { cmdq_id: id };
    // SAFETY: request number, direction, type size, and alignment are asserted in uapi tests.
    unsafe {
        ioctl::ioctl(
            file.as_fd(),
            ioctl::Setter::<{ DRM_IOCTL_IVPU_CMDQ_DESTROY }, DrmIvpuCmdqDestroy>::new(value),
        )
    }
    .map_err(ioctl_error)
}

/// SAFETY boundary for DRM_IOCTL_IVPU_CMDQ_SUBMIT (`_IOW`, 32-byte repr(C) value).
fn ioctl_cmdq_submit(file: &LinuxFile, value: DrmIvpuCmdqSubmit) -> Result<(), RuntimeError> {
    // SAFETY: the request number, direction, value layout, and pointer lifetime are checked by
    // the UAPI assertions and CommandQueue::submit; ioctl consumes the handle array synchronously.
    unsafe {
        ioctl::ioctl(
            file.as_fd(),
            ioctl::Setter::<{ DRM_IOCTL_IVPU_CMDQ_SUBMIT }, DrmIvpuCmdqSubmit>::new(value),
        )
    }
    .map_err(ioctl_error)
}

/// SAFETY boundary for DRM_IOCTL_GEM_CLOSE (`_IOW`, 8-byte repr(C) value).
fn ioctl_gem_close(file: &LinuxFile, handle: u32) -> Result<(), RuntimeError> {
    let value = DrmGemClose { handle, pad: 0 };
    // SAFETY: request number, direction, type size, and alignment are asserted in uapi tests.
    unsafe {
        ioctl::ioctl(
            file.as_fd(),
            ioctl::Setter::<{ DRM_IOCTL_GEM_CLOSE }, DrmGemClose>::new(value),
        )
    }
    .map_err(ioctl_error)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "bench-hardware")]
    use std::io::Read;

    use super::*;

    #[cfg(feature = "bench-hardware")]
    #[derive(Debug)]
    struct ResourceSnapshot {
        open_fds: usize,
        rss_kib: Option<u64>,
        swap_kib: Option<u64>,
        available_kib: Option<u64>,
    }

    #[cfg(feature = "bench-hardware")]
    fn proc_kib(path: &str, key: &str) -> Option<u64> {
        let mut file = std::fs::File::open(path).ok()?;
        let mut bytes = [0u8; 4096];
        let length = file.read(&mut bytes).ok()?;
        let text = core::str::from_utf8(&bytes[..length]).ok()?;
        text.lines()
            .find(|line| line.starts_with(key))?
            .split_ascii_whitespace()
            .nth(1)?
            .parse()
            .ok()
    }

    #[cfg(feature = "bench-hardware")]
    fn resources() -> ResourceSnapshot {
        ResourceSnapshot {
            open_fds: std::fs::read_dir("/proc/self/fd")
                .map(|entries| entries.count())
                .unwrap_or(0),
            rss_kib: proc_kib("/proc/self/status", "VmRSS:"),
            swap_kib: proc_kib("/proc/self/status", "VmSwap:"),
            available_kib: proc_kib("/proc/meminfo", "MemAvailable:"),
        }
    }

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
            device.raw_handle(),
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
        use std::vec::Vec;

        use super::elf::parse_intel_elf;
        use super::hpi::inspect_hpi_requirements;
        use super::prepare::{prepare_inference, PreparationBudget};
        use super::transport::release_host_writes;

        const PATH: &str = "/home/cassmantle/.cache/ggml-npu/ir2-8c91da4914adc7d5-n1.blob";
        const ARTIFACT_LIMIT: usize = 1024 * 1024;
        const ELEMENTS: usize = 32;
        const OUTPUT_BYTES: usize = ELEMENTS * core::mem::size_of::<f32>();
        const MAX_ABS_ERROR: f32 = 1.0e-5;
        const MAX_REL_ERROR: f32 = 1.0e-5;
        const SUBMISSIONS: usize = 2;

        let length = usize::try_from(std::fs::metadata(PATH).unwrap().len()).unwrap();
        assert!(length <= ARTIFACT_LIMIT);
        let mut artifact = Vec::new();
        artifact.try_reserve_exact(length).unwrap();
        artifact.resize(length, 0);
        std::fs::File::open(PATH)
            .unwrap()
            .read_exact(&mut artifact)
            .unwrap();

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
        let (path, queue_id, user_bytes, preempt_bytes, accounted_bytes, firmware_api) = {
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
                let now = LinuxOs.monotonic_now_ns().unwrap();
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
            );
            queue.destroy().unwrap();
            prepared.destroy();
            drop(device);
            observed
        };
        let after = resources();
        let mut max_absolute = 0.0f32;
        let mut max_relative = 0.0f32;
        for (actual, expected) in observed_outputs[0].iter().zip(oracle) {
            let absolute = (actual - expected).abs();
            max_absolute = max_absolute.max(absolute);
            max_relative = max_relative.max(absolute / expected.abs().max(1.0));
        }
        std::println!(
            "device={path} queue_id={queue_id} artifact={PATH} submissions={SUBMISSIONS} user_bo_bytes={user_bytes} preempt_buffer_bytes={preempt_bytes} accounted_bytes={accounted_bytes} firmware_api={firmware_api:?} oracle={oracle:?} direct={:?} max_absolute={max_absolute} max_relative={max_relative} tolerance_abs={MAX_ABS_ERROR} tolerance_rel={MAX_REL_ERROR} elapsed={elapsed:?} before={before:?} after={after:?}",
            observed_outputs[0],
        );
        assert_eq!(after.open_fds, before.open_fds);
        if let (Some(before_rss), Some(after_rss)) = (before.rss_kib, after.rss_kib) {
            assert!(after_rss <= before_rss.saturating_add(4096));
        }
        if let (Some(before_swap), Some(after_swap)) = (before.swap_kib, after.swap_kib) {
            assert!(after_swap <= before_swap);
        }
    }
}
