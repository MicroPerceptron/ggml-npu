//! Transport-neutral IVPU resource lifecycle.
//!
//! Compiler and artifact code depends on this trait, not rustix, DRM, VFIO, or an OS handle.

#![allow(dead_code)]

use core::sync::atomic::{fence, Ordering};

use crate::runtime::{Deadline, RuntimeError};

use super::{BoMapping, BufferInfo, BufferObject, CommandQueue, DeviceParameters, IvpuDevice};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferRole {
    Command,
    Program,
    Binding,
    Scratch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferSpec {
    pub role: BufferRole,
    pub bytes: u64,
    pub host_visible: bool,
    pub read_only: bool,
}

impl BufferSpec {
    pub fn validate(self) -> Result<Self, RuntimeError> {
        if self.bytes == 0 || self.bytes > usize::MAX as u64 {
            return Err(RuntimeError::ResourceLimit);
        }
        if self.role == BufferRole::Command && (!self.host_visible || self.read_only) {
            return Err(RuntimeError::InvalidArgument);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueSpec {
    pub priority: u32,
    pub turbo: bool,
}

pub trait IvpuTransport {
    type Device;
    type Buffer<'device>
    where
        Self: 'device;
    type Mapping<'buffer>
    where
        Self: 'buffer;
    type Queue<'device>
    where
        Self: 'device;

    fn open(&self, index: usize) -> Result<Self::Device, RuntimeError>;
    fn parameters(&self, device: &Self::Device) -> Result<DeviceParameters, RuntimeError>;
    fn create_buffer<'device>(
        &'device self,
        device: &'device Self::Device,
        spec: BufferSpec,
    ) -> Result<Self::Buffer<'device>, RuntimeError>;
    fn buffer_info(&self, buffer: &Self::Buffer<'_>) -> Result<BufferInfo, RuntimeError>;
    fn map_buffer<'buffer>(
        &'buffer self,
        buffer: &'buffer Self::Buffer<'_>,
        writable: bool,
    ) -> Result<Self::Mapping<'buffer>, RuntimeError>;
    fn create_queue<'device>(
        &'device self,
        device: &'device Self::Device,
        spec: QueueSpec,
    ) -> Result<Self::Queue<'device>, RuntimeError>;
    fn wait_buffer(
        &self,
        buffer: &Self::Buffer<'_>,
        deadline: Deadline,
    ) -> Result<u32, RuntimeError>;
    fn destroy_buffer(&self, buffer: Self::Buffer<'_>) -> Result<(), RuntimeError>;
    fn destroy_queue(&self, queue: Self::Queue<'_>) -> Result<(), RuntimeError>;
}

#[derive(Clone, Copy, Default)]
pub struct AccelTransport;

impl IvpuTransport for AccelTransport {
    type Device = IvpuDevice;
    type Buffer<'device> = BufferObject<'device>;
    type Mapping<'buffer> = BoMapping<'buffer>;
    type Queue<'device> = CommandQueue<'device>;

    fn open(&self, index: usize) -> Result<Self::Device, RuntimeError> {
        IvpuDevice::open_index(index)
    }

    fn parameters(&self, device: &Self::Device) -> Result<DeviceParameters, RuntimeError> {
        device.parameters()
    }

    fn create_buffer<'device>(
        &'device self,
        device: &'device Self::Device,
        spec: BufferSpec,
    ) -> Result<Self::Buffer<'device>, RuntimeError> {
        let spec = spec.validate()?;
        let mut flags = if spec.host_visible {
            super::uapi::DRM_IVPU_BO_MAPPABLE
        } else {
            0
        };
        if spec.read_only {
            flags |= super::uapi::DRM_IVPU_BO_READ_ONLY;
        }
        device.create_bo(spec.bytes, flags)
    }

    fn buffer_info(&self, buffer: &Self::Buffer<'_>) -> Result<BufferInfo, RuntimeError> {
        buffer.info()
    }

    fn map_buffer<'buffer>(
        &'buffer self,
        buffer: &'buffer Self::Buffer<'_>,
        writable: bool,
    ) -> Result<Self::Mapping<'buffer>, RuntimeError> {
        buffer.map(writable)
    }

    fn create_queue<'device>(
        &'device self,
        device: &'device Self::Device,
        spec: QueueSpec,
    ) -> Result<Self::Queue<'device>, RuntimeError> {
        device.create_queue(spec.priority, u32::from(spec.turbo))
    }

    fn wait_buffer(
        &self,
        buffer: &Self::Buffer<'_>,
        deadline: Deadline,
    ) -> Result<u32, RuntimeError> {
        buffer.wait_idle(deadline)
    }

    fn destroy_buffer(&self, buffer: Self::Buffer<'_>) -> Result<(), RuntimeError> {
        buffer.destroy()
    }

    fn destroy_queue(&self, queue: Self::Queue<'_>) -> Result<(), RuntimeError> {
        queue.destroy()
    }
}

/// Cached mappings are host coherent according to the IVPU UAPI. These barriers order CPU access
/// around a future ioctl; they are not a substitute for vendor protocol synchronization.
pub fn release_host_writes() {
    fence(Ordering::Release);
}

pub fn acquire_device_writes() {
    fence(Ordering::Acquire);
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use super::*;

    struct FakeBuffer {
        bytes: u64,
    }
    struct FakeMapping;
    struct FakeQueue;
    struct FakeTransport {
        fault: Cell<Option<RuntimeError>>,
        cleanup: Cell<u32>,
    }

    impl IvpuTransport for FakeTransport {
        type Device = ();
        type Buffer<'device> = FakeBuffer;
        type Mapping<'buffer> = FakeMapping;
        type Queue<'device> = FakeQueue;

        fn open(&self, _index: usize) -> Result<Self::Device, RuntimeError> {
            match self.fault.get() {
                Some(error) => Err(error),
                None => Ok(()),
            }
        }
        fn parameters(&self, _device: &Self::Device) -> Result<DeviceParameters, RuntimeError> {
            Err(RuntimeError::Unsupported)
        }
        fn create_buffer<'device>(
            &'device self,
            _device: &'device Self::Device,
            spec: BufferSpec,
        ) -> Result<Self::Buffer<'device>, RuntimeError> {
            spec.validate()?;
            match self.fault.get() {
                Some(error) => Err(error),
                None => Ok(FakeBuffer { bytes: spec.bytes }),
            }
        }
        fn buffer_info(&self, buffer: &Self::Buffer<'_>) -> Result<BufferInfo, RuntimeError> {
            Ok(BufferInfo {
                flags: 0,
                vpu_address: 0x1000,
                mmap_offset: 0,
                bytes: buffer.bytes,
            })
        }
        fn map_buffer<'buffer>(
            &'buffer self,
            _buffer: &'buffer Self::Buffer<'_>,
            _writable: bool,
        ) -> Result<Self::Mapping<'buffer>, RuntimeError> {
            Ok(FakeMapping)
        }
        fn create_queue<'device>(
            &'device self,
            _device: &'device Self::Device,
            _spec: QueueSpec,
        ) -> Result<Self::Queue<'device>, RuntimeError> {
            match self.fault.get() {
                Some(error) => Err(error),
                None => Ok(FakeQueue),
            }
        }
        fn wait_buffer(
            &self,
            _buffer: &Self::Buffer<'_>,
            deadline: Deadline,
        ) -> Result<u32, RuntimeError> {
            if deadline == Deadline::MonotonicNs(0) {
                return Err(RuntimeError::DeadlineExpired);
            }
            match self.fault.get() {
                Some(error) => Err(error),
                None => Ok(0),
            }
        }
        fn destroy_buffer(&self, _buffer: Self::Buffer<'_>) -> Result<(), RuntimeError> {
            self.cleanup.set(self.cleanup.get() | 1);
            Ok(())
        }
        fn destroy_queue(&self, _queue: Self::Queue<'_>) -> Result<(), RuntimeError> {
            self.cleanup.set(self.cleanup.get() | 2);
            Ok(())
        }
    }

    #[test]
    fn fake_transport_faults_are_bounded_and_cleanup_is_deterministic() {
        let transport = FakeTransport {
            fault: Cell::new(None),
            cleanup: Cell::new(0),
        };
        let device = transport.open(0).unwrap();
        let buffer = transport
            .create_buffer(
                &device,
                BufferSpec {
                    role: BufferRole::Command,
                    bytes: 4096,
                    host_visible: true,
                    read_only: false,
                },
            )
            .unwrap();
        assert_eq!(
            transport.wait_buffer(&buffer, Deadline::MonotonicNs(0)),
            Err(RuntimeError::DeadlineExpired)
        );
        transport.fault.set(Some(RuntimeError::OutOfMemory));
        assert!(matches!(
            transport.create_queue(
                &device,
                QueueSpec {
                    priority: 0,
                    turbo: false,
                },
            ),
            Err(RuntimeError::OutOfMemory)
        ));
        transport.fault.set(None);
        let queue = transport
            .create_queue(
                &device,
                QueueSpec {
                    priority: 0,
                    turbo: false,
                },
            )
            .unwrap();
        transport.destroy_queue(queue).unwrap();
        transport.destroy_buffer(buffer).unwrap();
        assert_eq!(transport.cleanup.get(), 3);

        for fault in [
            RuntimeError::OutOfMemory,
            RuntimeError::DeviceLost,
            RuntimeError::External {
                domain: super::super::IVPU_ERROR_DOMAIN,
                code: 5,
            },
        ] {
            transport.fault.set(Some(fault));
            assert_eq!(transport.open(0), Err(fault));
        }
    }
}
