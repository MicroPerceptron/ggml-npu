//! Portable runtime contracts.
//!
//! These traits describe ownership and lifecycle, not a vendor command protocol. Implementations
//! may use Level Zero, a direct kernel UAPI, XRT, QNN, or another OS service without exposing those
//! handles to graph lowering. All submission bindings are immutable for the duration of submit.

#![allow(dead_code)]

pub mod artifact;

use core::fmt;
use core::num::NonZeroU64;

pub const MAX_BINDINGS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct DeviceId(NonZeroU64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct MemoryId(NonZeroU64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ProgramId(NonZeroU64);

macro_rules! impl_id {
    ($name:ident) => {
        impl $name {
            pub const fn new(value: u64) -> Option<Self> {
                match NonZeroU64::new(value) {
                    Some(value) => Some(Self(value)),
                    None => None,
                }
            }

            pub const fn get(self) -> u64 {
                self.0.get()
            }
        }
    };
}
impl_id!(DeviceId);
impl_id!(MemoryId);
impl_id!(ProgramId);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeError {
    Unsupported,
    Incompatible,
    InvalidArgument,
    DeadlineExpired,
    OutOfMemory,
    ResourceLimit,
    DeviceLost,
    Os(i32),
    Vendor { domain: u32, code: i64 },
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Deadline {
    Infinite,
    MonotonicNs(u64),
}

impl Deadline {
    pub const fn expired_at(self, now_ns: u64) -> bool {
        matches!(self, Self::MonotonicNs(deadline) if now_ns >= deadline)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryLocation {
    Host,
    Device,
    Shared,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRequest {
    pub bytes: usize,
    pub alignment: usize,
    pub location: MemoryLocation,
    pub host_visible: bool,
}

impl MemoryRequest {
    pub fn validate(self, max_bytes: usize) -> Result<Self, RuntimeError> {
        if self.bytes == 0
            || self.bytes > max_bytes
            || self.alignment == 0
            || !self.alignment.is_power_of_two()
        {
            return Err(RuntimeError::InvalidArgument);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    /// Zero-based slot from the artifact's binding ABI.
    pub slot: u32,
    pub memory: MemoryId,
    pub offset: u64,
    pub bytes: u64,
    pub access: BindingAccess,
}

/// Validated immutable bindings. A backend may retain this slice only until `submit` returns.
pub struct SubmissionBindings<'a> {
    entries: &'a [Binding],
}

impl<'a> SubmissionBindings<'a> {
    pub fn new(entries: &'a [Binding]) -> Result<Self, RuntimeError> {
        if entries.len() > MAX_BINDINGS {
            return Err(RuntimeError::ResourceLimit);
        }
        for (index, binding) in entries.iter().enumerate() {
            if binding.bytes == 0 || binding.offset.checked_add(binding.bytes).is_none() {
                return Err(RuntimeError::InvalidArgument);
            }
            if entries[..index]
                .iter()
                .any(|prior| prior.slot == binding.slot)
            {
                return Err(RuntimeError::InvalidArgument);
            }
        }
        Ok(Self { entries })
    }

    pub const fn entries(&self) -> &[Binding] {
        self.entries
    }
}

/// Runtime lifecycle seam. Associated types preserve native ownership without boxing hot handles.
///
/// Destructive methods consume their handle, making unload/free/queue destruction explicit even
/// when an implementation also supplies a defensive `Drop`.
pub trait RuntimeHal {
    type Device;
    type Memory;
    type Program;
    type Queue;
    type Event;

    fn open_device(&self, id: DeviceId) -> Result<Self::Device, RuntimeError>;
    fn allocate(
        &self,
        device: &Self::Device,
        request: MemoryRequest,
    ) -> Result<Self::Memory, RuntimeError>;
    fn load_program(
        &self,
        device: &Self::Device,
        artifact: &artifact::ArtifactEnvelope<'_>,
    ) -> Result<Self::Program, RuntimeError>;
    fn create_queue(&self, device: &Self::Device) -> Result<Self::Queue, RuntimeError>;
    fn submit(
        &self,
        queue: &Self::Queue,
        program: &Self::Program,
        bindings: &SubmissionBindings<'_>,
        deadline: Deadline,
    ) -> Result<Self::Event, RuntimeError>;
    fn wait(&self, event: &Self::Event, deadline: Deadline) -> Result<(), RuntimeError>;
    fn map<'a>(&self, memory: &'a mut Self::Memory) -> Result<&'a mut [u8], RuntimeError>;
    fn unmap(&self, memory: &mut Self::Memory) -> Result<(), RuntimeError>;
    fn destroy_event(&self, event: Self::Event) -> Result<(), RuntimeError>;
    fn destroy_queue(&self, queue: Self::Queue) -> Result<(), RuntimeError>;
    fn unload_program(&self, program: Self::Program) -> Result<(), RuntimeError>;
    fn free_memory(&self, memory: Self::Memory) -> Result<(), RuntimeError>;
    fn close_device(&self, device: Self::Device) -> Result<(), RuntimeError>;
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use super::*;

    #[test]
    fn bindings_are_bounded_unique_and_nonempty() {
        let memory = MemoryId::new(1).unwrap();
        let valid = [Binding {
            slot: 0,
            memory,
            offset: 4,
            bytes: 8,
            access: BindingAccess::Read,
        }];
        assert_eq!(SubmissionBindings::new(&valid).unwrap().entries(), valid);

        let duplicate = [valid[0], valid[0]];
        assert_eq!(
            SubmissionBindings::new(&duplicate).err(),
            Some(RuntimeError::InvalidArgument)
        );
    }

    struct FakeMemory([u8; 16]);
    struct FakeRuntime {
        destroyed: Cell<u32>,
    }

    impl RuntimeHal for FakeRuntime {
        type Device = DeviceId;
        type Memory = FakeMemory;
        type Program = ProgramId;
        type Queue = u32;
        type Event = u32;

        fn open_device(&self, id: DeviceId) -> Result<Self::Device, RuntimeError> {
            Ok(id)
        }
        fn allocate(
            &self,
            _device: &Self::Device,
            request: MemoryRequest,
        ) -> Result<Self::Memory, RuntimeError> {
            request.validate(16)?;
            Ok(FakeMemory([0; 16]))
        }
        fn load_program(
            &self,
            _device: &Self::Device,
            artifact: &artifact::ArtifactEnvelope<'_>,
        ) -> Result<Self::Program, RuntimeError> {
            ProgramId::new(artifact.header().module_id).ok_or(RuntimeError::InvalidArgument)
        }
        fn create_queue(&self, _device: &Self::Device) -> Result<Self::Queue, RuntimeError> {
            Ok(1)
        }
        fn submit(
            &self,
            _queue: &Self::Queue,
            _program: &Self::Program,
            bindings: &SubmissionBindings<'_>,
            deadline: Deadline,
        ) -> Result<Self::Event, RuntimeError> {
            if bindings.entries().is_empty() || deadline.expired_at(10) {
                return Err(RuntimeError::DeadlineExpired);
            }
            Ok(1)
        }
        fn wait(&self, _event: &Self::Event, deadline: Deadline) -> Result<(), RuntimeError> {
            (!deadline.expired_at(10))
                .then_some(())
                .ok_or(RuntimeError::DeadlineExpired)
        }
        fn map<'a>(&self, memory: &'a mut Self::Memory) -> Result<&'a mut [u8], RuntimeError> {
            Ok(&mut memory.0)
        }
        fn unmap(&self, _memory: &mut Self::Memory) -> Result<(), RuntimeError> {
            Ok(())
        }
        fn destroy_event(&self, _event: Self::Event) -> Result<(), RuntimeError> {
            self.destroyed.set(self.destroyed.get() | 1);
            Ok(())
        }
        fn destroy_queue(&self, _queue: Self::Queue) -> Result<(), RuntimeError> {
            self.destroyed.set(self.destroyed.get() | 2);
            Ok(())
        }
        fn unload_program(&self, _program: Self::Program) -> Result<(), RuntimeError> {
            self.destroyed.set(self.destroyed.get() | 4);
            Ok(())
        }
        fn free_memory(&self, _memory: Self::Memory) -> Result<(), RuntimeError> {
            self.destroyed.set(self.destroyed.get() | 8);
            Ok(())
        }
        fn close_device(&self, _device: Self::Device) -> Result<(), RuntimeError> {
            self.destroyed.set(self.destroyed.get() | 16);
            Ok(())
        }
    }

    #[test]
    fn fake_runtime_conforms_to_explicit_lifecycle() {
        use artifact::{
            ArtifactEnvelope, ArtifactHeader, BindingAbi, NativePayloadKind, TargetFingerprint,
            ARTIFACT_ABI_MAJOR, ARTIFACT_MAGIC,
        };

        let runtime = FakeRuntime {
            destroyed: Cell::new(0),
        };
        let device = runtime.open_device(DeviceId::new(1).unwrap()).unwrap();
        let mut memory = runtime
            .allocate(
                &device,
                MemoryRequest {
                    bytes: 16,
                    alignment: 8,
                    location: MemoryLocation::Shared,
                    host_visible: true,
                },
            )
            .unwrap();
        runtime.map(&mut memory).unwrap()[0] = 7;
        runtime.unmap(&mut memory).unwrap();

        let payload = [1u8];
        let artifact = ArtifactEnvelope::new(
            ArtifactHeader {
                magic: ARTIFACT_MAGIC,
                envelope_major: ARTIFACT_ABI_MAJOR,
                envelope_minor: 0,
                header_bytes: core::mem::size_of::<ArtifactHeader>() as u32,
                module_id: 1,
                target: TargetFingerprint {
                    vendor_id: 0,
                    device_id: 0,
                    revision: 0,
                    architecture: 0,
                    firmware_api: [0; 8],
                },
                payload_kind: NativePayloadKind::CANONICAL_MLIR_BYTECODE,
                binding_abi: BindingAbi {
                    major: 1,
                    minor: 0,
                    input_count: 1,
                    output_count: 0,
                    scratch_count: 0,
                    reserved: 0,
                },
                payload_bytes: 1,
                resident_bytes: 16,
                flags: 0,
            },
            &payload,
        )
        .unwrap();
        let program = runtime.load_program(&device, &artifact).unwrap();
        let queue = runtime.create_queue(&device).unwrap();
        let entries = [Binding {
            slot: 0,
            memory: MemoryId::new(1).unwrap(),
            offset: 0,
            bytes: 16,
            access: BindingAccess::Read,
        }];
        let event = runtime
            .submit(
                &queue,
                &program,
                &SubmissionBindings::new(&entries).unwrap(),
                Deadline::MonotonicNs(11),
            )
            .unwrap();
        runtime.wait(&event, Deadline::Infinite).unwrap();
        runtime.destroy_event(event).unwrap();
        runtime.destroy_queue(queue).unwrap();
        runtime.unload_program(program).unwrap();
        runtime.free_memory(memory).unwrap();
        runtime.close_device(device).unwrap();
        assert_eq!(runtime.destroyed.get(), 31);
    }
}
