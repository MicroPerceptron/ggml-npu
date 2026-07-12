//! Allocation-free dry-run validation required before any future submit ioctl.

#![allow(dead_code)]

use crate::runtime::{MemoryId, SubmissionBindings, MAX_BINDINGS};

use super::elf::ArtifactInspection;
use super::firmware::*;
use super::transport::{BufferRole, QueueSpec};
use super::DeviceParameters;

pub const LUNAR_LAKE_DEVICE_ID: u32 = 0x643e;
pub const LUNAR_LAKE_REVISION: u32 = 4;
pub const INTEL_VENDOR_ID: u32 = 0x8086;
pub const MAX_SUBMISSION_BUFFERS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationError {
    WrongDevice,
    WrongRevision,
    FirmwareMismatch,
    UnsupportedPlatform,
    InvalidQueue,
    InvalidHandle,
    DuplicateHandle,
    TooManyBuffers,
    InvalidPreemptionIndex,
    InvalidCommandOffset,
    InvalidCommandHeader,
    InvalidCommandType,
    InvalidCommandLength,
    TooManyCommands,
    InvalidRange,
    InvalidReservedField,
    BindingMismatch,
    UnresolvedRelocations,
    ResidentBytesExceeded,
    ArithmeticOverflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRange {
    pub handle: u32,
    pub role: BufferRole,
    pub vpu_address: u64,
    pub bytes: u64,
}

impl MemoryRange {
    fn contains(self, address: u64, bytes: u64) -> bool {
        let Some(range_end) = self.vpu_address.checked_add(self.bytes) else {
            return false;
        };
        let Some(end) = address.checked_add(bytes) else {
            return false;
        };
        address >= self.vpu_address && end <= range_end
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundMemory {
    pub id: MemoryId,
    pub range: MemoryRange,
}

#[derive(Clone, Copy)]
pub struct CommandImage<'a> {
    pub bytes: &'a [u8],
    pub vpu_address: u64,
    /// Offset supplied to DRM_IOCTL_IVPU_CMDQ_SUBMIT, from the start of this BO.
    pub commands_offset: u32,
}

pub struct SubmissionPlan<'a> {
    pub device: &'a DeviceParameters,
    pub artifact: &'a ArtifactInspection,
    pub queue_id: u32,
    pub queue: QueueSpec,
    pub command_handle: u32,
    pub handles: &'a [u32],
    pub preemption_buffer_index: Option<usize>,
    pub command: CommandImage<'a>,
    pub ranges: &'a [MemoryRange],
    pub bindings: &'a SubmissionBindings<'a>,
    pub bound_memory: &'a [BoundMemory],
}

pub struct ValidatedSubmission<'a> {
    plan: &'a SubmissionPlan<'a>,
    command_count: usize,
}

impl ValidatedSubmission<'_> {
    pub const fn queue_id(&self) -> u32 {
        self.plan.queue_id
    }

    pub const fn handles(&self) -> &[u32] {
        self.plan.handles
    }

    pub const fn commands_offset(&self) -> u32 {
        self.plan.command.commands_offset
    }

    pub const fn command_count(&self) -> usize {
        self.command_count
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, ValidationError> {
    let value = bytes
        .get(offset..offset + 2)
        .ok_or(ValidationError::InvalidCommandLength)?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, ValidationError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or(ValidationError::InvalidCommandLength)?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, ValidationError> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or(ValidationError::InvalidCommandLength)?;
    Ok(u64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}

fn in_any_range(ranges: &[MemoryRange], address: u64, bytes: u64) -> bool {
    ranges.iter().any(|range| range.contains(address, bytes))
}

pub fn validate_device(parameters: &DeviceParameters) -> Result<(), ValidationError> {
    if parameters.device_id != LUNAR_LAKE_DEVICE_ID {
        return Err(ValidationError::WrongDevice);
    }
    if parameters.revision != LUNAR_LAKE_REVISION {
        return Err(ValidationError::WrongRevision);
    }
    if parameters.platform_type != 0 {
        return Err(ValidationError::UnsupportedPlatform);
    }
    if parameters.firmware_api[JOB_CMD_API_INDEX] != JOB_CMD_API_WORD as u64
        || parameters.firmware_api[NNRT_API_INDEX] != NNRT_API_WORD as u64
    {
        return Err(ValidationError::FirmwareMismatch);
    }
    Ok(())
}

fn validate_queue(id: u32, queue: QueueSpec) -> Result<(), ValidationError> {
    if id == 0 || queue.priority > 4 {
        return Err(ValidationError::InvalidQueue);
    }
    Ok(())
}

fn validate_handles(plan: &SubmissionPlan<'_>) -> Result<(), ValidationError> {
    if plan.handles.is_empty() || plan.handles.len() > MAX_SUBMISSION_BUFFERS {
        return Err(ValidationError::TooManyBuffers);
    }
    if plan.command_handle == 0 || plan.handles[0] != plan.command_handle {
        return Err(ValidationError::InvalidHandle);
    }
    let command_bytes =
        u64::try_from(plan.command.bytes.len()).map_err(|_| ValidationError::InvalidRange)?;
    let Some(command_range) = plan
        .ranges
        .iter()
        .find(|range| range.handle == plan.command_handle)
    else {
        return Err(ValidationError::InvalidRange);
    };
    if command_range.role != BufferRole::Command
        || !command_range.contains(plan.command.vpu_address, command_bytes)
    {
        return Err(ValidationError::InvalidRange);
    }
    for (index, &handle) in plan.handles.iter().enumerate() {
        if handle == 0 {
            return Err(ValidationError::InvalidHandle);
        }
        if plan.handles[..index].contains(&handle) {
            return Err(ValidationError::DuplicateHandle);
        }
        if !plan.ranges.iter().any(|range| range.handle == handle) {
            return Err(ValidationError::InvalidRange);
        }
    }
    if let Some(index) = plan.preemption_buffer_index {
        if index == 0 || index >= plan.handles.len() {
            return Err(ValidationError::InvalidPreemptionIndex);
        }
        let handle = plan.handles[index];
        if !plan.ranges.iter().any(|range| {
            range.handle == handle
                && range.role == BufferRole::Scratch
                && range.bytes >= plan.device.preempt_buffer_bytes
        }) {
            return Err(ValidationError::InvalidPreemptionIndex);
        }
    }
    Ok(())
}

fn validate_bindings(plan: &SubmissionPlan<'_>) -> Result<(), ValidationError> {
    if plan.bindings.entries().len() > MAX_BINDINGS {
        return Err(ValidationError::BindingMismatch);
    }
    for binding in plan.bindings.entries() {
        let Some(memory) = plan
            .bound_memory
            .iter()
            .find(|memory| memory.id == binding.memory)
        else {
            return Err(ValidationError::BindingMismatch);
        };
        let end = binding
            .offset
            .checked_add(binding.bytes)
            .ok_or(ValidationError::ArithmeticOverflow)?;
        if end > memory.range.bytes || !plan.handles.contains(&memory.range.handle) {
            return Err(ValidationError::BindingMismatch);
        }
    }
    Ok(())
}

fn validate_artifact(plan: &SubmissionPlan<'_>) -> Result<(), ValidationError> {
    if plan.artifact.target.vendor_id != INTEL_VENDOR_ID
        || plan.artifact.target.device_id != LUNAR_LAKE_DEVICE_ID
        || plan.artifact.target.revision != LUNAR_LAKE_REVISION
        || (plan.artifact.target.firmware_api[JOB_CMD_API_INDEX] != 0
            && plan.artifact.target.firmware_api[JOB_CMD_API_INDEX] != JOB_CMD_API_WORD)
        || (plan.artifact.target.firmware_api[NNRT_API_INDEX] != 0
            && plan.artifact.target.firmware_api[NNRT_API_INDEX] != NNRT_API_WORD)
    {
        return Err(ValidationError::WrongDevice);
    }
    if plan.artifact.elf.relocation_count != 0 {
        return Err(ValidationError::UnresolvedRelocations);
    }
    let resident = plan
        .ranges
        .iter()
        .filter(|range| matches!(range.role, BufferRole::Program | BufferRole::Scratch))
        .try_fold(0u64, |sum, range| sum.checked_add(range.bytes))
        .ok_or(ValidationError::ArithmeticOverflow)?;
    if resident < plan.artifact.bindings.resident_bytes {
        return Err(ValidationError::ResidentBytesExceeded);
    }
    if plan.artifact.bindings.inputs as usize + plan.artifact.bindings.outputs as usize
        != plan.bindings.entries().len()
    {
        return Err(ValidationError::BindingMismatch);
    }
    Ok(())
}

fn validate_copy_descriptors(
    image: CommandImage<'_>,
    ranges: &[MemoryRange],
    descriptor_heap: u64,
    descriptor_offset: u64,
    count: u32,
) -> Result<(), ValidationError> {
    if count == 0 || count > MAX_COPY_DESCRIPTORS {
        return Err(ValidationError::InvalidCommandLength);
    }
    let start = descriptor_heap
        .checked_add(descriptor_offset)
        .ok_or(ValidationError::ArithmeticOverflow)?;
    let bytes = u64::from(count)
        .checked_mul(COPY_DESCRIPTOR_40XX_BYTES)
        .ok_or(ValidationError::ArithmeticOverflow)?;
    if !in_any_range(ranges, start, bytes) {
        return Err(ValidationError::InvalidRange);
    }
    let image_offset = start
        .checked_sub(image.vpu_address)
        .and_then(|offset| usize::try_from(offset).ok())
        .ok_or(ValidationError::InvalidRange)?;
    for index in 0..count as usize {
        let base = image_offset
            .checked_add(index * COPY_DESCRIPTOR_40XX_BYTES as usize)
            .ok_or(ValidationError::ArithmeticOverflow)?;
        let size = read_u32(image.bytes, base + 24)? as u64;
        let source = read_u64(image.bytes, base + 40)?;
        let destination = read_u64(image.bytes, base + 48)?;
        if size == 0
            || size > MAX_MEMORY_FILL_BYTES
            || !in_any_range(ranges, source, size)
            || !in_any_range(ranges, destination, size)
        {
            return Err(ValidationError::InvalidRange);
        }
    }
    Ok(())
}

pub fn validate_command_image(
    image: CommandImage<'_>,
    ranges: &[MemoryRange],
) -> Result<usize, ValidationError> {
    let header = image.commands_offset as usize;
    if image.commands_offset % 64 != 0 || header + 48 > image.bytes.len() {
        return Err(ValidationError::InvalidCommandOffset);
    }
    let total = read_u32(image.bytes, header)? as usize;
    let command_offset = read_u32(image.bytes, header + 4)? as usize;
    let api = read_u32(image.bytes, header + 8)?;
    let reserved = read_u32(image.bytes, header + 12)?;
    let descriptor_heap = read_u64(image.bytes, header + 16)?;
    let fence_heap = read_u64(image.bytes, header + 32)?;
    let context_save = read_u64(image.bytes, header + 40)?;
    if api != JOB_CMD_API_WORD || reserved != 0 || total < 48 || command_offset < 48 {
        return Err(ValidationError::InvalidCommandHeader);
    }
    let end = header
        .checked_add(total)
        .ok_or(ValidationError::ArithmeticOverflow)?;
    let mut cursor = header
        .checked_add(command_offset)
        .ok_or(ValidationError::ArithmeticOverflow)?;
    if end > image.bytes.len() || cursor > end || !in_any_range(ranges, context_save, 64) {
        return Err(ValidationError::InvalidRange);
    }
    if descriptor_heap != 0 && !in_any_range(ranges, descriptor_heap, 1) {
        return Err(ValidationError::InvalidRange);
    }
    if fence_heap != 0 && !in_any_range(ranges, fence_heap, 8) {
        return Err(ValidationError::InvalidRange);
    }

    let mut count = 0usize;
    while cursor < end {
        count += 1;
        if count > MAX_COMMANDS {
            return Err(ValidationError::TooManyCommands);
        }
        let command_type = read_u16(image.bytes, cursor)?;
        let command_bytes = read_u16(image.bytes, cursor + 2)? as usize;
        let command_end = cursor
            .checked_add(command_bytes)
            .ok_or(ValidationError::ArithmeticOverflow)?;
        if command_bytes < 4 || command_end > end {
            return Err(ValidationError::InvalidCommandLength);
        }
        let exact = match command_type {
            VPU_CMD_NOP => None,
            VPU_CMD_TIMESTAMP => Some(16),
            VPU_CMD_FENCE_WAIT | VPU_CMD_FENCE_SIGNAL => Some(24),
            VPU_CMD_BARRIER => Some(8),
            VPU_CMD_METRIC_QUERY_BEGIN | VPU_CMD_METRIC_QUERY_END => Some(16),
            VPU_CMD_MEMORY_FILL => Some(32),
            VPU_CMD_COPY => Some(24),
            VPU_CMD_INFERENCE_EXECUTE => Some(32),
            _ => return Err(ValidationError::InvalidCommandType),
        };
        if exact.is_some_and(|expected| expected != command_bytes) {
            return Err(ValidationError::InvalidCommandLength);
        }
        match command_type {
            VPU_CMD_TIMESTAMP => {
                if !in_any_range(ranges, read_u64(image.bytes, cursor + 8)?, 8) {
                    return Err(ValidationError::InvalidRange);
                }
            }
            VPU_CMD_FENCE_WAIT | VPU_CMD_FENCE_SIGNAL => {
                let offset = read_u64(image.bytes, cursor + 8)?;
                let address = fence_heap
                    .checked_add(offset)
                    .ok_or(ValidationError::ArithmeticOverflow)?;
                if !in_any_range(ranges, address, 8) {
                    return Err(ValidationError::InvalidRange);
                }
            }
            VPU_CMD_MEMORY_FILL => {
                let address = read_u64(image.bytes, cursor + 8)?;
                let bytes = read_u64(image.bytes, cursor + 16)?;
                if bytes == 0
                    || bytes > MAX_MEMORY_FILL_BYTES
                    || !in_any_range(ranges, address, bytes)
                {
                    return Err(ValidationError::InvalidRange);
                }
            }
            VPU_CMD_COPY => validate_copy_descriptors(
                image,
                ranges,
                descriptor_heap,
                read_u64(image.bytes, cursor + 8)?,
                read_u32(image.bytes, cursor + 16)?,
            )?,
            VPU_CMD_INFERENCE_EXECUTE => {
                let address = read_u64(image.bytes, cursor + 16)?;
                let bytes = read_u32(image.bytes, cursor + 24)? as u64;
                if read_u32(image.bytes, cursor + 28)? != 0
                    || bytes == 0
                    || !in_any_range(ranges, address, bytes)
                {
                    return Err(ValidationError::InvalidRange);
                }
            }
            _ => {}
        }
        cursor = command_end;
    }
    if cursor != end {
        return Err(ValidationError::InvalidCommandLength);
    }
    Ok(count)
}

pub fn validate_submission<'a>(
    plan: &'a SubmissionPlan<'a>,
) -> Result<ValidatedSubmission<'a>, ValidationError> {
    validate_device(plan.device)?;
    validate_queue(plan.queue_id, plan.queue)?;
    validate_handles(plan)?;
    validate_bindings(plan)?;
    validate_artifact(plan)?;
    let command_count = validate_command_image(plan.command, plan.ranges)?;
    Ok(ValidatedSubmission {
        plan,
        command_count,
    })
}

#[cfg(test)]
mod tests {
    use crate::runtime::artifact::TargetFingerprint;
    use crate::runtime::{Binding, BindingAccess};

    use super::super::elf::{BindingMetadata, ElfInspection};
    use super::*;

    fn parameters() -> DeviceParameters {
        let mut firmware_api = [0u64; 8];
        firmware_api[JOB_CMD_API_INDEX] = JOB_CMD_API_WORD as u64;
        firmware_api[NNRT_API_INDEX] = NNRT_API_WORD as u64;
        DeviceParameters {
            device_id: LUNAR_LAKE_DEVICE_ID,
            revision: LUNAR_LAKE_REVISION,
            platform_type: 0,
            core_clock_rate: 0,
            context_id: 1,
            context_base_address: 0x1000,
            firmware_api,
            tile_config: 0,
            sku: 0,
            capability_bits: 0,
            max_contexts: 1,
            preempt_buffer_bytes: 0,
        }
    }

    fn artifact() -> ArtifactInspection {
        let mut firmware_api = [0u32; 8];
        firmware_api[JOB_CMD_API_INDEX] = JOB_CMD_API_WORD;
        firmware_api[NNRT_API_INDEX] = NNRT_API_WORD;
        ArtifactInspection {
            elf: ElfInspection {
                file_type: 2,
                machine: 0,
                flags: 0,
                entry: 0,
                program_headers: 0,
                section_headers: 0,
                relocation_count: 0,
                symbol_count: 0,
                loadable_file_bytes: 0,
                loadable_memory_bytes: 0,
                resident_bytes: 0,
            },
            bindings: BindingMetadata {
                abi_major: 1,
                abi_minor: 0,
                inputs: 1,
                outputs: 0,
                scratch: 0,
                resident_bytes: 4096,
            },
            target: TargetFingerprint {
                vendor_id: INTEL_VENDOR_ID,
                device_id: LUNAR_LAKE_DEVICE_ID,
                revision: LUNAR_LAKE_REVISION,
                architecture: 40,
                firmware_api,
            },
        }
    }

    fn command_image(bytes: &mut [u8; 256]) -> CommandImage<'_> {
        let header = 128usize;
        bytes[header..header + 4].copy_from_slice(&56u32.to_le_bytes());
        bytes[header + 4..header + 8].copy_from_slice(&48u32.to_le_bytes());
        bytes[header + 8..header + 12].copy_from_slice(&JOB_CMD_API_WORD.to_le_bytes());
        bytes[header + 40..header + 48].copy_from_slice(&0x1000u64.to_le_bytes());
        bytes[header + 48..header + 50].copy_from_slice(&VPU_CMD_BARRIER.to_le_bytes());
        bytes[header + 50..header + 52].copy_from_slice(&8u16.to_le_bytes());
        CommandImage {
            bytes,
            vpu_address: 0x1000,
            commands_offset: 128,
        }
    }

    #[test]
    fn dry_run_accepts_public_barrier_packet_without_submit() {
        let parameters = parameters();
        let artifact = artifact();
        let ranges = [
            MemoryRange {
                handle: 1,
                role: BufferRole::Command,
                vpu_address: 0x1000,
                bytes: 4096,
            },
            MemoryRange {
                handle: 2,
                role: BufferRole::Program,
                vpu_address: 0x2000,
                bytes: 4096,
            },
        ];
        let memory_id = MemoryId::new(1).unwrap();
        let bound = [BoundMemory {
            id: memory_id,
            range: ranges[1],
        }];
        let binding_entries = [Binding {
            slot: 0,
            memory: memory_id,
            offset: 0,
            bytes: 16,
            access: BindingAccess::Read,
        }];
        let bindings = SubmissionBindings::new(&binding_entries).unwrap();
        let mut command = [0u8; 256];
        let plan = SubmissionPlan {
            device: &parameters,
            artifact: &artifact,
            queue_id: 1,
            queue: QueueSpec {
                priority: 0,
                turbo: false,
            },
            command_handle: 1,
            handles: &[1, 2],
            preemption_buffer_index: None,
            command: command_image(&mut command),
            ranges: &ranges,
            bindings: &bindings,
            bound_memory: &bound,
        };
        let validated = validate_submission(&plan).unwrap();
        assert_eq!(validated.command_count(), 1);
    }

    #[test]
    fn dry_run_rejects_faults_before_submit() {
        let mut wrong = parameters();
        wrong.firmware_api[JOB_CMD_API_INDEX] = 0;
        assert_eq!(
            validate_device(&wrong),
            Err(ValidationError::FirmwareMismatch)
        );
        let mut artifact = artifact();
        artifact.elf.relocation_count = 1;
        artifact.bindings.inputs = 0;
        let ranges = [
            MemoryRange {
                handle: 1,
                role: BufferRole::Command,
                vpu_address: 0x1000,
                bytes: 4096,
            },
            MemoryRange {
                handle: 2,
                role: BufferRole::Program,
                vpu_address: 0x2000,
                bytes: 4096,
            },
        ];
        let no_bindings = SubmissionBindings::new(&[]).unwrap();
        let mut command = [0u8; 256];
        let plan = SubmissionPlan {
            device: &parameters(),
            artifact: &artifact,
            queue_id: 1,
            queue: QueueSpec {
                priority: 0,
                turbo: false,
            },
            command_handle: 1,
            handles: &[1, 2],
            preemption_buffer_index: None,
            command: command_image(&mut command),
            ranges: &ranges,
            bindings: &no_bindings,
            bound_memory: &[],
        };
        assert!(matches!(
            validate_submission(&plan),
            Err(ValidationError::UnresolvedRelocations)
        ));
    }
}
