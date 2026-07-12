//! Binding, HostParsedInference, and command-buffer dry-run preparation.
//!
//! No submit ioctl exists in this module.

#![allow(dead_code)]

use alloc::vec::Vec;
use core::ptr;

use crate::identity::{SemanticDigest, SemanticHasher};
use crate::os::MemoryMapping;
use crate::runtime::RuntimeError;

use super::elf::ParsedIntelElf;
use super::firmware::{JOB_CMD_API_WORD, VPU_CMD_INFERENCE_EXECUTE, VPU_CMD_NOP};
use super::hpi::{HpiRequirements, HOST_PARSED_INFERENCE_ALIGNMENT, HOST_PARSED_INFERENCE_BYTES};
use super::link::{
    visit_binding_requirements, BindingRequirement, DeviceMemoryClass, PlacedSection,
};
use super::load::{stage_program, LoadError, StagedProgram, MAX_TINY_SUBMISSION_BO_BYTES};
use super::transport::release_host_writes;
use super::uapi::{
    DRM_IVPU_BO_DMA_MEM, DRM_IVPU_BO_MAPPABLE, DRM_IVPU_BO_WC, DRM_IVPU_PARAM_UNIQUE_INFERENCE_ID,
};
use super::validate::{
    validate_command_image, validate_device, CommandImage, MemoryRange, ValidationError,
};
use super::{BufferObject, DeviceParameters, IvpuDevice};

const VPU_SHT_PERF_METRICS: u32 = 0x8aaa_aaae;
const PERFORMANCE_METRICS_BYTES: usize = 320;
const COMMAND_HEADER_OFFSET: usize = 128;
const COMMAND_LIST_OFFSET: usize = 224;
const INFERENCE_COMMAND_BYTES: usize = 32;
const NOP_COMMAND_BYTES: usize = 24;
const COMMAND_LOGICAL_BYTES: usize =
    COMMAND_LIST_OFFSET + INFERENCE_COMMAND_BYTES + NOP_COMMAND_BYTES;
const MAX_PREPARED_HANDLES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrepareError {
    Runtime(RuntimeError),
    Load(LoadError),
    Validation(ValidationError),
    InvalidMetadata,
    BindingMismatch,
    BudgetExceeded,
    Allocation,
    ExplicitQueuePreemptionExceedsBudget {
        user_bytes: u64,
        preemption_bytes: u64,
    },
}

pub struct BindingBuffer<'device> {
    pub section: usize,
    pub class: DeviceMemoryClass,
    pub logical_bytes: u64,
    pub bo: BufferObject<'device>,
}

pub struct PreparedInference<'device> {
    pub staged: StagedProgram<'device>,
    pub bindings: Vec<BindingBuffer<'device>>,
    pub hpi: BufferObject<'device>,
    pub command: BufferObject<'device>,
    pub handles: Vec<u32>,
    pub total_user_bo_bytes: u64,
    pub command_count: usize,
    pub inference_id: u64,
    pub command_digest: SemanticDigest,
}

impl PreparedInference<'_> {
    pub fn destroy(self) {
        drop(self);
    }
}

fn collect_bindings<'a>(
    elf: &'a ParsedIntelElf<'a>,
) -> Result<Vec<BindingRequirement>, PrepareError> {
    let mut count = 0usize;
    visit_binding_requirements(elf, |_| count += 1);
    let mut bindings = Vec::new();
    bindings
        .try_reserve_exact(count)
        .map_err(|_| PrepareError::Allocation)?;
    visit_binding_requirements(elf, |binding| bindings.push(binding));
    Ok(bindings)
}

fn map_and_initialize(
    bo: &BufferObject<'_>,
    logical_bytes: usize,
    initial: Option<&[u8]>,
) -> Result<(), PrepareError> {
    let mapping = bo.map(true).map_err(PrepareError::Runtime)?;
    if logical_bytes > mapping.len() || initial.is_some_and(|data| data.len() > logical_bytes) {
        return Err(PrepareError::InvalidMetadata);
    }
    // SAFETY: unique live mapping, with both zeroing and copy bounded to the logical object.
    unsafe {
        ptr::write_bytes(mapping.address().as_ptr(), 0, logical_bytes);
        if let Some(data) = initial {
            ptr::copy_nonoverlapping(data.as_ptr(), mapping.address().as_ptr(), data.len());
        }
    }
    release_host_writes();
    Ok(())
}

fn performance_metrics<'a>(elf: &'a ParsedIntelElf<'a>) -> Result<&'a [u8], PrepareError> {
    let mut metrics = None;
    let mut duplicate = false;
    elf.visit_sections(|section| {
        if section.section_type == VPU_SHT_PERF_METRICS {
            if metrics.is_some() {
                duplicate = true;
            } else {
                metrics = section.data;
            }
        }
    });
    let metrics = metrics.ok_or(PrepareError::InvalidMetadata)?;
    if duplicate || metrics.len() < PERFORMANCE_METRICS_BYTES {
        return Err(PrepareError::InvalidMetadata);
    }
    Ok(&metrics[..PERFORMANCE_METRICS_BYTES])
}

fn create_hpi<'device>(
    device: &'device IvpuDevice,
    elf: &ParsedIntelElf<'_>,
    hpi: &HpiRequirements,
    entry_address: u64,
) -> Result<(BufferObject<'device>, u64), PrepareError> {
    let bo = device
        .create_bo(
            HOST_PARSED_INFERENCE_BYTES,
            DRM_IVPU_BO_WC | DRM_IVPU_BO_MAPPABLE,
        )
        .map_err(PrepareError::Runtime)?;
    let info = bo.info().map_err(PrepareError::Runtime)?;
    if info.vpu_address % HOST_PARSED_INFERENCE_ALIGNMENT != 0 {
        return Err(PrepareError::InvalidMetadata);
    }
    let metrics = performance_metrics(elf)?;
    let mapping = bo.map(true).map_err(PrepareError::Runtime)?;
    if mapping.len() < HOST_PARSED_INFERENCE_BYTES as usize {
        return Err(PrepareError::InvalidMetadata);
    }
    let target = mapping.address().as_ptr();
    // SAFETY: exact public VpuHostParsedInference v1.32.0 offsets, each bounded by 384 bytes.
    unsafe {
        ptr::write_bytes(target, 0, HOST_PARSED_INFERENCE_BYTES as usize);
        ptr::copy_nonoverlapping(
            hpi.resources.cmx_slice_bytes.to_le_bytes().as_ptr(),
            target.add(8),
            4,
        );
        *target.add(18) = hpi.resources.slice_count;
        *target.add(19) = hpi.resources.barriers;
        *target.add(20) = u8::try_from(hpi.versions.mapped_inference_format)
            .map_err(|_| PrepareError::InvalidMetadata)?;
        ptr::copy_nonoverlapping(metrics.as_ptr(), target.add(24), metrics.len());
        ptr::copy_nonoverlapping(entry_address.to_le_bytes().as_ptr(), target.add(368), 8);
        ptr::copy_nonoverlapping(1u64.to_le_bytes().as_ptr(), target.add(376), 8);
    }
    release_host_writes();
    Ok((bo, info.bytes))
}

fn create_bindings<'device>(
    device: &'device IvpuDevice,
    requirements: &[BindingRequirement],
    input_data: &[&[u8]],
) -> Result<(Vec<BindingBuffer<'device>>, Vec<PlacedSection>, u64), PrepareError> {
    let expected_inputs = requirements
        .iter()
        .filter(|binding| binding.class == DeviceMemoryClass::Input)
        .count();
    if input_data.len() != expected_inputs {
        return Err(PrepareError::BindingMismatch);
    }
    let mut buffers = Vec::new();
    let mut placements = Vec::new();
    buffers
        .try_reserve_exact(requirements.len())
        .map_err(|_| PrepareError::Allocation)?;
    placements
        .try_reserve_exact(requirements.len())
        .map_err(|_| PrepareError::Allocation)?;
    let mut input_index = 0usize;
    let mut total = 0u64;
    for requirement in requirements {
        let initial = if requirement.class == DeviceMemoryClass::Input {
            let data = input_data[input_index];
            input_index += 1;
            if data.len() != requirement.bytes as usize {
                return Err(PrepareError::BindingMismatch);
            }
            Some(data)
        } else {
            None
        };
        let bo = device
            .create_bo(
                requirement.bytes.max(1),
                DRM_IVPU_BO_MAPPABLE | DRM_IVPU_BO_DMA_MEM,
            )
            .map_err(PrepareError::Runtime)?;
        let info = bo.info().map_err(PrepareError::Runtime)?;
        if info.vpu_address % requirement.alignment != 0 {
            return Err(PrepareError::InvalidMetadata);
        }
        map_and_initialize(&bo, requirement.bytes as usize, initial)?;
        total = total
            .checked_add(info.bytes)
            .ok_or(PrepareError::BudgetExceeded)?;
        placements.push(PlacedSection {
            section: requirement.section,
            device_address: info.vpu_address,
            bytes: requirement.bytes,
        });
        buffers.push(BindingBuffer {
            section: requirement.section,
            class: requirement.class,
            logical_bytes: requirement.bytes,
            bo,
        });
    }
    Ok((buffers, placements, total))
}

fn build_command_bytes(
    command: &mut [u8; COMMAND_LOGICAL_BYTES],
    command_vpu_address: u64,
    hpi_address: u64,
    inference_id: u64,
) {
    *command = [0; COMMAND_LOGICAL_BYTES];
    let command_buffer_bytes = (COMMAND_LOGICAL_BYTES - COMMAND_HEADER_OFFSET) as u32;
    let command_offset = (COMMAND_LIST_OFFSET - COMMAND_HEADER_OFFSET) as u32;
    command[COMMAND_HEADER_OFFSET..COMMAND_HEADER_OFFSET + 4]
        .copy_from_slice(&command_buffer_bytes.to_le_bytes());
    command[COMMAND_HEADER_OFFSET + 4..COMMAND_HEADER_OFFSET + 8]
        .copy_from_slice(&command_offset.to_le_bytes());
    command[COMMAND_HEADER_OFFSET + 8..COMMAND_HEADER_OFFSET + 12]
        .copy_from_slice(&JOB_CMD_API_WORD.to_le_bytes());
    command[COMMAND_HEADER_OFFSET + 40..COMMAND_HEADER_OFFSET + 48]
        .copy_from_slice(&command_vpu_address.to_le_bytes());

    let inference = COMMAND_LIST_OFFSET;
    command[inference..inference + 2].copy_from_slice(&VPU_CMD_INFERENCE_EXECUTE.to_le_bytes());
    command[inference + 2..inference + 4]
        .copy_from_slice(&(INFERENCE_COMMAND_BYTES as u16).to_le_bytes());
    command[inference + 8..inference + 16].copy_from_slice(&inference_id.to_le_bytes());
    command[inference + 16..inference + 24].copy_from_slice(&hpi_address.to_le_bytes());
    command[inference + 24..inference + 28]
        .copy_from_slice(&(HOST_PARSED_INFERENCE_BYTES as u32).to_le_bytes());

    let nop = inference + INFERENCE_COMMAND_BYTES;
    command[nop..nop + 2].copy_from_slice(&VPU_CMD_NOP.to_le_bytes());
    command[nop + 2..nop + 4].copy_from_slice(&(NOP_COMMAND_BYTES as u16).to_le_bytes());
}

fn create_command<'device>(
    device: &'device IvpuDevice,
    hpi: &BufferObject<'_>,
    inference_id: u64,
) -> Result<(BufferObject<'device>, u64, SemanticDigest), PrepareError> {
    let command = device
        .create_bo(
            COMMAND_LOGICAL_BYTES as u64,
            DRM_IVPU_BO_WC | DRM_IVPU_BO_MAPPABLE,
        )
        .map_err(PrepareError::Runtime)?;
    let command_info = command.info().map_err(PrepareError::Runtime)?;
    let hpi_info = hpi.info().map_err(PrepareError::Runtime)?;
    let mut bytes = [0u8; COMMAND_LOGICAL_BYTES];
    build_command_bytes(
        &mut bytes,
        command_info.vpu_address,
        hpi_info.vpu_address,
        inference_id,
    );
    map_and_initialize(&command, COMMAND_LOGICAL_BYTES, Some(&bytes))?;
    let mut hasher = SemanticHasher::new(b"direct-ivpu/inference-command");
    hasher.field(b"command-bytes", &bytes);
    Ok((command, command_info.bytes, hasher.finish()))
}

pub fn prepare_inference<'device>(
    device: &'device IvpuDevice,
    parameters: &DeviceParameters,
    elf: &ParsedIntelElf<'_>,
    hpi_requirements: &HpiRequirements,
    input_data: &[&[u8]],
) -> Result<PreparedInference<'device>, PrepareError> {
    validate_device(parameters).map_err(PrepareError::Validation)?;
    let mut staged = stage_program(device, elf, hpi_requirements).map_err(PrepareError::Load)?;
    let requirements = collect_bindings(elf)?;
    let (bindings, binding_placements, binding_bo_bytes) =
        create_bindings(device, &requirements, input_data)?;
    staged
        .apply_jit_bindings(elf, &binding_placements)
        .map_err(PrepareError::Load)?;
    let (hpi, hpi_bo_bytes) = create_hpi(device, elf, hpi_requirements, staged.entry_address)?;
    let inference_id = device
        .get_param(DRM_IVPU_PARAM_UNIQUE_INFERENCE_ID, 0)
        .map_err(PrepareError::Runtime)?;
    let (command, command_bo_bytes, command_digest) = create_command(device, &hpi, inference_id)?;
    let total_user_bo_bytes = staged
        .total_bo_bytes
        .checked_add(binding_bo_bytes)
        .and_then(|total| total.checked_add(hpi_bo_bytes))
        .and_then(|total| total.checked_add(command_bo_bytes))
        .ok_or(PrepareError::BudgetExceeded)?;
    if total_user_bo_bytes > MAX_TINY_SUBMISSION_BO_BYTES {
        return Err(PrepareError::BudgetExceeded);
    }

    let handle_count = 1 + staged.sections().len() + 1 + bindings.len();
    if handle_count > MAX_PREPARED_HANDLES {
        return Err(PrepareError::BudgetExceeded);
    }
    let mut handles = Vec::new();
    handles
        .try_reserve_exact(handle_count)
        .map_err(|_| PrepareError::Allocation)?;
    handles.push(command.handle());
    handles.extend(staged.sections().iter().map(|section| section.bo.handle()));
    handles.push(hpi.handle());
    handles.extend(bindings.iter().map(|binding| binding.bo.handle()));
    for (index, &handle) in handles.iter().enumerate() {
        if handle == 0 || handles[..index].contains(&handle) {
            return Err(PrepareError::InvalidMetadata);
        }
    }

    let command_info = command.info().map_err(PrepareError::Runtime)?;
    let hpi_info = hpi.info().map_err(PrepareError::Runtime)?;
    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(handle_count)
        .map_err(|_| PrepareError::Allocation)?;
    ranges.push(MemoryRange {
        handle: command.handle(),
        role: super::transport::BufferRole::Command,
        vpu_address: command_info.vpu_address,
        bytes: command_info.bytes,
    });
    ranges.extend(
        staged
            .sections()
            .iter()
            .zip(staged.placements())
            .map(|(section, placement)| MemoryRange {
                handle: section.bo.handle(),
                role: super::transport::BufferRole::Program,
                vpu_address: placement.device_address,
                bytes: placement.bytes.max(1),
            }),
    );
    ranges.push(MemoryRange {
        handle: hpi.handle(),
        role: super::transport::BufferRole::Program,
        vpu_address: hpi_info.vpu_address,
        bytes: HOST_PARSED_INFERENCE_BYTES,
    });
    ranges.extend(
        bindings
            .iter()
            .zip(&binding_placements)
            .map(|(binding, placement)| MemoryRange {
                handle: binding.bo.handle(),
                role: super::transport::BufferRole::Binding,
                vpu_address: placement.device_address,
                bytes: placement.bytes.max(1),
            }),
    );

    let command_mapping = command.map(false).map_err(PrepareError::Runtime)?;
    if command_mapping.len() < COMMAND_LOGICAL_BYTES {
        return Err(PrepareError::InvalidMetadata);
    }
    // SAFETY: bounded read-only view for command validation while mapping is alive.
    let command_bytes = unsafe {
        core::slice::from_raw_parts(command_mapping.address().as_ptr(), COMMAND_LOGICAL_BYTES)
    };
    let command_count = validate_command_image(
        CommandImage {
            bytes: command_bytes,
            vpu_address: command_info.vpu_address,
            commands_offset: COMMAND_HEADER_OFFSET as u32,
        },
        &ranges,
    )
    .map_err(PrepareError::Validation)?;
    if command_count != 2 {
        return Err(PrepareError::InvalidMetadata);
    }

    // Managed CMDQ submit with preemption index 0 causes the kernel to allocate the reported
    // preemption buffer. On this machine that is 28,389,376 bytes, violating the explicit 8 MiB
    // proof budget. Keep the fully prepared resources unreachable from submit.
    if parameters.preempt_buffer_bytes > 0
        && total_user_bo_bytes
            .checked_add(parameters.preempt_buffer_bytes)
            .is_none_or(|total| total > MAX_TINY_SUBMISSION_BO_BYTES)
    {
        return Err(PrepareError::ExplicitQueuePreemptionExceedsBudget {
            user_bytes: total_user_bo_bytes,
            preemption_bytes: parameters.preempt_buffer_bytes,
        });
    }

    Ok(PreparedInference {
        staged,
        bindings,
        hpi,
        command,
        handles,
        total_user_bo_bytes,
        command_count,
        inference_id,
        command_digest,
    })
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "bench-hardware")]
    use std::io::Read;

    #[cfg(feature = "bench-hardware")]
    use super::super::elf::parse_intel_elf;
    #[cfg(feature = "bench-hardware")]
    use super::super::hpi::inspect_hpi_requirements;
    use super::*;

    #[test]
    fn mapped_inference_command_matches_public_layout() {
        let mut command = [0u8; COMMAND_LOGICAL_BYTES];
        build_command_bytes(&mut command, 0x1000, 0x2000, 7);
        assert_eq!(
            u32::from_le_bytes(command[128..132].try_into().unwrap()),
            152
        );
        assert_eq!(
            u32::from_le_bytes(command[132..136].try_into().unwrap()),
            96
        );
        assert_eq!(
            u16::from_le_bytes(command[224..226].try_into().unwrap()),
            VPU_CMD_INFERENCE_EXECUTE
        );
        assert_eq!(
            u16::from_le_bytes(command[258..260].try_into().unwrap()),
            NOP_COMMAND_BYTES as u16
        );
        assert_eq!(
            u64::from_le_bytes(command[240..248].try_into().unwrap()),
            0x2000
        );
    }

    #[test]
    #[cfg(feature = "bench-hardware")]
    #[ignore = "full dry-run preparation only; expected to stop at preemption budget, never submit"]
    fn prepare_tiny_artifact_stops_before_submit() {
        const PATH: &str = "/home/cassmantle/.cache/ggml-npu/ir2-8c91da4914adc7d5-n1.blob";
        let length = usize::try_from(std::fs::metadata(PATH).unwrap().len()).unwrap();
        assert!(length <= 1024 * 1024);
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).unwrap();
        bytes.resize(length, 0);
        std::fs::File::open(PATH)
            .unwrap()
            .read_exact(&mut bytes)
            .unwrap();
        let elf = parse_intel_elf(&bytes).unwrap();
        let device = IvpuDevice::open_first().unwrap();
        let parameters = device.parameters().unwrap();
        let hpi = inspect_hpi_requirements(&elf, &parameters).unwrap();
        let input0 = [0x11u8; 128];
        let input1 = [0x22u8; 128];
        let result = prepare_inference(&device, &parameters, &elf, &hpi, &[&input0, &input1]);
        std::println!(
            "device={} artifact={PATH} preempt_buffer_bytes={} error={:?}",
            device.device_path().to_string_lossy(),
            parameters.preempt_buffer_bytes,
            result.as_ref().err(),
        );
        assert!(matches!(
            result,
            Err(PrepareError::ExplicitQueuePreemptionExceedsBudget {
                user_bytes,
                preemption_bytes,
            }) if user_bytes < 1024 * 1024
                && preemption_bytes == parameters.preempt_buffer_bytes
        ));
    }
}
