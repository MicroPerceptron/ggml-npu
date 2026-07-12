//! Binding, HostParsedInference, and validated command-buffer preparation.

#![allow(dead_code)]

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::ptr;

use crate::host::MemoryMapping;
use crate::identity::{SemanticDigest, SemanticHasher};
use crate::interop::{SharedAllocationOwner, SharedAllocationView};
use crate::runtime::RuntimeError;

use super::elf::ParsedIntelElf;
use super::firmware::{JOB_CMD_API_WORD, VPU_CMD_INFERENCE_EXECUTE, VPU_CMD_NOP};
use super::hpi::{HpiRequirements, HOST_PARSED_INFERENCE_ALIGNMENT, HOST_PARSED_INFERENCE_BYTES};
use super::link::{
    visit_binding_requirements, BindingRequirement, DeviceMemoryClass, PlacedSection,
};
use super::linux::{DirectSharedBuffer, LinuxIvpuSharedAllocation};
use super::load::{stage_program, LoadError, StagedProgram};
use super::transport::{acquire_device_writes, release_host_writes};
use super::uapi::{
    DRM_IVPU_BO_DMA_MEM, DRM_IVPU_BO_MAPPABLE, DRM_IVPU_BO_WC, DRM_IVPU_PARAM_UNIQUE_INFERENCE_ID,
};
use super::validate::{
    validate_command_image, validate_device, CommandImage, MemoryRange, ValidationError,
};
use super::{BufferBacking, BufferObject, DeviceParameters, IvpuDevice};

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
pub struct PreparationBudget {
    pub max_user_bo_bytes: u64,
    pub max_total_accounted_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreparationTimings {
    pub stage_program_ns: u64,
    pub binding_import_ns: u64,
    pub binding_allocate_ns: u64,
    pub binding_initialize_ns: u64,
    pub link_ns: u64,
    pub hpi_ns: u64,
    pub command_ns: u64,
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingBacking {
    /// Host-owned transferable allocation imported into the IVPU context.
    ImportedShared,
    /// Correct fallback allocated by IVPU. It remains exportable when PRIME export is available.
    IvpuGem,
}

pub struct BindingBuffer<'device> {
    pub section: usize,
    pub class: DeviceMemoryClass,
    pub logical_bytes: u64,
    pub bo: Rc<BufferObject<'device>>,
    pub backing: BindingBacking,
    pub allocation_offset: u64,
    pub direct: bool,
    pub host_address: Option<core::ptr::NonNull<u8>>,
}

/// Host-side state that lets a prepared program re-JIT IO bindings without re-staging the ELF.
/// Only retained for small artifacts so warm reuse cannot double-keep baked-weight blobs (#5).
pub struct WarmRebindState {
    pub pre_jit_sections: Vec<(usize, Vec<u8>)>,
    pub jit_relocations: usize,
}

pub struct PreparedInference<'device> {
    pub staged: StagedProgram<'device>,
    pub bindings: Vec<BindingBuffer<'device>>,
    pub hpi: BufferObject<'device>,
    pub command: BufferObject<'device>,
    pub handles: Vec<u32>,
    pub executable_bo_bytes: u64,
    pub binding_bo_bytes: u64,
    pub hpi_bo_bytes: u64,
    pub command_bo_bytes: u64,
    pub preemption_bytes: u64,
    pub total_user_bo_bytes: u64,
    pub total_accounted_bytes: u64,
    pub command_count: usize,
    pub inference_id: u64,
    pub command_digest: SemanticDigest,
    pub timings: PreparationTimings,
    pub warm_rebind: Option<WarmRebindState>,
}

impl PreparedInference<'_> {
    pub const fn commands_offset(&self) -> u32 {
        COMMAND_HEADER_OFFSET as u32
    }

    pub fn copy_output(&self, output_index: usize, target: &mut [u8]) -> Result<(), PrepareError> {
        let binding = self
            .bindings
            .iter()
            .filter(|binding| binding.class == DeviceMemoryClass::Output)
            .nth(output_index)
            .ok_or(PrepareError::BindingMismatch)?;
        if target.len() != binding.logical_bytes as usize {
            return Err(PrepareError::BindingMismatch);
        }
        if let Some(address) = binding.host_address {
            acquire_device_writes();
            // SAFETY: owner-complete tensor allocation remains live with the prepared program.
            unsafe {
                ptr::copy_nonoverlapping(address.as_ptr(), target.as_mut_ptr(), target.len());
            }
        } else {
            let mapping = binding.bo.map(false).map_err(PrepareError::Runtime)?;
            let offset = usize::try_from(binding.allocation_offset)
                .map_err(|_| PrepareError::InvalidMetadata)?;
            if offset
                .checked_add(target.len())
                .is_none_or(|end| end > mapping.len())
            {
                return Err(PrepareError::InvalidMetadata);
            }
            acquire_device_writes();
            // SAFETY: both regions are live, non-overlapping, and exactly bounded by target.
            unsafe {
                ptr::copy_nonoverlapping(
                    mapping.address().as_ptr().add(offset),
                    target.as_mut_ptr(),
                    target.len(),
                );
            }
        }
        Ok(())
    }

    pub fn write_inputs(&self, inputs: &[&[u8]]) -> Result<(), PrepareError> {
        let expected = self
            .bindings
            .iter()
            .filter(|binding| binding.class == DeviceMemoryClass::Input)
            .count();
        if inputs.len() != expected {
            return Err(PrepareError::BindingMismatch);
        }
        for (binding, input) in self
            .bindings
            .iter()
            .filter(|binding| binding.class == DeviceMemoryClass::Input)
            .zip(inputs)
        {
            if binding.direct {
                continue;
            }
            if input.len() != binding.logical_bytes as usize {
                return Err(PrepareError::BindingMismatch);
            }
            map_and_initialize(
                &binding.bo,
                binding.allocation_offset as usize,
                binding.logical_bytes as usize,
                Some(input),
            )?;
        }
        Ok(())
    }

    pub fn input_bytes(&self, input_index: usize) -> Option<usize> {
        self.bindings
            .iter()
            .filter(|binding| binding.class == DeviceMemoryClass::Input)
            .nth(input_index)
            .and_then(|binding| usize::try_from(binding.logical_bytes).ok())
    }

    pub fn input_is_direct(&self, input_index: usize) -> bool {
        self.bindings
            .iter()
            .filter(|binding| binding.class == DeviceMemoryClass::Input)
            .nth(input_index)
            .is_some_and(|binding| binding.direct)
    }

    pub fn output_bytes(&self, output_index: usize) -> Option<usize> {
        self.bindings
            .iter()
            .filter(|binding| binding.class == DeviceMemoryClass::Output)
            .nth(output_index)
            .and_then(|binding| usize::try_from(binding.logical_bytes).ok())
    }

    pub fn output_is_direct(&self, output_index: usize) -> bool {
        self.bindings
            .iter()
            .filter(|binding| binding.class == DeviceMemoryClass::Output)
            .nth(output_index)
            .is_some_and(|binding| binding.direct)
    }

    pub fn direct_binding_count(&self) -> usize {
        self.bindings
            .iter()
            .filter(|binding| binding.direct)
            .count()
    }

    pub(super) fn output_shared_allocation(
        &self,
        output_index: usize,
    ) -> Result<(LinuxIvpuSharedAllocation, usize, BindingBacking), PrepareError> {
        let binding = self
            .bindings
            .iter()
            .filter(|binding| binding.class == DeviceMemoryClass::Output)
            .nth(output_index)
            .ok_or(PrepareError::BindingMismatch)?;
        let bytes =
            usize::try_from(binding.logical_bytes).map_err(|_| PrepareError::InvalidMetadata)?;
        binding
            .bo
            .export_shared()
            .map(|allocation| (allocation, bytes, binding.backing))
            .map_err(PrepareError::Runtime)
    }

    pub fn zero_outputs(&self) -> Result<(), PrepareError> {
        for binding in self
            .bindings
            .iter()
            .filter(|binding| binding.class == DeviceMemoryClass::Output)
        {
            if let Some(address) = binding.host_address {
                // SAFETY: direct output view is live and bounded by logical_bytes.
                unsafe {
                    ptr::write_bytes(address.as_ptr(), 0, binding.logical_bytes as usize);
                }
                release_host_writes();
            } else {
                map_and_initialize(
                    &binding.bo,
                    binding.allocation_offset as usize,
                    binding.logical_bytes as usize,
                    None,
                )?;
            }
        }
        Ok(())
    }

    pub fn outputs_are_zero(&self) -> Result<bool, PrepareError> {
        for binding in self
            .bindings
            .iter()
            .filter(|binding| binding.class == DeviceMemoryClass::Output)
        {
            if let Some(address) = binding.host_address {
                // SAFETY: owner-complete tensor allocation remains live and bounded.
                let bytes = unsafe {
                    core::slice::from_raw_parts(address.as_ptr(), binding.logical_bytes as usize)
                };
                if bytes.iter().any(|byte| *byte != 0) {
                    return Ok(false);
                }
            } else {
                let mapping = binding.bo.map(false).map_err(PrepareError::Runtime)?;
                let offset = binding.allocation_offset as usize;
                if offset
                    .checked_add(binding.logical_bytes as usize)
                    .is_none_or(|end| end > mapping.len())
                {
                    return Err(PrepareError::InvalidMetadata);
                }
                // SAFETY: bounded read-only view while the mapping is alive.
                let bytes = unsafe {
                    core::slice::from_raw_parts(
                        mapping.address().as_ptr().add(offset),
                        binding.logical_bytes as usize,
                    )
                };
                if bytes.iter().any(|byte| *byte != 0) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

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
    offset: usize,
    logical_bytes: usize,
    initial: Option<&[u8]>,
) -> Result<(), PrepareError> {
    let mapping = bo.map(true).map_err(PrepareError::Runtime)?;
    if offset
        .checked_add(logical_bytes)
        .is_none_or(|end| end > mapping.len())
        || initial.is_some_and(|data| data.len() > logical_bytes)
    {
        return Err(PrepareError::InvalidMetadata);
    }
    // SAFETY: unique live mapping, with both zeroing and copy bounded to the logical object.
    unsafe {
        let target = mapping.address().as_ptr().add(offset);
        ptr::write_bytes(target, 0, logical_bytes);
        if let Some(data) = initial {
            ptr::copy_nonoverlapping(data.as_ptr(), target, data.len());
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
    input_views: &[Option<SharedAllocationView>],
    output_views: &[Option<SharedAllocationView>],
) -> Result<
    (
        Vec<BindingBuffer<'device>>,
        Vec<PlacedSection>,
        u64,
        PreparationTimings,
    ),
    PrepareError,
> {
    let expected_inputs = requirements
        .iter()
        .filter(|binding| binding.class == DeviceMemoryClass::Input)
        .count();
    if input_data.len() != expected_inputs {
        return Err(PrepareError::BindingMismatch);
    }
    let expected_outputs = requirements
        .iter()
        .filter(|binding| binding.class == DeviceMemoryClass::Output)
        .count();
    if (!input_views.is_empty() && input_views.len() != expected_inputs)
        || (!output_views.is_empty() && output_views.len() != expected_outputs)
    {
        return Err(PrepareError::BindingMismatch);
    }
    let mut buffers = Vec::new();
    let mut placements = Vec::new();
    let mut imported: Vec<(
        crate::interop::AllocationIdentity,
        Rc<BufferObject<'device>>,
    )> = Vec::new();
    buffers
        .try_reserve_exact(requirements.len())
        .map_err(|_| PrepareError::Allocation)?;
    placements
        .try_reserve_exact(requirements.len())
        .map_err(|_| PrepareError::Allocation)?;
    let mut input_index = 0usize;
    let mut output_index = 0usize;
    let mut total = 0u64;
    let measure = crate::trace::perf_enabled();
    let mut timings = PreparationTimings::default();
    for requirement in requirements {
        let (initial, shared_view) = if requirement.class == DeviceMemoryClass::Input {
            let data = input_data[input_index];
            let view = input_views.get(input_index).copied().flatten();
            input_index += 1;
            if data.len() != requirement.bytes as usize {
                return Err(PrepareError::BindingMismatch);
            }
            (Some(data), view)
        } else {
            let view = output_views.get(output_index).copied().flatten();
            output_index += 1;
            (None, view)
        };
        let requested_bytes = requirement.allocation_bytes.max(1);
        let (bo, backing, allocation_offset, direct, is_new) = if let Some(view) = shared_view {
            let owner =
                DirectSharedBuffer::from_token(view.token).ok_or(PrepareError::BindingMismatch)?;
            if owner.identity() != view.identity
                || owner.ownership() != crate::interop::AllocationOwnership::Backend
                || view.bytes < requirement.bytes as usize
                || view
                    .offset
                    .checked_add(requested_bytes as usize)
                    .is_none_or(|end| end > view.allocation_bytes)
            {
                return Err(PrepareError::BindingMismatch);
            }
            if let Some((_, existing)) = imported
                .iter()
                .find(|(identity, _)| *identity == view.identity)
            {
                (
                    existing.clone(),
                    BindingBacking::ImportedShared,
                    view.offset as u64,
                    true,
                    false,
                )
            } else {
                let started = if measure {
                    super::super::profile::now_ns()
                } else {
                    0
                };
                let imported_bo = Rc::new(
                    device
                        .import_shared_unretained(
                            owner
                                .duplicate_allocation()
                                .map_err(PrepareError::Runtime)?,
                        )
                        .map_err(|_| PrepareError::BindingMismatch)?,
                );
                if measure {
                    timings.binding_import_ns = timings
                        .binding_import_ns
                        .saturating_add(super::super::profile::now_ns().saturating_sub(started));
                }
                imported
                    .try_reserve(1)
                    .map_err(|_| PrepareError::Allocation)?;
                imported.push((view.identity, imported_bo.clone()));
                (
                    imported_bo,
                    BindingBacking::ImportedShared,
                    view.offset as u64,
                    true,
                    true,
                )
            }
        } else {
            let started = if measure {
                super::super::profile::now_ns()
            } else {
                0
            };
            let (bo, backing) = match LinuxIvpuSharedAllocation::allocate(
                requested_bytes,
                requirement.alignment.min(rustix::param::page_size() as u64),
            )
            .and_then(|allocation| device.import_shared(allocation))
            {
                Ok(bo) => (bo, BindingBacking::ImportedShared),
                Err(_) => (
                    device
                        .create_bo(requested_bytes, DRM_IVPU_BO_MAPPABLE | DRM_IVPU_BO_DMA_MEM)
                        .map_err(PrepareError::Runtime)?,
                    BindingBacking::IvpuGem,
                ),
            };
            if measure {
                timings.binding_allocate_ns = timings
                    .binding_allocate_ns
                    .saturating_add(super::super::profile::now_ns().saturating_sub(started));
            }
            (Rc::new(bo), backing, 0, false, true)
        };
        let info = bo.info().map_err(PrepareError::Runtime)?;
        let device_address = info
            .vpu_address
            .checked_add(allocation_offset)
            .ok_or(PrepareError::InvalidMetadata)?;
        if device_address % requirement.alignment != 0 {
            return Err(PrepareError::InvalidMetadata);
        }
        if !direct {
            let started = if measure {
                super::super::profile::now_ns()
            } else {
                0
            };
            map_and_initialize(
                &bo,
                allocation_offset as usize,
                requirement.allocation_bytes as usize,
                initial,
            )?;
            if measure {
                timings.binding_initialize_ns = timings
                    .binding_initialize_ns
                    .saturating_add(super::super::profile::now_ns().saturating_sub(started));
            }
        }
        if is_new && !direct {
            total = total
                .checked_add(info.bytes)
                .ok_or(PrepareError::BudgetExceeded)?;
        }
        placements.push(PlacedSection {
            section: requirement.section,
            device_address,
            bytes: requirement.allocation_bytes,
        });
        buffers.push(BindingBuffer {
            section: requirement.section,
            class: requirement.class,
            logical_bytes: requirement.bytes,
            bo,
            backing,
            allocation_offset,
            direct,
            host_address: shared_view.map(|view| view.address),
        });
    }
    Ok((buffers, placements, total, timings))
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
    map_and_initialize(&command, 0, COMMAND_LOGICAL_BYTES, Some(&bytes))?;
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
    budget: PreparationBudget,
) -> Result<PreparedInference<'device>, PrepareError> {
    prepare_inference_shared(
        device,
        parameters,
        elf,
        hpi_requirements,
        input_data,
        &[],
        &[],
        budget,
    )
}

pub(in crate::vendor::ivpu) fn prepare_inference_shared<'device>(
    device: &'device IvpuDevice,
    parameters: &DeviceParameters,
    elf: &ParsedIntelElf<'_>,
    hpi_requirements: &HpiRequirements,
    input_data: &[&[u8]],
    input_views: &[Option<SharedAllocationView>],
    output_views: &[Option<SharedAllocationView>],
    budget: PreparationBudget,
) -> Result<PreparedInference<'device>, PrepareError> {
    validate_device(parameters).map_err(PrepareError::Validation)?;
    let measure = crate::trace::perf_enabled();
    let started = if measure {
        super::super::profile::now_ns()
    } else {
        0
    };
    let mut staged = stage_program(device, elf, hpi_requirements, budget.max_user_bo_bytes)
        .map_err(PrepareError::Load)?;
    let stage_program_ns = if measure {
        super::super::profile::now_ns().saturating_sub(started)
    } else {
        0
    };
    let requirements = collect_bindings(elf)?;
    let (bindings, binding_placements, binding_bo_bytes, mut timings) =
        create_bindings(device, &requirements, input_data, input_views, output_views)?;
    timings.stage_program_ns = stage_program_ns;
    let jit_relocations = staged.jit_relocations;
    let warm_rebind = if jit_relocations > 0 {
        match staged.snapshot_logical_bytes() {
            Ok(pre_jit_sections) => Some(WarmRebindState {
                pre_jit_sections,
                jit_relocations,
            }),
            Err(_) => None,
        }
    } else {
        None
    };
    let started = if measure {
        super::super::profile::now_ns()
    } else {
        0
    };
    staged
        .apply_jit_bindings(elf, &binding_placements)
        .map_err(PrepareError::Load)?;
    if measure {
        timings.link_ns = super::super::profile::now_ns().saturating_sub(started);
    }
    let started = if measure {
        super::super::profile::now_ns()
    } else {
        0
    };
    let (hpi, hpi_bo_bytes) = create_hpi(device, elf, hpi_requirements, staged.entry_address)?;
    if measure {
        timings.hpi_ns = super::super::profile::now_ns().saturating_sub(started);
    }
    let inference_id = device
        .get_param(DRM_IVPU_PARAM_UNIQUE_INFERENCE_ID, 0)
        .map_err(PrepareError::Runtime)?;
    let started = if measure {
        super::super::profile::now_ns()
    } else {
        0
    };
    let (command, command_bo_bytes, command_digest) = create_command(device, &hpi, inference_id)?;
    if measure {
        timings.command_ns = super::super::profile::now_ns().saturating_sub(started);
    }
    let total_user_bo_bytes = staged
        .total_bo_bytes
        .checked_add(binding_bo_bytes)
        .and_then(|total| total.checked_add(hpi_bo_bytes))
        .and_then(|total| total.checked_add(command_bo_bytes))
        .ok_or(PrepareError::BudgetExceeded)?;
    if total_user_bo_bytes > budget.max_user_bo_bytes {
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
    for binding in &bindings {
        let handle = binding.bo.handle();
        if !handles.contains(&handle) {
            handles.push(handle);
        }
    }
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
    for binding in &bindings {
        if ranges
            .iter()
            .any(|range| range.handle == binding.bo.handle())
        {
            continue;
        }
        let info = binding.bo.info().map_err(PrepareError::Runtime)?;
        ranges.push(MemoryRange {
            handle: binding.bo.handle(),
            role: super::transport::BufferRole::Binding,
            vpu_address: info.vpu_address,
            bytes: info.bytes,
        });
    }

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

    // Managed CMDQ submit with preemption index 0 causes the kernel to account the reported
    // preemption allocation. Include it in the caller's queried/configured total ceiling.
    let Some(total_accounted_bytes) =
        total_user_bo_bytes.checked_add(parameters.preempt_buffer_bytes)
    else {
        return Err(PrepareError::BudgetExceeded);
    };
    if total_accounted_bytes > budget.max_total_accounted_bytes {
        return Err(PrepareError::ExplicitQueuePreemptionExceedsBudget {
            user_bytes: total_user_bo_bytes,
            preemption_bytes: parameters.preempt_buffer_bytes,
        });
    }
    if command_info.flags & (DRM_IVPU_BO_MAPPABLE | DRM_IVPU_BO_WC)
        != DRM_IVPU_BO_MAPPABLE | DRM_IVPU_BO_WC
        || command_info.flags & super::uapi::DRM_IVPU_BO_READ_ONLY != 0
        || command_info.vpu_address != command.vpu_address()
        || hpi_info.flags & DRM_IVPU_BO_MAPPABLE == 0
    {
        return Err(PrepareError::InvalidMetadata);
    }
    for (section, placement) in staged.sections().iter().zip(staged.placements()) {
        let info = section.bo.info().map_err(PrepareError::Runtime)?;
        if info.vpu_address != placement.device_address
            || info.bytes < placement.bytes.max(1)
            || info.flags & !super::uapi::DRM_IVPU_BO_FLAGS != 0
        {
            return Err(PrepareError::InvalidMetadata);
        }
    }
    for binding in &bindings {
        let info = binding.bo.info().map_err(PrepareError::Runtime)?;
        let expected_backing = match binding.backing {
            BindingBacking::ImportedShared => binding.bo.backing() == BufferBacking::ImportedShared,
            BindingBacking::IvpuGem => {
                binding.bo.backing() == BufferBacking::IvpuGem
                    && info.flags & (DRM_IVPU_BO_MAPPABLE | DRM_IVPU_BO_DMA_MEM)
                        == DRM_IVPU_BO_MAPPABLE | DRM_IVPU_BO_DMA_MEM
            }
        };
        if !expected_backing
            || info.flags & super::uapi::DRM_IVPU_BO_READ_ONLY != 0
            || info.vpu_address != binding.bo.vpu_address()
            || binding
                .allocation_offset
                .checked_add(binding.logical_bytes)
                .is_none_or(|end| end > info.bytes)
        {
            return Err(PrepareError::InvalidMetadata);
        }
    }

    let executable_bo_bytes = staged.total_bo_bytes;
    let prepared = PreparedInference {
        staged,
        bindings,
        hpi,
        command,
        handles,
        executable_bo_bytes,
        binding_bo_bytes,
        hpi_bo_bytes,
        command_bo_bytes,
        preemption_bytes: parameters.preempt_buffer_bytes,
        total_user_bo_bytes,
        total_accounted_bytes,
        command_count,
        inference_id,
        command_digest,
        timings,
        warm_rebind,
    };
    prepared.zero_outputs()?;
    if !prepared.outputs_are_zero()? {
        return Err(PrepareError::InvalidMetadata);
    }
    Ok(prepared)
}

fn rebuild_handles(
    staged: &StagedProgram<'_>,
    hpi: &BufferObject<'_>,
    command: &BufferObject<'_>,
    bindings: &[BindingBuffer<'_>],
) -> Result<Vec<u32>, PrepareError> {
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
    for binding in bindings {
        let handle = binding.bo.handle();
        if !handles.contains(&handle) {
            handles.push(handle);
        }
    }
    for (index, &handle) in handles.iter().enumerate() {
        if handle == 0 || handles[..index].contains(&handle) {
            return Err(PrepareError::InvalidMetadata);
        }
    }
    Ok(handles)
}

/// Re-JIT IO bindings onto an already-staged program using its warm pre-JIT snapshot.
/// Keeps executable/HPI/command BOs; only binding imports and JIT patches change (#5).
pub(in crate::vendor::ivpu) fn rebind_inference_shared<'device>(
    device: &'device IvpuDevice,
    prepared: &mut PreparedInference<'device>,
    elf: &ParsedIntelElf<'_>,
    input_data: &[&[u8]],
    input_views: &[Option<SharedAllocationView>],
    output_views: &[Option<SharedAllocationView>],
) -> Result<(), PrepareError> {
    let warm = prepared
        .warm_rebind
        .as_ref()
        .ok_or(PrepareError::InvalidMetadata)?;
    prepared
        .staged
        .restore_logical_bytes(&warm.pre_jit_sections, warm.jit_relocations)
        .map_err(PrepareError::Load)?;
    let requirements = collect_bindings(elf)?;
    let (bindings, binding_placements, binding_bo_bytes, _) =
        create_bindings(device, &requirements, input_data, input_views, output_views)?;
    prepared
        .staged
        .apply_jit_bindings(elf, &binding_placements)
        .map_err(PrepareError::Load)?;
    let handles = rebuild_handles(&prepared.staged, &prepared.hpi, &prepared.command, &bindings)?;
    let total_user_bo_bytes = prepared
        .executable_bo_bytes
        .checked_add(binding_bo_bytes)
        .and_then(|total| total.checked_add(prepared.hpi_bo_bytes))
        .and_then(|total| total.checked_add(prepared.command_bo_bytes))
        .ok_or(PrepareError::BudgetExceeded)?;
    let total_accounted_bytes = total_user_bo_bytes
        .checked_add(prepared.preemption_bytes)
        .ok_or(PrepareError::BudgetExceeded)?;
    prepared.bindings = bindings;
    prepared.handles = handles;
    prepared.binding_bo_bytes = binding_bo_bytes;
    prepared.total_user_bo_bytes = total_user_bo_bytes;
    prepared.total_accounted_bytes = total_accounted_bytes;
    prepared.zero_outputs()?;
    Ok(())
}

#[cfg(test)]
mod tests {
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
    #[ignore = "full dry-run preparation only; allocates bounded BOs but never submits"]
    fn prepare_tiny_artifact_with_accounted_preemption() {
        let bytes = super::super::load_test_artifact(1024 * 1024);
        let elf = parse_intel_elf(&bytes).unwrap();
        let device = IvpuDevice::open_first().unwrap();
        let parameters = device.parameters().unwrap();
        let hpi = inspect_hpi_requirements(&elf, &parameters).unwrap();
        let input0 = [0x11u8; 128];
        let input1 = [0x22u8; 128];
        let prepared = prepare_inference(
            &device,
            &parameters,
            &elf,
            &hpi,
            &[&input0, &input1],
            PreparationBudget {
                max_user_bo_bytes: 64 * 1024 * 1024,
                max_total_accounted_bytes: 64 * 1024 * 1024,
            },
        )
        .unwrap();
        std::println!(
            "device={} artifact={} preempt_buffer_bytes={} user_bo_bytes={} accounted_bytes={}",
            device.device_path().to_string_lossy(),
            super::super::test_artifact_location().to_string_lossy(),
            parameters.preempt_buffer_bytes,
            prepared.total_user_bo_bytes,
            prepared.total_accounted_bytes,
        );
        assert!(prepared.total_user_bo_bytes < 1024 * 1024);
        assert_eq!(
            prepared.total_accounted_bytes,
            prepared.total_user_bo_bytes + parameters.preempt_buffer_bytes
        );
        assert!(prepared.total_accounted_bytes <= 64 * 1024 * 1024);
        assert!(prepared.outputs_are_zero().unwrap());
    }
}
