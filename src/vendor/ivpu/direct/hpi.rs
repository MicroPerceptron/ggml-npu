//! Minimal HostParsedInference compatibility semantics for NPU40XX.
//!
//! Sources are pinned in `docs/direct-runtime-foundation.md`:
//! `npu_compiler_elf@82c444...` and `linux-npu-driver v1.32.0`.

#![allow(dead_code)]

use goblin::elf::section_header;

use super::elf::{
    DeviceSection, ElfError, ParsedIntelElf, VPU_SHT_CMX_METADATA, VPU_SHT_CMX_WORKSPACE,
};
use super::firmware::{ELF_LOADER_ABI_MAJOR, ELF_LOADER_ABI_MINOR, NNRT_API_MAJOR, NNRT_API_MINOR};
use super::link::{
    visit_binding_requirements, BindingRequirement, RuntimeSection, VPU_SHF_PROFILING_OUTPUT,
    VPU_SHF_USER_INPUT, VPU_SHF_USER_OUTPUT,
};
use super::DeviceParameters;

const VPU_SHT_NETDESC: u32 = 0x8aaa_aaaa;
const VPU_STT_ENTRY: u8 = 10;
const NT_GNU_ABI_TAG: u32 = 1;
const NT_NPU_MPI_VERSION: u32 = 0xa000;
const MAX_METADATA_IO: u64 = 32;
const SERIAL_DESCRIPTOR_BYTES: usize = 32;
const IDENTIFICATION_BYTES: u64 = 512;
const RESOURCE_REQUIREMENTS_BYTES: u64 = 12;
const TENSOR_REF_BYTES: u64 = 384;

// linux-npu-driver v1.32.0:
// firmware/include/api/vpu_cmx_info_40xx.h and hpi_component/src/4000/hpi_4000.cpp.
pub const CMX_METADATA_ADDRESS: u64 = 0x4020_3c00;
pub const CMX_METADATA_BYTES: u64 = 81 * 1024;
pub const CMX_WORKSPACE_ADDRESS: u64 = 0x4021_8000;
pub const CMX_WORKSPACE_BYTES: u64 = 1440 * 1024;
pub const HOST_PARSED_INFERENCE_BYTES: u64 = 384;
pub const HOST_PARSED_INFERENCE_ALIGNMENT: u64 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HpiError {
    Elf(ElfError),
    MissingMetadata,
    DuplicateMetadata,
    InvalidSerialization,
    VersionMismatch,
    TileCountExceeded,
    BindingCountMismatch,
    MissingEntry,
    DuplicateEntry,
    UnresolvedSymbol,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceRequirements {
    pub cmx_slice_bytes: u32,
    pub slice_count: u8,
    pub barriers: u8,
    pub inputs: u8,
    pub outputs: u8,
    pub profiling_outputs: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactVersions {
    pub loader_major: u32,
    pub loader_minor: u32,
    pub loader_patch: u32,
    pub mapped_inference_major: u32,
    pub mapped_inference_minor: u32,
    pub mapped_inference_patch: u32,
    pub mapped_inference_format: u32,
}

pub const NPU40XX_RUNTIME_SYMBOLS: [RuntimeSection; 2] = [
    RuntimeSection {
        section_type: VPU_SHT_CMX_METADATA,
        device_address: CMX_METADATA_ADDRESS,
        bytes: CMX_METADATA_BYTES,
    },
    RuntimeSection {
        section_type: VPU_SHT_CMX_WORKSPACE,
        device_address: CMX_WORKSPACE_ADDRESS,
        bytes: CMX_WORKSPACE_BYTES,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HpiRequirements {
    pub resources: ResourceRequirements,
    pub versions: ArtifactVersions,
    pub entry_section: usize,
    pub entry_bytes: u64,
    pub runtime_symbols: [RuntimeSection; 2],
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, HpiError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or(HpiError::InvalidSerialization)?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, HpiError> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or(HpiError::InvalidSerialization)?;
    Ok(u64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}

#[derive(Clone, Copy)]
struct SerialDescriptor {
    data_offset: usize,
    next_offset: usize,
    count: u64,
    element_bytes: u64,
}

fn serial_descriptor(bytes: &[u8], offset: usize) -> Result<SerialDescriptor, HpiError> {
    let descriptor = bytes
        .get(offset..offset + SERIAL_DESCRIPTOR_BYTES)
        .ok_or(HpiError::InvalidSerialization)?;
    let data_offset =
        usize::try_from(read_u64(descriptor, 0)?).map_err(|_| HpiError::InvalidSerialization)?;
    let next_offset =
        usize::try_from(read_u64(descriptor, 8)?).map_err(|_| HpiError::InvalidSerialization)?;
    let count = read_u64(descriptor, 16)?;
    let element_bytes = read_u64(descriptor, 24)?;
    let payload = count
        .checked_mul(element_bytes)
        .and_then(|size| u64::try_from(data_offset).ok()?.checked_add(size))
        .ok_or(HpiError::InvalidSerialization)?;
    if payload > bytes.len() as u64
        || (next_offset != 0
            && (next_offset < offset + SERIAL_DESCRIPTOR_BYTES || payload > next_offset as u64))
    {
        return Err(HpiError::InvalidSerialization);
    }
    Ok(SerialDescriptor {
        data_offset,
        next_offset,
        count,
        element_bytes,
    })
}

fn metadata_section<'a>(elf: &ParsedIntelElf<'a>) -> Result<DeviceSection<'a>, HpiError> {
    let mut metadata = None;
    let mut duplicate = false;
    elf.visit_sections(|section| {
        if section.section_type == VPU_SHT_NETDESC {
            if metadata.is_some() {
                duplicate = true;
            } else {
                metadata = Some(section);
            }
        }
    });
    if duplicate {
        return Err(HpiError::DuplicateMetadata);
    }
    metadata.ok_or(HpiError::MissingMetadata)
}

pub fn inspect_metadata(elf: &ParsedIntelElf<'_>) -> Result<ResourceRequirements, HpiError> {
    let bytes = metadata_section(elf)?
        .data
        .ok_or(HpiError::MissingMetadata)?;
    let identification = serial_descriptor(bytes, 0)?;
    if identification.count != 1
        || identification.element_bytes != IDENTIFICATION_BYTES
        || identification.next_offset == 0
    {
        return Err(HpiError::InvalidSerialization);
    }
    let resources = serial_descriptor(bytes, identification.next_offset)?;
    if resources.count != 1
        || resources.element_bytes != RESOURCE_REQUIREMENTS_BYTES
        || resources.next_offset == 0
    {
        return Err(HpiError::InvalidSerialization);
    }
    let resource_data = bytes
        .get(resources.data_offset..resources.data_offset + RESOURCE_REQUIREMENTS_BYTES as usize)
        .ok_or(HpiError::InvalidSerialization)?;
    if resource_data[4..10] != [0; 6] {
        return Err(HpiError::InvalidSerialization);
    }
    let inputs = serial_descriptor(bytes, resources.next_offset)?;
    let outputs = serial_descriptor(bytes, inputs.next_offset)?;
    if inputs.element_bytes != TENSOR_REF_BYTES
        || outputs.element_bytes != TENSOR_REF_BYTES
        || inputs.count > MAX_METADATA_IO
        || outputs.count > MAX_METADATA_IO
    {
        return Err(HpiError::InvalidSerialization);
    }
    let profiling = if outputs.next_offset != 0 {
        let input_descriptors = serial_descriptor(bytes, outputs.next_offset)?;
        let output_descriptors = serial_descriptor(bytes, input_descriptors.next_offset)?;
        serial_descriptor(bytes, output_descriptors.next_offset)?
    } else {
        return Err(HpiError::InvalidSerialization);
    };
    if profiling.count > MAX_METADATA_IO {
        return Err(HpiError::InvalidSerialization);
    }
    Ok(ResourceRequirements {
        cmx_slice_bytes: read_u32(resource_data, 0)?,
        slice_count: resource_data[10],
        barriers: resource_data[11],
        inputs: inputs.count as u8,
        outputs: outputs.count as u8,
        profiling_outputs: profiling.count as u8,
    })
}

fn align4(value: usize) -> Result<usize, HpiError> {
    value
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or(HpiError::InvalidSerialization)
}

fn parse_note(section: DeviceSection<'_>) -> Result<(u32, [u32; 4]), HpiError> {
    let data = section.data.ok_or(HpiError::InvalidSerialization)?;
    let name_bytes = read_u32(data, 0)? as usize;
    let descriptor_bytes = read_u32(data, 4)? as usize;
    let note_type = read_u32(data, 8)?;
    let descriptor = 12usize
        .checked_add(align4(name_bytes)?)
        .ok_or(HpiError::InvalidSerialization)?;
    if descriptor_bytes != 16 || descriptor + descriptor_bytes > data.len() {
        return Err(HpiError::InvalidSerialization);
    }
    Ok((
        note_type,
        [
            read_u32(data, descriptor)?,
            read_u32(data, descriptor + 4)?,
            read_u32(data, descriptor + 8)?,
            read_u32(data, descriptor + 12)?,
        ],
    ))
}

fn parse_versions(elf: &ParsedIntelElf<'_>) -> Result<ArtifactVersions, HpiError> {
    let mut loader = None;
    let mut mapped = None;
    let mut invalid = None;
    elf.visit_sections(|section| {
        if section.section_type != section_header::SHT_NOTE {
            return;
        }
        match parse_note(section) {
            Ok((NT_GNU_ABI_TAG, version)) => loader = Some(version),
            Ok((NT_NPU_MPI_VERSION, version)) => mapped = Some(version),
            Ok(_) => {}
            Err(error) => invalid = Some(error),
        }
    });
    if let Some(error) = invalid {
        return Err(error);
    }
    let loader = loader.ok_or(HpiError::VersionMismatch)?;
    let mapped = mapped.ok_or(HpiError::VersionMismatch)?;
    // n_desc[0] is the GNU OS field; version is n_desc[1..=3].
    Ok(ArtifactVersions {
        loader_major: loader[1],
        loader_minor: loader[2],
        loader_patch: loader[3],
        mapped_inference_major: mapped[1],
        mapped_inference_minor: mapped[2],
        mapped_inference_patch: mapped[3],
        mapped_inference_format: mapped[0],
    })
}

fn validate_bindings(
    elf: &ParsedIntelElf<'_>,
    resources: ResourceRequirements,
) -> Result<(), HpiError> {
    let mut inputs = 0u8;
    let mut outputs = 0u8;
    let mut profiling = 0u8;
    visit_binding_requirements(elf, |binding: BindingRequirement| match binding.class {
        super::link::DeviceMemoryClass::Input => inputs = inputs.saturating_add(1),
        super::link::DeviceMemoryClass::Output => outputs = outputs.saturating_add(1),
        super::link::DeviceMemoryClass::Profiling => profiling = profiling.saturating_add(1),
        _ => {}
    });
    if inputs != resources.inputs
        || outputs != resources.outputs
        || profiling != resources.profiling_outputs
    {
        return Err(HpiError::BindingCountMismatch);
    }
    Ok(())
}

fn entry_symbol(elf: &ParsedIntelElf<'_>) -> Result<(usize, u64), HpiError> {
    let mut entry = None;
    let mut duplicate = false;
    elf.visit_symbols(|symbol| {
        if symbol.symbol_type == VPU_STT_ENTRY {
            if entry.is_some() {
                duplicate = true;
            } else {
                entry = Some(symbol.section_index);
            }
        }
    })
    .map_err(HpiError::Elf)?;
    if duplicate {
        return Err(HpiError::DuplicateEntry);
    }
    let section_index = entry.ok_or(HpiError::MissingEntry)?;
    let section = elf.find_section(section_index).map_err(HpiError::Elf)?;
    Ok((section_index, section.memory_bytes))
}

fn validate_symbol_sources(elf: &ParsedIntelElf<'_>) -> Result<(), HpiError> {
    let mut unresolved = false;
    elf.visit_relocations(|relocation| {
        let Ok(symbol) = elf.find_symbol(relocation.symbol_table, relocation.symbol) else {
            unresolved = true;
            return;
        };
        let Ok(section) = elf.find_section(symbol.section_index) else {
            unresolved = true;
            return;
        };
        let allocated = section.flags & u64::from(section_header::SHF_ALLOC) != 0
            && matches!(
                section.section_type,
                section_header::SHT_PROGBITS | section_header::SHT_NOBITS
            )
            && section.flags
                & (VPU_SHF_USER_INPUT | VPU_SHF_USER_OUTPUT | VPU_SHF_PROFILING_OUTPUT)
                == 0;
        let binding = section.flags
            & (VPU_SHF_USER_INPUT | VPU_SHF_USER_OUTPUT | VPU_SHF_PROFILING_OUTPUT)
            != 0;
        let runtime = NPU40XX_RUNTIME_SYMBOLS
            .iter()
            .any(|runtime| runtime.section_type == section.section_type);
        if !allocated && !binding && !runtime {
            unresolved = true;
        }
    })
    .map_err(HpiError::Elf)?;
    if unresolved {
        return Err(HpiError::UnresolvedSymbol);
    }
    Ok(())
}

pub fn inspect_hpi_requirements(
    elf: &ParsedIntelElf<'_>,
    device: &DeviceParameters,
) -> Result<HpiRequirements, HpiError> {
    let resources = inspect_metadata(elf)?;
    let versions = parse_versions(elf)?;
    if versions.loader_major != ELF_LOADER_ABI_MAJOR
        || versions.loader_minor > ELF_LOADER_ABI_MINOR
        || versions.mapped_inference_major != NNRT_API_MAJOR
        || versions.mapped_inference_minor > NNRT_API_MINOR
    {
        return Err(HpiError::VersionMismatch);
    }
    // vpu_device.cpp at v1.32.0 decodes the raw fused-off TILE_CONFIG bits this way for NPU40XX.
    let available_tiles = (!device.tile_config & 0x3f).count_ones();
    if u32::from(resources.slice_count) > available_tiles {
        return Err(HpiError::TileCountExceeded);
    }
    validate_bindings(elf, resources)?;
    validate_symbol_sources(elf)?;
    let (entry_section, entry_bytes) = entry_symbol(elf)?;
    Ok(HpiRequirements {
        resources,
        versions,
        entry_section,
        entry_bytes,
        runtime_symbols: NPU40XX_RUNTIME_SYMBOLS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_runtime_symbols_match_public_npu40xx_headers() {
        assert_eq!(CMX_METADATA_ADDRESS, 0x4020_3c00);
        assert_eq!(CMX_WORKSPACE_ADDRESS, 0x4021_8000);
        assert_eq!(CMX_WORKSPACE_BYTES, 0x0016_8000);
        assert_eq!(HOST_PARSED_INFERENCE_BYTES, 384);
    }

    #[test]
    fn malformed_serial_descriptor_is_rejected() {
        let mut bytes = [0u8; 40];
        bytes[0..8].copy_from_slice(&32u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        bytes[24..32].copy_from_slice(&8u64.to_le_bytes());
        assert!(matches!(
            serial_descriptor(&bytes, 0),
            Err(HpiError::InvalidSerialization)
        ));
    }
}
