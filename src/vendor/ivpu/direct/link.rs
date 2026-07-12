//! Controlled device-link operations for a deliberately small relocation allowlist.
//!
//! This operates on bounded host slices only. It never maps or submits device memory.

#![allow(dead_code)]

use goblin::elf::section_header;

use crate::address::ByteLen;

use super::super::address::{IvpuAddr, IvpuOffset, SectionOffset, SectionRange};
use super::elf::{DeviceRelocation, DeviceSection, ElfError, ParsedIntelElf};

pub const VPU_SHF_USER_INPUT: u64 = 0x0020_0000;
pub const VPU_SHF_USER_OUTPUT: u64 = 0x0040_0000;
pub const VPU_SHF_PROFILING_OUTPUT: u64 = 0x0080_0000;
pub const VPU_SHF_PROC_SHAVE: u64 = 0x4000_0000;

pub const R_VPU_64: u32 = 0;
pub const R_VPU_64_OR: u32 = 1;
pub const R_VPU_32: u32 = 4;
pub const R_VPU_32_SUM: u32 = 6;
pub const R_VPU_16_SUM: u32 = 1011;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkError {
    InvalidElf(ElfError),
    UnsupportedRelocation(u32),
    TargetOutOfBounds,
    ArithmeticOverflow,
    MissingPlacement,
    UnsupportedRuntimeSymbol,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkOperation {
    pub target_section: usize,
    pub target_offset: u64,
    pub symbol_table: usize,
    pub symbol: usize,
    pub relocation_type: u32,
    pub addend: i64,
    pub write_bytes: u8,
    pub jit: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceMemoryClass {
    Firmware,
    Shave,
    Dma,
    Input,
    Output,
    Profiling,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SectionPlacement<'a> {
    pub section: usize,
    pub class: DeviceMemoryClass,
    pub bytes: u64,
    pub allocation_bytes: u64,
    pub alignment: u64,
    pub bo_flags: u32,
    pub initial_data: Option<&'a [u8]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingRequirement {
    pub section: usize,
    pub class: DeviceMemoryClass,
    /// Host-visible tensor bytes from the section symbol. This can be smaller than the ELF
    /// section because the compiler reserves private DMA padding after the logical tensor.
    pub bytes: u64,
    /// Full NOBITS section extent that firmware may address.
    pub allocation_bytes: u64,
    pub alignment: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacedSection {
    pub section: usize,
    pub device_address: u64,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeSection {
    pub section_type: u32,
    pub device_address: u64,
    pub bytes: u64,
}

fn write_width(relocation_type: u32) -> Result<u8, LinkError> {
    match relocation_type {
        R_VPU_64 | R_VPU_64_OR => Ok(8),
        R_VPU_32 | R_VPU_32_SUM => Ok(4),
        R_VPU_16_SUM => Ok(2),
        other => Err(LinkError::UnsupportedRelocation(other)),
    }
}

/// Produce link operations without allocating or mutating the artifact.
pub fn visit_link_operations(
    elf: &ParsedIntelElf<'_>,
    mut visitor: impl FnMut(LinkOperation),
) -> Result<(), LinkError> {
    let mut result = Ok(());
    elf.visit_relocations(|relocation: DeviceRelocation| {
        if result.is_err() {
            return;
        }
        match write_width(relocation.relocation_type) {
            Ok(write_bytes) => visitor(LinkOperation {
                target_section: relocation.target_section,
                target_offset: relocation.offset,
                symbol_table: relocation.symbol_table,
                symbol: relocation.symbol,
                relocation_type: relocation.relocation_type,
                addend: relocation.addend.unwrap_or(0),
                write_bytes,
                jit: relocation.jit,
            }),
            Err(error) => result = Err(error),
        }
    })
    .map_err(LinkError::InvalidElf)?;
    result
}

pub fn visit_section_placements<'a>(
    elf: &ParsedIntelElf<'a>,
    mut visitor: impl FnMut(SectionPlacement<'a>),
) {
    elf.visit_sections(|section: DeviceSection<'a>| {
        let allocate = section.flags & u64::from(section_header::SHF_ALLOC) != 0
            && matches!(
                section.section_type,
                section_header::SHT_PROGBITS | section_header::SHT_NOBITS
            )
            && section.flags
                & (VPU_SHF_USER_INPUT | VPU_SHF_USER_OUTPUT | VPU_SHF_PROFILING_OUTPUT)
                == 0;
        let class = allocate.then_some(
            if section.flags & u64::from(section_header::SHF_EXECINSTR) != 0 {
                DeviceMemoryClass::Firmware
            } else if section.flags & VPU_SHF_PROC_SHAVE != 0 {
                DeviceMemoryClass::Shave
            } else {
                DeviceMemoryClass::Dma
            },
        );
        if let Some(class) = class {
            let bo_flags = match class {
                DeviceMemoryClass::Firmware => {
                    super::uapi::DRM_IVPU_BO_WC | super::uapi::DRM_IVPU_BO_MAPPABLE
                }
                DeviceMemoryClass::Shave => {
                    super::uapi::DRM_IVPU_BO_WC
                        | super::uapi::DRM_IVPU_BO_MAPPABLE
                        | super::uapi::DRM_IVPU_BO_HIGH_MEM
                }
                DeviceMemoryClass::Dma => {
                    super::uapi::DRM_IVPU_BO_WC
                        | super::uapi::DRM_IVPU_BO_MAPPABLE
                        | super::uapi::DRM_IVPU_BO_DMA_MEM
                }
                _ => return,
            };
            visitor(SectionPlacement {
                section: section.index,
                class,
                bytes: section.memory_bytes,
                allocation_bytes: section.memory_bytes.max(1),
                alignment: section.alignment.max(1),
                bo_flags,
                initial_data: section.data,
            });
        }
    });
}

pub fn visit_binding_requirements(
    elf: &ParsedIntelElf<'_>,
    mut visitor: impl FnMut(BindingRequirement),
) {
    elf.visit_sections(|section| {
        if section.section_type != section_header::SHT_NOBITS {
            return;
        }
        let class = if section.flags & VPU_SHF_USER_INPUT != 0 {
            Some(DeviceMemoryClass::Input)
        } else if section.flags & VPU_SHF_USER_OUTPUT != 0 {
            Some(DeviceMemoryClass::Output)
        } else if section.flags & VPU_SHF_PROFILING_OUTPUT != 0 {
            Some(DeviceMemoryClass::Profiling)
        } else {
            None
        };
        if let Some(class) = class {
            let mut logical = BindingLogicalSize::new(section.memory_bytes);
            if elf
                .visit_symbols(|symbol| {
                    if symbol.section_index == section.index {
                        logical.consider(symbol.value, symbol.bytes);
                    }
                })
                .is_err()
            {
                logical.ambiguous = true;
            }
            visitor(BindingRequirement {
                section: section.index,
                class,
                bytes: logical.finish(),
                allocation_bytes: section.memory_bytes,
                alignment: section.alignment.max(1),
            });
        }
    });
}

#[derive(Clone, Copy)]
struct BindingLogicalSize {
    allocation_bytes: u64,
    selected: Option<u64>,
    ambiguous: bool,
}

impl BindingLogicalSize {
    const fn new(allocation_bytes: u64) -> Self {
        Self {
            allocation_bytes,
            selected: None,
            ambiguous: false,
        }
    }

    fn consider(&mut self, value: u64, bytes: u64) {
        if value != 0 || bytes == 0 || bytes > self.allocation_bytes {
            return;
        }
        match self.selected {
            None => self.selected = Some(bytes),
            Some(selected) if selected == bytes => {}
            Some(_) => self.ambiguous = true,
        }
    }

    const fn finish(self) -> u64 {
        if self.ambiguous {
            self.allocation_bytes
        } else if let Some(bytes) = self.selected {
            bytes
        } else {
            self.allocation_bytes
        }
    }
}

pub fn resolve_symbol(
    elf: &ParsedIntelElf<'_>,
    symbol_table: usize,
    symbol_index: usize,
    placements: &[PlacedSection],
    runtime_sections: &[RuntimeSection],
) -> Result<u64, LinkError> {
    let symbol = elf
        .find_symbol(symbol_table, symbol_index)
        .map_err(LinkError::InvalidElf)?;
    if symbol.section_index >= 0xff00 {
        // Runtime-owned symbols (CMX slice bases, barriers, FIFOs, registers) require values from
        // the version-matched HostParsedInference/device descriptor and are never guessed.
        return Err(LinkError::UnsupportedRuntimeSymbol);
    }
    let target_section = elf
        .find_section(symbol.section_index)
        .map_err(LinkError::InvalidElf)?;
    let (base, available, add_symbol_value) = if let Some(placement) = placements
        .iter()
        .find(|placement| placement.section == symbol.section_index)
    {
        (
            IvpuAddr::new(placement.device_address),
            ByteLen::new(placement.bytes),
            true,
        )
    } else if let Some(runtime) = runtime_sections
        .iter()
        .find(|runtime| runtime.section_type == target_section.section_type)
    {
        // Pinned HPI symtab override mode replaces the blob symbol for non-allocated CMX
        // sections; it does not add the blob's st_value.
        (
            IvpuAddr::new(runtime.device_address),
            ByteLen::new(runtime.bytes),
            false,
        )
    } else {
        return Err(LinkError::MissingPlacement);
    };
    let available = SectionRange::from_start_len(SectionOffset::new(0), available)
        .map_err(|_| LinkError::ArithmeticOverflow)?;
    let symbol_range =
        SectionRange::from_start_len(SectionOffset::new(symbol.value), ByteLen::new(symbol.bytes))
            .map_err(|_| LinkError::ArithmeticOverflow)?;
    if placements
        .iter()
        .any(|placement| placement.section == symbol.section_index)
        && !available.contains_range(symbol_range)
    {
        return Err(LinkError::TargetOutOfBounds);
    }
    if add_symbol_value {
        base.checked_add_offset(IvpuOffset::new(symbol.value))
            .map(IvpuAddr::get)
            .ok_or(LinkError::ArithmeticOverflow)
    } else {
        Ok(base.get())
    }
}

fn relocated_value(symbol: u64, addend: i64) -> Result<u64, LinkError> {
    if addend >= 0 {
        symbol
            .checked_add(addend as u64)
            .ok_or(LinkError::ArithmeticOverflow)
    } else {
        symbol
            .checked_sub(addend.unsigned_abs())
            .ok_or(LinkError::ArithmeticOverflow)
    }
}

/// Apply one already validated operation to a bounded host section image.
pub fn apply_link_operation(
    target: &mut [u8],
    operation: LinkOperation,
    resolved_symbol: u64,
) -> Result<(), LinkError> {
    let target_range = SectionRange::from_start_len(
        SectionOffset::new(0),
        ByteLen::try_from_usize(target.len()).map_err(|_| LinkError::TargetOutOfBounds)?,
    )
    .map_err(|_| LinkError::TargetOutOfBounds)?;
    let write_range = SectionRange::from_start_len_non_empty(
        SectionOffset::new(operation.target_offset),
        ByteLen::new(u64::from(operation.write_bytes)),
    )
    .map_err(|_| LinkError::TargetOutOfBounds)?;
    if !target_range.contains_range(write_range) {
        return Err(LinkError::TargetOutOfBounds);
    }
    let write_range = write_range
        .try_usize_range()
        .map_err(|_| LinkError::TargetOutOfBounds)?;
    let bytes = target
        .get_mut(write_range)
        .ok_or(LinkError::TargetOutOfBounds)?;
    let value = relocated_value(resolved_symbol, operation.addend)?;
    match operation.relocation_type {
        R_VPU_64 => bytes.copy_from_slice(&value.to_le_bytes()),
        R_VPU_64_OR => {
            let current = u64::from_le_bytes(bytes.try_into().unwrap());
            bytes.copy_from_slice(&(current | value).to_le_bytes());
        }
        R_VPU_32 => {
            let value = u32::try_from(value).map_err(|_| LinkError::ArithmeticOverflow)?;
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        R_VPU_32_SUM => {
            let current = u32::from_le_bytes(bytes.try_into().unwrap());
            let add = u32::try_from(value).map_err(|_| LinkError::ArithmeticOverflow)?;
            bytes.copy_from_slice(
                &current
                    .checked_add(add)
                    .ok_or(LinkError::ArithmeticOverflow)?
                    .to_le_bytes(),
            );
        }
        R_VPU_16_SUM => {
            let current = u16::from_le_bytes(bytes.try_into().unwrap());
            let add = u16::try_from(value).map_err(|_| LinkError::ArithmeticOverflow)?;
            bytes.copy_from_slice(
                &current
                    .checked_add(add)
                    .ok_or(LinkError::ArithmeticOverflow)?
                    .to_le_bytes(),
            );
        }
        other => return Err(LinkError::UnsupportedRelocation(other)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn operation(relocation_type: u32, write_bytes: u8) -> LinkOperation {
        LinkOperation {
            target_section: 1,
            target_offset: 0,
            symbol_table: 2,
            symbol: 3,
            relocation_type,
            addend: 4,
            write_bytes,
            jit: false,
        }
    }

    #[test]
    fn controlled_linker_applies_allowlisted_relocations() {
        let mut target = [0u8; 8];
        apply_link_operation(&mut target, operation(R_VPU_64, 8), 8).unwrap();
        assert_eq!(u64::from_le_bytes(target), 12);

        let mut target = [1u8, 0, 0, 0];
        apply_link_operation(&mut target, operation(R_VPU_32_SUM, 4), 2).unwrap();
        assert_eq!(u32::from_le_bytes(target), 7);
    }

    #[test]
    fn controlled_linker_rejects_overflow_and_unknown_types() {
        let mut target = [0u8; 4];
        assert_eq!(
            apply_link_operation(&mut target, operation(R_VPU_32, 4), u64::MAX),
            Err(LinkError::ArithmeticOverflow)
        );
        assert_eq!(
            apply_link_operation(&mut target, operation(25, 4), 0),
            Err(LinkError::UnsupportedRelocation(25))
        );
        assert_eq!(
            apply_link_operation(&mut [], operation(R_VPU_64, 8), 0),
            Err(LinkError::TargetOutOfBounds)
        );
    }

    #[test]
    fn binding_symbol_size_preserves_private_section_padding() {
        let mut size = BindingLogicalSize::new(138_240);
        size.consider(0, 131_072);
        assert_eq!(size.finish(), 131_072);

        let mut invalid = BindingLogicalSize::new(138_240);
        invalid.consider(1, 131_072);
        invalid.consider(0, 140_000);
        assert_eq!(invalid.finish(), 138_240);

        let mut ambiguous = BindingLogicalSize::new(138_240);
        ambiguous.consider(0, 131_072);
        ambiguous.consider(0, 130_000);
        assert_eq!(ambiguous.finish(), 138_240);
    }
}
