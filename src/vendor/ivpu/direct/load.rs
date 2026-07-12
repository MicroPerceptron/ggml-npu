//! Bounded ELF section staging and static relocation.
//!
//! This module creates no queue or command and cannot submit work.

#![allow(dead_code)]

use alloc::vec::Vec;
use core::ptr;

use crate::host::MemoryMapping;
use crate::identity::{SemanticDigest, SemanticHasher};
use crate::runtime::RuntimeError;

use super::elf::ParsedIntelElf;
use super::hpi::{HpiRequirements, NPU40XX_RUNTIME_SYMBOLS};
use super::link::{
    apply_link_operation, resolve_symbol, visit_link_operations, visit_section_placements,
    DeviceMemoryClass, LinkError, LinkOperation, PlacedSection, SectionPlacement,
};
use super::transport::{acquire_device_writes, release_host_writes};
use super::{BufferObject, IvpuDevice};

pub const MAX_STAGED_SECTIONS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadError {
    Runtime(RuntimeError),
    Link(LinkError),
    TooManySections,
    BudgetExceeded,
    InvalidAlignment,
    InvalidSection,
    Allocation,
}

pub struct StagedSection<'device> {
    pub section: usize,
    pub class: DeviceMemoryClass,
    pub logical_bytes: u64,
    pub bo: BufferObject<'device>,
}

pub struct StagedProgram<'device> {
    sections: Vec<StagedSection<'device>>,
    placements: Vec<PlacedSection>,
    pub total_bo_bytes: u64,
    pub entry_address: u64,
    pub jit_relocations: usize,
    pub plan_digest: SemanticDigest,
    pub linked_digest: SemanticDigest,
}

impl StagedProgram<'_> {
    pub fn sections(&self) -> &[StagedSection<'_>] {
        &self.sections
    }

    pub fn placements(&self) -> &[PlacedSection] {
        &self.placements
    }

    pub fn destroy(self) {
        drop(self);
    }

    /// Host copy of each section's logical bytes after static link and before JIT binding patches.
    /// Used to re-apply JIT for a new IO binding set without re-staging the ELF (#5 warm reuse).
    pub fn snapshot_logical_bytes(&self) -> Result<Vec<(usize, Vec<u8>)>, LoadError> {
        let mut snap = Vec::new();
        snap.try_reserve_exact(self.sections.len())
            .map_err(|_| LoadError::Allocation)?;
        for section in &self.sections {
            let mapping = section.bo.map(false).map_err(LoadError::Runtime)?;
            let logical =
                usize::try_from(section.logical_bytes).map_err(|_| LoadError::InvalidSection)?;
            if logical > mapping.len() {
                return Err(LoadError::InvalidSection);
            }
            acquire_device_writes();
            // SAFETY: read-only copy of the live, bounds-checked mapping.
            let bytes =
                unsafe { core::slice::from_raw_parts(mapping.address().as_ptr(), logical) };
            let mut owned = Vec::new();
            owned
                .try_reserve_exact(logical)
                .map_err(|_| LoadError::Allocation)?;
            owned.extend_from_slice(bytes);
            snap.push((section.section, owned));
        }
        Ok(snap)
    }

    pub fn restore_logical_bytes(
        &mut self,
        snapshot: &[(usize, Vec<u8>)],
        jit_relocations: usize,
    ) -> Result<(), LoadError> {
        if snapshot.len() != self.sections.len() {
            return Err(LoadError::InvalidSection);
        }
        for (section, bytes) in self.sections.iter().zip(snapshot.iter()) {
            if section.section != bytes.0
                || bytes.1.len() as u64 != section.logical_bytes
            {
                return Err(LoadError::InvalidSection);
            }
            stage_bytes(section, Some(&bytes.1))?;
        }
        self.jit_relocations = jit_relocations;
        self.linked_digest = linked_digest(&self.sections)?;
        Ok(())
    }

    pub fn apply_jit_bindings(
        &mut self,
        elf: &ParsedIntelElf<'_>,
        bindings: &[PlacedSection],
    ) -> Result<(), LoadError> {
        let mut all_placements = Vec::new();
        all_placements
            .try_reserve_exact(self.placements.len() + bindings.len())
            .map_err(|_| LoadError::Allocation)?;
        all_placements.extend_from_slice(&self.placements);
        all_placements.extend_from_slice(bindings);
        let mut result = Ok(());
        let mut applied = 0usize;
        visit_link_operations(elf, |operation| {
            if result.is_err() || !operation.jit {
                return;
            }
            let Some(target) = self
                .sections
                .iter()
                .find(|section| section.section == operation.target_section)
            else {
                result = Err(LoadError::InvalidSection);
                return;
            };
            let symbol = match resolve_symbol(
                elf,
                operation.symbol_table,
                operation.symbol,
                &all_placements,
                &NPU40XX_RUNTIME_SYMBOLS,
            ) {
                Ok(symbol) => symbol,
                Err(error) => {
                    result = Err(LoadError::Link(error));
                    return;
                }
            };
            let mapping = match target.bo.map(true) {
                Ok(mapping) => mapping,
                Err(error) => {
                    result = Err(LoadError::Runtime(error));
                    return;
                }
            };
            let logical = match usize::try_from(target.logical_bytes) {
                Ok(logical) if logical <= mapping.len() => logical,
                _ => {
                    result = Err(LoadError::InvalidSection);
                    return;
                }
            };
            // SAFETY: one live, bounds-checked mapping for the relocation target.
            let target_bytes =
                unsafe { core::slice::from_raw_parts_mut(mapping.address().as_ptr(), logical) };
            if let Err(error) = apply_link_operation(target_bytes, operation, symbol) {
                result = Err(LoadError::Link(error));
                return;
            }
            release_host_writes();
            applied += 1;
        })
        .map_err(LoadError::Link)?;
        result?;
        if applied != self.jit_relocations {
            return Err(LoadError::InvalidSection);
        }
        self.linked_digest = linked_digest(&self.sections)?;
        self.jit_relocations = 0;
        Ok(())
    }
}

pub fn placement_digest(elf: &ParsedIntelElf<'_>) -> Result<SemanticDigest, LoadError> {
    let mut hasher = SemanticHasher::new(b"direct-ivpu/placement-plan");
    visit_section_placements(elf, |placement| {
        hasher
            .u64(b"section", placement.section as u64)
            .u32(b"class", placement.class as u32)
            .u64(b"bytes", placement.bytes)
            .u64(b"allocation-bytes", placement.allocation_bytes)
            .u64(b"alignment", placement.alignment)
            .u32(b"bo-flags", placement.bo_flags);
        if let Some(data) = placement.initial_data {
            hasher.field(b"initial-data", data);
        }
    });
    visit_link_operations(elf, |operation| {
        hasher
            .u64(
                b"relocation-target-section",
                operation.target_section as u64,
            )
            .u64(b"relocation-target-offset", operation.target_offset)
            .u64(b"relocation-symbol-table", operation.symbol_table as u64)
            .u64(b"relocation-symbol", operation.symbol as u64)
            .u32(b"relocation-type", operation.relocation_type)
            .i64(b"relocation-addend", operation.addend)
            .u8(b"relocation-write-bytes", operation.write_bytes)
            .u8(b"relocation-jit", u8::from(operation.jit));
    })
    .map_err(LoadError::Link)?;
    Ok(hasher.finish())
}

fn collect_placements<'a>(
    elf: &ParsedIntelElf<'a>,
) -> Result<Vec<SectionPlacement<'a>>, LoadError> {
    let mut count = 0usize;
    visit_section_placements(elf, |_| count += 1);
    if count > MAX_STAGED_SECTIONS {
        return Err(LoadError::TooManySections);
    }
    let mut placements = Vec::new();
    placements
        .try_reserve_exact(count)
        .map_err(|_| LoadError::Allocation)?;
    visit_section_placements(elf, |placement| placements.push(placement));
    Ok(placements)
}

fn stage_bytes(section: &StagedSection<'_>, data: Option<&[u8]>) -> Result<(), LoadError> {
    let mapping = section.bo.map(true).map_err(LoadError::Runtime)?;
    let logical = usize::try_from(section.logical_bytes).map_err(|_| LoadError::InvalidSection)?;
    if logical > mapping.len() || data.is_some_and(|data| data.len() > logical) {
        return Err(LoadError::InvalidSection);
    }
    // SAFETY: this mapping exclusively belongs to `section.bo`; bounds were checked against the
    // returned mapping length and no reference to it is retained after mapping drops.
    unsafe {
        ptr::write_bytes(mapping.address().as_ptr(), 0, logical);
        if let Some(data) = data {
            ptr::copy_nonoverlapping(data.as_ptr(), mapping.address().as_ptr(), data.len());
        }
    }
    release_host_writes();
    Ok(())
}

fn apply_static_relocations(
    elf: &ParsedIntelElf<'_>,
    sections: &[StagedSection<'_>],
    placements: &[PlacedSection],
) -> Result<usize, LoadError> {
    let mut result = Ok(());
    let mut jit = 0usize;
    visit_link_operations(elf, |operation: LinkOperation| {
        if result.is_err() {
            return;
        }
        if operation.jit {
            jit += 1;
            return;
        }
        let Some(target) = sections
            .iter()
            .find(|section| section.section == operation.target_section)
        else {
            result = Err(LoadError::InvalidSection);
            return;
        };
        let symbol = match resolve_symbol(
            elf,
            operation.symbol_table,
            operation.symbol,
            placements,
            &NPU40XX_RUNTIME_SYMBOLS,
        ) {
            Ok(symbol) => symbol,
            Err(error) => {
                result = Err(LoadError::Link(error));
                return;
            }
        };
        let mapping = match target.bo.map(true) {
            Ok(mapping) => mapping,
            Err(error) => {
                result = Err(LoadError::Runtime(error));
                return;
            }
        };
        let logical = match usize::try_from(target.logical_bytes) {
            Ok(logical) if logical <= mapping.len() => logical,
            _ => {
                result = Err(LoadError::InvalidSection);
                return;
            }
        };
        // SAFETY: one live mapping for this BO, constrained to its validated logical section size.
        let target_bytes =
            unsafe { core::slice::from_raw_parts_mut(mapping.address().as_ptr(), logical) };
        if let Err(error) = apply_link_operation(target_bytes, operation, symbol) {
            result = Err(LoadError::Link(error));
            return;
        }
        release_host_writes();
    })
    .map_err(LoadError::Link)?;
    result.map(|_| jit)
}

fn linked_digest(sections: &[StagedSection<'_>]) -> Result<SemanticDigest, LoadError> {
    let mut hasher = SemanticHasher::new(b"direct-ivpu/linked-sections");
    for section in sections {
        let mapping = section.bo.map(false).map_err(LoadError::Runtime)?;
        let logical =
            usize::try_from(section.logical_bytes).map_err(|_| LoadError::InvalidSection)?;
        if logical > mapping.len() {
            return Err(LoadError::InvalidSection);
        }
        acquire_device_writes();
        // SAFETY: read-only hashing inside the live, bounds-checked mapping.
        let bytes = unsafe { core::slice::from_raw_parts(mapping.address().as_ptr(), logical) };
        hasher
            .u64(b"section", section.section as u64)
            .field(b"linked-bytes", bytes);
    }
    Ok(hasher.finish())
}

pub fn stage_program<'device>(
    device: &'device IvpuDevice,
    elf: &ParsedIntelElf<'_>,
    hpi: &HpiRequirements,
    max_user_bo_bytes: u64,
) -> Result<StagedProgram<'device>, LoadError> {
    let plans = collect_placements(elf)?;
    let mut sections = Vec::new();
    let mut placements = Vec::new();
    sections
        .try_reserve_exact(plans.len())
        .map_err(|_| LoadError::Allocation)?;
    placements
        .try_reserve_exact(plans.len())
        .map_err(|_| LoadError::Allocation)?;
    let mut total_bo_bytes = 0u64;

    for plan in plans {
        if plan.alignment > rustix::param::page_size() as u64 || !plan.alignment.is_power_of_two() {
            return Err(LoadError::InvalidAlignment);
        }
        let bo = device
            .create_bo(plan.allocation_bytes, plan.bo_flags)
            .map_err(LoadError::Runtime)?;
        let info = bo.info().map_err(LoadError::Runtime)?;
        if info.vpu_address % plan.alignment != 0 || info.bytes < plan.allocation_bytes {
            return Err(LoadError::InvalidAlignment);
        }
        total_bo_bytes = total_bo_bytes
            .checked_add(info.bytes)
            .ok_or(LoadError::BudgetExceeded)?;
        if total_bo_bytes > max_user_bo_bytes {
            return Err(LoadError::BudgetExceeded);
        }
        let section = StagedSection {
            section: plan.section,
            class: plan.class,
            logical_bytes: plan.bytes,
            bo,
        };
        stage_bytes(&section, plan.initial_data)?;
        placements.push(PlacedSection {
            section: plan.section,
            device_address: info.vpu_address,
            bytes: plan.bytes,
        });
        sections.push(section);
    }

    let jit_relocations = apply_static_relocations(elf, &sections, &placements)?;
    let entry_address = placements
        .iter()
        .find(|placement| placement.section == hpi.entry_section)
        .map(|placement| placement.device_address)
        .ok_or(LoadError::InvalidSection)?;
    let plan_digest = placement_digest(elf)?;
    let linked_digest = linked_digest(&sections)?;
    Ok(StagedProgram {
        sections,
        placements,
        total_bo_bytes,
        entry_address,
        jit_relocations,
        plan_digest,
        linked_digest,
    })
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "bench-hardware")]
    use super::super::elf::parse_intel_elf;
    #[cfg(feature = "bench-hardware")]
    use super::super::hpi::inspect_hpi_requirements;
    use super::*;

    #[test]
    fn staging_section_count_is_bounded() {
        assert!(MAX_STAGED_SECTIONS <= 64);
    }

    #[test]
    #[cfg(feature = "bench-hardware")]
    #[ignore = "bounded section staging and static linking only; no queue or submit"]
    fn stage_existing_tiny_intel_artifact() {
        const MAX_ARTIFACT: usize = 1024 * 1024;
        let bytes = super::super::load_test_artifact(MAX_ARTIFACT);
        let elf = parse_intel_elf(&bytes).unwrap();
        let device = IvpuDevice::open_first().unwrap();
        let parameters = device.parameters().unwrap();
        let hpi = inspect_hpi_requirements(&elf, &parameters).unwrap();
        let staged = stage_program(&device, &elf, &hpi, 64 * 1024 * 1024).unwrap();
        std::println!(
            "artifact={} sections={} total_bo_bytes={} entry_address={:#x} entry_bytes={} jit_relocations={} plan_digest={} linked_digest={}",
            super::super::test_artifact_location().to_string_lossy(),
            staged.sections().len(),
            staged.total_bo_bytes,
            staged.entry_address,
            hpi.entry_bytes,
            staged.jit_relocations,
            staged.plan_digest.hex(),
            staged.linked_digest.hex(),
        );
        assert!(staged.total_bo_bytes <= 64 * 1024 * 1024);
        assert_eq!(staged.sections().len(), 17);
        assert_eq!(hpi.entry_bytes, 0x980);
        assert_eq!(staged.jit_relocations, 3);
        assert_eq!(staged.plan_digest, placement_digest(&elf).unwrap());
        staged.destroy();
        drop(device);
    }
}
