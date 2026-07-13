//! Hermetic differential-oracle hooks for Direct vs Level Zero (or CPU) comparisons.
//!
//! These helpers do not open a device. Hardware and `level-zero-compat` tests feed Direct and
//! oracle buffers through the same tolerance and binding-metadata checks.

#![allow(dead_code)]

/// Numeric comparison result for float buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericDiff {
    pub max_absolute: f32,
    pub max_relative: f32,
    pub first_mismatch: Option<usize>,
    pub compared: usize,
}

/// Binding ABI metadata used for Direct ↔ oracle parity checks before numerics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingMeta<'a> {
    pub inputs: u32,
    pub outputs: u32,
    pub input_bytes: &'a [u64],
    pub output_bytes: &'a [u64],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingMetaMismatch {
    InputCount,
    OutputCount,
    InputBytes { index: usize },
    OutputBytes { index: usize },
}

impl NumericDiff {
    pub const fn within(self, abs_tol: f32, rel_tol: f32) -> bool {
        self.first_mismatch.is_none()
            && self.max_absolute <= abs_tol
            && self.max_relative <= rel_tol
    }
}

/// Compare two f32 buffers element-wise with absolute and relative tolerances.
///
/// Relative error uses `|a-e| / max(|e|, 1e-12)` so exact zeros remain absolute-only.
pub fn compare_f32(actual: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32) -> NumericDiff {
    let compared = core::cmp::min(actual.len(), expected.len());
    let mut max_absolute = 0.0f32;
    let mut max_relative = 0.0f32;
    let mut first_mismatch = None;
    if actual.len() != expected.len() {
        first_mismatch = Some(compared);
    }
    for index in 0..compared {
        let a = actual[index];
        let e = expected[index];
        let absolute = (a - e).abs();
        let relative = absolute / e.abs().max(1e-12);
        max_absolute = max_absolute.max(absolute);
        max_relative = max_relative.max(relative);
        if first_mismatch.is_none() && (absolute > abs_tol && relative > rel_tol) {
            first_mismatch = Some(index);
        }
    }
    NumericDiff {
        max_absolute,
        max_relative,
        first_mismatch,
        compared,
    }
}

pub fn binding_meta_matches(
    actual: BindingMeta<'_>,
    expected: BindingMeta<'_>,
) -> Result<(), BindingMetaMismatch> {
    if actual.inputs != expected.inputs || actual.input_bytes.len() != expected.input_bytes.len() {
        return Err(BindingMetaMismatch::InputCount);
    }
    if actual.outputs != expected.outputs
        || actual.output_bytes.len() != expected.output_bytes.len()
    {
        return Err(BindingMetaMismatch::OutputCount);
    }
    for (index, (a, e)) in actual
        .input_bytes
        .iter()
        .zip(expected.input_bytes.iter())
        .enumerate()
    {
        if a != e {
            return Err(BindingMetaMismatch::InputBytes { index });
        }
    }
    for (index, (a, e)) in actual
        .output_bytes
        .iter()
        .zip(expected.output_bytes.iter())
        .enumerate()
    {
        if a != e {
            return Err(BindingMetaMismatch::OutputBytes { index });
        }
    }
    Ok(())
}

/// Structural Direct ELF / HPI fingerprint used for corpus goldens (no device required).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralFingerprint {
    pub inputs: u32,
    pub outputs: u32,
    pub input_bytes: alloc::vec::Vec<u64>,
    pub output_bytes: alloc::vec::Vec<u64>,
    pub input_allocation_bytes: alloc::vec::Vec<u64>,
    pub output_allocation_bytes: alloc::vec::Vec<u64>,
    pub sections: u16,
    pub symbols: u64,
    pub relocations: u64,
    pub resident_bytes: u64,
    pub placements: u32,
    pub placement_bytes: u64,
    pub link_operations: u64,
    pub jit_operations: u64,
    pub loader_major: u32,
    pub loader_minor: u32,
    pub loader_patch: u32,
    pub nnrt_major: u32,
    pub nnrt_minor: u32,
    pub nnrt_patch: u32,
    pub cmx_slice_bytes: u32,
    pub slices: u8,
    pub barriers: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuralMismatch {
    Inputs,
    Outputs,
    InputBytes { index: usize },
    OutputBytes { index: usize },
    InputAllocationBytes { index: usize },
    OutputAllocationBytes { index: usize },
    Sections,
    Symbols,
    Relocations,
    ResidentBytes,
    Placements,
    PlacementBytes,
    LinkOperations,
    JitOperations,
    LoaderVersion,
    NnrtVersion,
    CmxSliceBytes,
    Slices,
    Barriers,
}

impl StructuralFingerprint {
    pub fn binding_meta(&self) -> BindingMeta<'_> {
        BindingMeta {
            inputs: self.inputs,
            outputs: self.outputs,
            input_bytes: &self.input_bytes,
            output_bytes: &self.output_bytes,
        }
    }
}

pub fn structural_matches(
    actual: &StructuralFingerprint,
    expected: &StructuralFingerprint,
) -> Result<(), StructuralMismatch> {
    if actual.inputs != expected.inputs || actual.input_bytes.len() != expected.input_bytes.len() {
        return Err(StructuralMismatch::Inputs);
    }
    if actual.outputs != expected.outputs
        || actual.output_bytes.len() != expected.output_bytes.len()
    {
        return Err(StructuralMismatch::Outputs);
    }
    for (index, (a, e)) in actual
        .input_bytes
        .iter()
        .zip(expected.input_bytes.iter())
        .enumerate()
    {
        if a != e {
            return Err(StructuralMismatch::InputBytes { index });
        }
    }
    for (index, (a, e)) in actual
        .output_bytes
        .iter()
        .zip(expected.output_bytes.iter())
        .enumerate()
    {
        if a != e {
            return Err(StructuralMismatch::OutputBytes { index });
        }
    }
    for (index, (a, e)) in actual
        .input_allocation_bytes
        .iter()
        .zip(expected.input_allocation_bytes.iter())
        .enumerate()
    {
        if a != e {
            return Err(StructuralMismatch::InputAllocationBytes { index });
        }
    }
    for (index, (a, e)) in actual
        .output_allocation_bytes
        .iter()
        .zip(expected.output_allocation_bytes.iter())
        .enumerate()
    {
        if a != e {
            return Err(StructuralMismatch::OutputAllocationBytes { index });
        }
    }
    if actual.sections != expected.sections {
        return Err(StructuralMismatch::Sections);
    }
    if actual.symbols != expected.symbols {
        return Err(StructuralMismatch::Symbols);
    }
    if actual.relocations != expected.relocations {
        return Err(StructuralMismatch::Relocations);
    }
    if actual.resident_bytes != expected.resident_bytes {
        return Err(StructuralMismatch::ResidentBytes);
    }
    if actual.placements != expected.placements {
        return Err(StructuralMismatch::Placements);
    }
    if actual.placement_bytes != expected.placement_bytes {
        return Err(StructuralMismatch::PlacementBytes);
    }
    if actual.link_operations != expected.link_operations {
        return Err(StructuralMismatch::LinkOperations);
    }
    if actual.jit_operations != expected.jit_operations {
        return Err(StructuralMismatch::JitOperations);
    }
    if (
        actual.loader_major,
        actual.loader_minor,
        actual.loader_patch,
    ) != (
        expected.loader_major,
        expected.loader_minor,
        expected.loader_patch,
    ) {
        return Err(StructuralMismatch::LoaderVersion);
    }
    if (actual.nnrt_major, actual.nnrt_minor, actual.nnrt_patch)
        != (expected.nnrt_major, expected.nnrt_minor, expected.nnrt_patch)
    {
        return Err(StructuralMismatch::NnrtVersion);
    }
    if actual.cmx_slice_bytes != expected.cmx_slice_bytes {
        return Err(StructuralMismatch::CmxSliceBytes);
    }
    if actual.slices != expected.slices {
        return Err(StructuralMismatch::Slices);
    }
    if actual.barriers != expected.barriers {
        return Err(StructuralMismatch::Barriers);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_passes_default_tolerance() {
        let actual = [1.0f32, -2.5, 0.0];
        let expected = actual;
        let diff = compare_f32(&actual, &expected, 1e-5, 1e-5);
        assert!(diff.within(1e-5, 1e-5));
        assert_eq!(diff.first_mismatch, None);
        assert_eq!(diff.compared, 3);
        assert_eq!(diff.max_absolute, 0.0);
    }

    #[test]
    fn length_mismatch_and_tolerance_edges() {
        let actual = [1.0f32, 2.0];
        let expected = [1.0f32];
        let diff = compare_f32(&actual, &expected, 1e-5, 1e-5);
        assert_eq!(diff.first_mismatch, Some(1));
        assert!(!diff.within(1e-5, 1e-5));

        let near = [1.0 + 1e-6];
        let base = [1.0f32];
        let ok = compare_f32(&near, &base, 1e-5, 1e-5);
        assert!(ok.within(1e-5, 1e-5));

        let far = [1.0 + 1e-3];
        let bad = compare_f32(&far, &base, 1e-5, 1e-5);
        assert_eq!(bad.first_mismatch, Some(0));
    }

    #[test]
    fn binding_meta_parity_is_strict() {
        let a = BindingMeta {
            inputs: 2,
            outputs: 1,
            input_bytes: &[128, 128],
            output_bytes: &[128],
        };
        let b = BindingMeta {
            inputs: 2,
            outputs: 1,
            input_bytes: &[128, 128],
            output_bytes: &[128],
        };
        assert_eq!(binding_meta_matches(a, b), Ok(()));
        assert_eq!(
            binding_meta_matches(
                a,
                BindingMeta {
                    input_bytes: &[128, 64],
                    ..b
                }
            ),
            Err(BindingMetaMismatch::InputBytes { index: 1 })
        );
        assert_eq!(
            binding_meta_matches(
                a,
                BindingMeta {
                    outputs: 2,
                    output_bytes: &[128, 128],
                    ..b
                }
            ),
            Err(BindingMetaMismatch::OutputCount)
        );
    }

    #[test]
    fn structural_fingerprint_detects_relocation_drift() {
        let base = StructuralFingerprint {
            inputs: 2,
            outputs: 1,
            input_bytes: alloc::vec![128, 128],
            output_bytes: alloc::vec![128],
            input_allocation_bytes: alloc::vec![128, 128],
            output_allocation_bytes: alloc::vec![128],
            sections: 43,
            symbols: 33,
            relocations: 42,
            resident_bytes: 1_533_728,
            placements: 17,
            placement_bytes: 8_096,
            link_operations: 42,
            jit_operations: 3,
            loader_major: 1,
            loader_minor: 2,
            loader_patch: 2,
            nnrt_major: 11,
            nnrt_minor: 4,
            nnrt_patch: 10,
            cmx_slice_bytes: 1_473_536,
            slices: 4,
            barriers: 0,
        };
        assert_eq!(structural_matches(&base, &base), Ok(()));
        let mut drifted = base.clone();
        drifted.relocations = 41;
        assert_eq!(
            structural_matches(&drifted, &base),
            Err(StructuralMismatch::Relocations)
        );
    }
}
