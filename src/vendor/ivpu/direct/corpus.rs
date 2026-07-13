//! Model-corpus Direct ↔ Level Zero parity helpers (#63 corpus gate).
//!
//! Hermetic path: parse Intel ELF + HPI metadata and compare against manifest goldens
//! (bindings, relocations, loader/NNRT pins, resident/placement accounting).
//! Lab path: optional Direct vs L0 numeric execute for FP32-eligible entries.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use super::elf::{parse_intel_elf, ElfError};
use super::hpi::{inspect_hpi_requirements, HpiError};
use super::link::{visit_binding_requirements, visit_link_operations, visit_section_placements};
use super::link::{DeviceMemoryClass, LinkError};
use super::oracle::{StructuralFingerprint, StructuralMismatch};
use super::DeviceParameters;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorpusError {
    Elf(ElfError),
    Hpi(HpiError),
    Link(LinkError),
    Manifest,
    MissingArtifact,
    HashMismatch,
    SizeMismatch,
}

impl From<ElfError> for CorpusError {
    fn from(value: ElfError) -> Self {
        Self::Elf(value)
    }
}

impl From<HpiError> for CorpusError {
    fn from(value: HpiError) -> Self {
        Self::Hpi(value)
    }
}

impl From<LinkError> for CorpusError {
    fn from(value: LinkError) -> Self {
        Self::Link(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorpusMode {
    /// Structural Direct ELF/HPI golden only (no device, no L0).
    Structural,
    /// Structural + Direct↔L0 FP32 numeric execute when lab hardware is available.
    NumericFp32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorpusEntry {
    pub id: String,
    pub artifact: String,
    pub bytes: u64,
    pub sha256: String,
    pub family: String,
    pub mode: CorpusMode,
    pub notes: String,
    pub expected: StructuralFingerprint,
}

fn split_csv_line(line: &str) -> Vec<&str> {
    line.split(',').map(str::trim).collect()
}

fn parse_u64_list(raw: &str) -> Result<Vec<u64>, CorpusError> {
    if raw.is_empty() || raw == "-" {
        return Ok(Vec::new());
    }
    raw.split('|')
        .map(|part| part.parse::<u64>().map_err(|_| CorpusError::Manifest))
        .collect()
}

fn parse_mode(raw: &str) -> Result<CorpusMode, CorpusError> {
    match raw {
        "structural" => Ok(CorpusMode::Structural),
        "numeric_fp32" => Ok(CorpusMode::NumericFp32),
        _ => Err(CorpusError::Manifest),
    }
}

/// Parse the checked-in Direct↔L0 corpus manifest (CSV with `|`-separated byte lists).
pub fn parse_manifest(csv: &str) -> Result<Vec<CorpusEntry>, CorpusError> {
    let mut entries = Vec::new();
    let mut header_seen = false;
    for raw_line in csv.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols = split_csv_line(line);
        if !header_seen {
            header_seen = true;
            continue;
        }
        if cols.len() < 30 {
            return Err(CorpusError::Manifest);
        }
        entries.push(CorpusEntry {
            id: cols[0].into(),
            artifact: cols[1].into(),
            bytes: cols[2].parse().map_err(|_| CorpusError::Manifest)?,
            sha256: cols[3].into(),
            family: cols[4].into(),
            mode: parse_mode(cols[5])?,
            notes: cols[6].into(),
            expected: StructuralFingerprint {
                inputs: cols[7].parse().map_err(|_| CorpusError::Manifest)?,
                outputs: cols[8].parse().map_err(|_| CorpusError::Manifest)?,
                input_bytes: parse_u64_list(cols[9])?,
                output_bytes: parse_u64_list(cols[10])?,
                input_allocation_bytes: parse_u64_list(cols[11])?,
                output_allocation_bytes: parse_u64_list(cols[12])?,
                sections: cols[13].parse().map_err(|_| CorpusError::Manifest)?,
                symbols: cols[14].parse().map_err(|_| CorpusError::Manifest)?,
                relocations: cols[15].parse().map_err(|_| CorpusError::Manifest)?,
                resident_bytes: cols[16].parse().map_err(|_| CorpusError::Manifest)?,
                placements: cols[17].parse().map_err(|_| CorpusError::Manifest)?,
                placement_bytes: cols[18].parse().map_err(|_| CorpusError::Manifest)?,
                link_operations: cols[19].parse().map_err(|_| CorpusError::Manifest)?,
                jit_operations: cols[20].parse().map_err(|_| CorpusError::Manifest)?,
                loader_major: cols[21].parse().map_err(|_| CorpusError::Manifest)?,
                loader_minor: cols[22].parse().map_err(|_| CorpusError::Manifest)?,
                loader_patch: cols[23].parse().map_err(|_| CorpusError::Manifest)?,
                nnrt_major: cols[24].parse().map_err(|_| CorpusError::Manifest)?,
                nnrt_minor: cols[25].parse().map_err(|_| CorpusError::Manifest)?,
                nnrt_patch: cols[26].parse().map_err(|_| CorpusError::Manifest)?,
                cmx_slice_bytes: cols[27].parse().map_err(|_| CorpusError::Manifest)?,
                slices: cols[28].parse().map_err(|_| CorpusError::Manifest)?,
                barriers: cols[29].parse().map_err(|_| CorpusError::Manifest)?,
            },
        });
    }
    Ok(entries)
}

fn synthetic_device_parameters() -> DeviceParameters {
    let mut firmware_api = [0u64; 8];
    firmware_api[super::firmware::JOB_CMD_API_INDEX] =
        super::firmware::JOB_CMD_API_WORD as u64;
    firmware_api[super::firmware::NNRT_API_INDEX] = super::firmware::NNRT_API_WORD as u64;
    DeviceParameters {
        device_id: 0x643e,
        revision: 4,
        platform_type: 0,
        core_clock_rate: 1_950_000_000,
        context_id: 1,
        context_base_address: 0x1_0000_0000,
        firmware_api,
        tile_config: 0x20,
        sku: 0,
        capability_bits: 0x1e,
        max_contexts: 64,
        preempt_buffer_bytes: 28_389_376,
    }
}

/// Inspect a raw Intel native blob into a structural fingerprint (no device open).
pub fn fingerprint_blob(bytes: &[u8]) -> Result<StructuralFingerprint, CorpusError> {
    let parsed = parse_intel_elf(bytes)?;
    let device = synthetic_device_parameters();
    let hpi = inspect_hpi_requirements(&parsed, &device)?;

    let mut input_bytes = Vec::new();
    let mut output_bytes = Vec::new();
    let mut input_allocation_bytes = Vec::new();
    let mut output_allocation_bytes = Vec::new();
    visit_binding_requirements(&parsed, |binding| match binding.class {
        DeviceMemoryClass::Input => {
            input_bytes.push(binding.bytes);
            input_allocation_bytes.push(binding.allocation_bytes);
        }
        DeviceMemoryClass::Output => {
            output_bytes.push(binding.bytes);
            output_allocation_bytes.push(binding.allocation_bytes);
        }
        _ => {}
    });

    let mut link_operations = 0u64;
    let mut jit_operations = 0u64;
    visit_link_operations(&parsed, |operation| {
        link_operations += 1;
        jit_operations += u64::from(operation.jit);
    })?;

    let mut placements = 0u32;
    let mut placement_bytes = 0u64;
    visit_section_placements(&parsed, |placement| {
        placements = placements.saturating_add(1);
        placement_bytes = placement_bytes.saturating_add(placement.bytes);
    });

    Ok(StructuralFingerprint {
        inputs: u32::from(hpi.resources.inputs),
        outputs: u32::from(hpi.resources.outputs),
        input_bytes,
        output_bytes,
        input_allocation_bytes,
        output_allocation_bytes,
        sections: parsed.inspection().section_headers,
        symbols: parsed.inspection().symbol_count,
        relocations: parsed.inspection().relocation_count,
        resident_bytes: parsed.inspection().resident_bytes,
        placements,
        placement_bytes,
        link_operations,
        jit_operations,
        loader_major: hpi.versions.loader_major,
        loader_minor: hpi.versions.loader_minor,
        loader_patch: hpi.versions.loader_patch,
        nnrt_major: hpi.versions.mapped_inference_major,
        nnrt_minor: hpi.versions.mapped_inference_minor,
        nnrt_patch: hpi.versions.mapped_inference_patch,
        cmx_slice_bytes: hpi.resources.cmx_slice_bytes,
        slices: hpi.resources.slice_count,
        barriers: hpi.resources.barriers,
    })
}

pub fn verify_entry_fingerprint(
    entry: &CorpusEntry,
    bytes: &[u8],
) -> Result<(), StructuralMismatch> {
    let actual = fingerprint_blob(bytes).expect("fingerprint must succeed for corpus blobs");
    super::oracle::structural_matches(&actual, &entry.expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::oracle::{binding_meta_matches, compare_f32, BindingMeta};

    const MANIFEST: &str = include_str!("../../../../tests/direct_l0_corpus/manifest.csv");

    #[test]
    fn manifest_parses_and_lists_representative_entries() {
        let entries = parse_manifest(MANIFEST).expect("manifest");
        assert!(entries.len() >= 4);
        assert!(entries.iter().any(|e| e.id == "vector-add-tiny"));
        assert!(entries.iter().any(|e| e.mode == CorpusMode::NumericFp32));
        assert!(entries.iter().any(|e| e.family == "LFM"));
        let tiny = entries.iter().find(|e| e.id == "vector-add-tiny").unwrap();
        assert_eq!(tiny.expected.inputs, 2);
        assert_eq!(tiny.expected.outputs, 1);
        assert_eq!(tiny.expected.input_bytes, alloc::vec![128, 128]);
        assert_eq!(tiny.expected.output_bytes, alloc::vec![128]);
    }

    #[test]
    fn structural_and_binding_helpers_agree_on_synthetic() {
        let expected = StructuralFingerprint {
            inputs: 2,
            outputs: 1,
            input_bytes: alloc::vec![128, 128],
            output_bytes: alloc::vec![128],
            input_allocation_bytes: alloc::vec![4096, 4096],
            output_allocation_bytes: alloc::vec![4096],
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
        let mut actual = expected.clone();
        assert_eq!(
            super::super::oracle::structural_matches(&actual, &expected),
            Ok(())
        );
        actual.relocations = 41;
        assert_eq!(
            super::super::oracle::structural_matches(&actual, &expected),
            Err(StructuralMismatch::Relocations)
        );
        assert_eq!(
            binding_meta_matches(expected.binding_meta(), expected.binding_meta()),
            Ok(())
        );
        let diff = compare_f32(&[1.0, 2.0], &[1.0, 2.0], 1e-5, 1e-5);
        assert!(diff.within(1e-5, 1e-5));
        let _ = BindingMeta {
            inputs: 0,
            outputs: 0,
            input_bytes: &[],
            output_bytes: &[],
        };
    }

    #[test]
    #[cfg(feature = "direct-ivpu")]
    fn corpus_structural_goldens_match_when_artifacts_present() {
        let entries = parse_manifest(MANIFEST).expect("manifest");
        let mut checked = 0usize;
        let mut missing = 0usize;
        for entry in &entries {
            let Some(path) = resolve_artifact(&entry.artifact) else {
                missing += 1;
                continue;
            };
            let bytes = std::fs::read(&path).expect("read artifact");
            assert_eq!(
                bytes.len() as u64,
                entry.bytes,
                "{} size mismatch at {}",
                entry.id,
                path.display()
            );
            let digest = sha256_hex(&bytes);
            assert_eq!(
                digest, entry.sha256,
                "{} sha256 mismatch at {}",
                entry.id,
                path.display()
            );
            let actual = fingerprint_blob(&bytes).expect("fingerprint");
            assert_eq!(
                super::super::oracle::structural_matches(&actual, &entry.expected),
                Ok(()),
                "{} structural mismatch actual={actual:?} expected={:?}",
                entry.id,
                entry.expected
            );
            checked += 1;
        }
        std::println!("corpus structural checked={checked} missing={missing}");
        // Hermetic default: pass even when lab blobs are absent. Lab script requires all present.
        let _ = missing;
        if std::env::var_os("GGML_NPU_CORPUS_REQUIRE_ALL").is_some() {
            assert_eq!(
                missing, 0,
                "GGML_NPU_CORPUS_REQUIRE_ALL set but {missing} corpus blobs missing"
            );
            assert_eq!(checked, entries.len());
        }
    }

    #[test]
    #[cfg(all(feature = "bench-hardware", feature = "level-zero-compat"))]
    #[ignore = "lab Direct↔L0 corpus binding + numeric parity; requires NPU + Level Zero + corpus blobs"]
    fn corpus_direct_level_zero_parity() {
        use alloc::vec;
        use alloc::vec::Vec;
        use crate::vendor::ivpu::graph;
        use crate::vendor::ivpu::{ze, HardwareBenchmarkFixture};
        use super::super::oracle::{binding_meta_matches, compare_f32, BindingMeta};

        const ABS_TOL: f32 = 1.0e-5;
        const REL_TOL: f32 = 1.0e-5;

        let entries = parse_manifest(MANIFEST).expect("manifest");
        assert!(!entries.is_empty());
        let level_zero = ze::probe();
        assert_eq!(level_zero.len(), 1, "expected one Level Zero NPU device");

        let mut passed = 0usize;
        for entry in &entries {
            let path = resolve_artifact(&entry.artifact).unwrap_or_else(|| {
                panic!(
                    "missing corpus artifact {} (set GGML_NPU_CORPUS_DIR)",
                    entry.artifact
                )
            });
            let bytes = std::fs::read(&path).expect("read artifact");
            assert_eq!(bytes.len() as u64, entry.bytes, "{} size", entry.id);
            assert_eq!(sha256_hex(&bytes), entry.sha256, "{} hash", entry.id);

            let direct_fp = fingerprint_blob(&bytes).expect("direct fingerprint");
            assert_eq!(
                super::super::oracle::structural_matches(&direct_fp, &entry.expected),
                Ok(()),
                "{} direct structural golden",
                entry.id
            );

            let (l0_inputs, l0_outputs) =
                graph::native_blob_binding_bytes(&level_zero[0], &bytes).unwrap_or_else(|code| {
                    panic!("{} Level Zero binding probe failed: {code}", entry.id)
                });
            assert_eq!(
                binding_meta_matches(
                    BindingMeta {
                        inputs: direct_fp.inputs,
                        outputs: direct_fp.outputs,
                        input_bytes: &direct_fp.input_bytes,
                        output_bytes: &direct_fp.output_bytes,
                    },
                    BindingMeta {
                        inputs: l0_inputs.len() as u32,
                        outputs: l0_outputs.len() as u32,
                        input_bytes: &l0_inputs,
                        output_bytes: &l0_outputs,
                    }
                ),
                Ok(()),
                "{} Direct↔L0 binding meta mismatch direct_in={:?} l0_in={:?} direct_out={:?} l0_out={:?}",
                entry.id,
                direct_fp.input_bytes,
                l0_inputs,
                direct_fp.output_bytes,
                l0_outputs
            );

            if entry.mode == CorpusMode::NumericFp32 {
                let input_lens: Vec<usize> = direct_fp
                    .input_bytes
                    .iter()
                    .map(|&b| {
                        assert_eq!(b % 4, 0, "{} input bytes not FP32-aligned", entry.id);
                        (b / 4) as usize
                    })
                    .collect();
                let output_elems = (direct_fp.output_bytes[0] / 4) as usize;
                let mut owned_inputs: Vec<Vec<f32>> = input_lens
                    .iter()
                    .enumerate()
                    .map(|(input_index, &len)| {
                        (0..len)
                            .map(|i| ((i * 7 + input_index * 13) % 29) as f32 * 0.125 - 1.5)
                            .collect()
                    })
                    .collect();
                let input_refs: Vec<&[f32]> =
                    owned_inputs.iter().map(|input| input.as_slice()).collect();

                let mut l0_out = vec![0.0f32; output_elems];
                let written = graph::run_native_blob(
                    &level_zero[0],
                    &bytes,
                    &input_refs,
                    &mut l0_out,
                )
                .expect("L0 execute");
                assert_eq!(written, output_elems);

                let mut direct = HardwareBenchmarkFixture::new(0, &bytes, &input_refs)
                    .expect("Direct fixture");
                direct.execute().expect("Direct execute");
                let mut direct_out = vec![0.0f32; output_elems];
                let copied = direct.copy_output(&mut direct_out).expect("Direct copy");
                assert_eq!(copied, output_elems);

                let diff = compare_f32(&direct_out, &l0_out, ABS_TOL, REL_TOL);
                assert!(
                    diff.within(ABS_TOL, REL_TOL),
                    "{} numeric Direct↔L0 mismatch: {diff:?}",
                    entry.id
                );
                std::println!(
                    "PASS {} numeric max_abs={} max_rel={}",
                    entry.id, diff.max_absolute, diff.max_relative
                );
            } else {
                std::println!("PASS {} structural+binding", entry.id);
            }
            passed += 1;
        }
        std::println!("corpus Direct↔L0 parity passed={passed}/{}", entries.len());
        assert_eq!(passed, entries.len());
    }

    #[test]
    #[cfg(feature = "direct-ivpu")]
    fn dump_fingerprints_for_lab_fill() {
        // Opt-in helper for refreshing goldens: GGML_NPU_CORPUS_DUMP=1 cargo test ...
        if std::env::var_os("GGML_NPU_CORPUS_DUMP").is_none() {
            return;
        }
        let names = [
            "ir2-8c91da4914adc7d5-n1.blob",
            "ir2-d152331ee29c46c3-n1.blob",
            "ir2-shortconv-e2e-v1.blob",
            "ir2-batched-get-rows-e2e-v1.blob",
            "ir2-2fc11a4f22b54cfc-n3.blob",
            "ir2-1ee5f0b4434437e8-n8.blob",
        ];
        for name in names {
            let Some(path) = resolve_artifact(name) else {
                std::println!("MISSING {name}");
                continue;
            };
            let bytes = std::fs::read(&path).unwrap();
            let fp = fingerprint_blob(&bytes).unwrap();
            let digest = sha256_hex(&bytes);
            let fmt = |v: &[u64]| {
                if v.is_empty() {
                    "-".into()
                } else {
                    v.iter()
                        .map(|x| alloc::format!("{x}"))
                        .collect::<Vec<_>>()
                        .join("|")
                }
            };
            std::println!(
                "{name},{},{digest},inputs={} outputs={} in={} out={} in_alloc={} out_alloc={} sec={} sym={} reloc={} resid={} place={} place_b={} link={} jit={} loader={}.{}.{} nnrt={}.{}.{} cmx={} slices={} barriers={}",
                bytes.len(),
                fp.inputs,
                fp.outputs,
                fmt(&fp.input_bytes),
                fmt(&fp.output_bytes),
                fmt(&fp.input_allocation_bytes),
                fmt(&fp.output_allocation_bytes),
                fp.sections,
                fp.symbols,
                fp.relocations,
                fp.resident_bytes,
                fp.placements,
                fp.placement_bytes,
                fp.link_operations,
                fp.jit_operations,
                fp.loader_major,
                fp.loader_minor,
                fp.loader_patch,
                fp.nnrt_major,
                fp.nnrt_minor,
                fp.nnrt_patch,
                fp.cmx_slice_bytes,
                fp.slices,
                fp.barriers,
            );
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        sha256_hex_impl(bytes)
    }

    fn sha256_hex_impl(bytes: &[u8]) -> String {
        // Compact SHA-256 (FIPS 180-4) for corpus identity checks — no new crate dependency.
        fn rotr(x: u32, n: u32) -> u32 {
            (x >> n) | (x << (32 - n))
        }
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let bit_len = (bytes.len() as u64).saturating_mul(8);
        let mut msg = bytes.to_vec();
        msg.push(0x80);
        while (msg.len() % 64) != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&bit_len.to_be_bytes());
        for chunk in msg.chunks_exact(64) {
            let mut w = [0u32; 64];
            for i in 0..16 {
                w[i] = u32::from_be_bytes([
                    chunk[i * 4],
                    chunk[i * 4 + 1],
                    chunk[i * 4 + 2],
                    chunk[i * 4 + 3],
                ]);
            }
            for i in 16..64 {
                let s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                let s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let mut a = h[0];
            let mut b = h[1];
            let mut c = h[2];
            let mut d = h[3];
            let mut e = h[4];
            let mut f = h[5];
            let mut g = h[6];
            let mut hh = h[7];
            for i in 0..64 {
                let s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
                let ch = (e & f) ^ ((!e) & g);
                let temp1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);
                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }
            h[0] = h[0].wrapping_add(a);
            h[1] = h[1].wrapping_add(b);
            h[2] = h[2].wrapping_add(c);
            h[3] = h[3].wrapping_add(d);
            h[4] = h[4].wrapping_add(e);
            h[5] = h[5].wrapping_add(f);
            h[6] = h[6].wrapping_add(g);
            h[7] = h[7].wrapping_add(hh);
        }
        let mut out = String::with_capacity(64);
        for word in h {
            use core::fmt::Write;
            let _ = write!(out, "{word:08x}");
        }
        out
    }

    fn resolve_artifact(name: &str) -> Option<std::path::PathBuf> {
        use std::path::PathBuf;
        let mut candidates = Vec::new();
        if let Ok(root) = std::env::var("GGML_NPU_CORPUS_DIR") {
            candidates.push(PathBuf::from(root).join(name));
        }
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(PathBuf::from(home).join(".cache/ggml-npu").join(name));
        }
        candidates.push(PathBuf::from("benchmarks/results/lfm-pp512-cache").join(name));
        // From `.worktrees/<name>/` back to the primary checkout's gitignored LFM cache.
        candidates.push(PathBuf::from("../../benchmarks/results/lfm-pp512-cache").join(name));
        candidates.push(PathBuf::from("artifacts").join(name));
        candidates.into_iter().find(|path| path.is_file())
    }
}
