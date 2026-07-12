//! Pointer-free artifact envelope shared by future compiler adapters and runtime backends.

#![allow(dead_code)]

pub const ARTIFACT_MAGIC: [u8; 8] = *b"GGNPUART";
pub const ARTIFACT_ABI_MAJOR: u16 = 1;
pub const MAX_ARTIFACT_BYTES: usize = 512 * 1024 * 1024;
pub const MAX_RESIDENT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct TargetFingerprint {
    pub vendor_id: u32,
    pub device_id: u32,
    pub revision: u32,
    pub architecture: u32,
    pub firmware_api: [u32; 8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct NativePayloadKind(pub u32);

impl NativePayloadKind {
    pub const INTEL_LEVEL_ZERO_GRAPH: Self = Self(1);
    pub const INTEL_IVPU_ELF: Self = Self(2);
    pub const AMD_XCLBIN: Self = Self(3);
    pub const QUALCOMM_CONTEXT_BINARY: Self = Self(4);
    pub const CANONICAL_MLIR_BYTECODE: Self = Self(5);

    pub const fn is_known(self) -> bool {
        self.0 >= Self::INTEL_LEVEL_ZERO_GRAPH.0 && self.0 <= Self::CANONICAL_MLIR_BYTECODE.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct BindingAbi {
    pub major: u16,
    pub minor: u16,
    pub input_count: u16,
    pub output_count: u16,
    pub scratch_count: u16,
    pub reserved: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ArtifactHeader {
    pub magic: [u8; 8],
    pub envelope_major: u16,
    pub envelope_minor: u16,
    pub header_bytes: u32,
    pub module_id: u64,
    pub target: TargetFingerprint,
    pub payload_kind: NativePayloadKind,
    pub binding_abi: BindingAbi,
    pub payload_bytes: u64,
    pub resident_bytes: u64,
    pub flags: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompatibilityError {
    BadMagic,
    UnsupportedEnvelope { major: u16, minor: u16 },
    InvalidHeaderSize,
    InvalidModuleId,
    PayloadTooLarge,
    PayloadLengthMismatch,
    ResidentLimitExceeded,
    TargetMismatch,
    BindingAbiMismatch,
    PayloadKindUnsupported,
}

pub struct ArtifactEnvelope<'a> {
    header: ArtifactHeader,
    payload: &'a [u8],
}

impl<'a> ArtifactEnvelope<'a> {
    pub fn new(header: ArtifactHeader, payload: &'a [u8]) -> Result<Self, CompatibilityError> {
        if header.magic != ARTIFACT_MAGIC {
            return Err(CompatibilityError::BadMagic);
        }
        if header.envelope_major != ARTIFACT_ABI_MAJOR {
            return Err(CompatibilityError::UnsupportedEnvelope {
                major: header.envelope_major,
                minor: header.envelope_minor,
            });
        }
        if header.header_bytes as usize != core::mem::size_of::<ArtifactHeader>() {
            return Err(CompatibilityError::InvalidHeaderSize);
        }
        if header.module_id == 0 {
            return Err(CompatibilityError::InvalidModuleId);
        }
        if payload.len() > MAX_ARTIFACT_BYTES {
            return Err(CompatibilityError::PayloadTooLarge);
        }
        if header.payload_bytes != payload.len() as u64 {
            return Err(CompatibilityError::PayloadLengthMismatch);
        }
        if header.resident_bytes > MAX_RESIDENT_BYTES {
            return Err(CompatibilityError::ResidentLimitExceeded);
        }
        if !header.payload_kind.is_known() {
            return Err(CompatibilityError::PayloadKindUnsupported);
        }
        Ok(Self { header, payload })
    }

    pub const fn header(&self) -> &ArtifactHeader {
        &self.header
    }

    pub const fn payload(&self) -> &'a [u8] {
        self.payload
    }

    pub fn ensure_target(&self, target: &TargetFingerprint) -> Result<(), CompatibilityError> {
        if self.header.target == *target {
            Ok(())
        } else {
            Err(CompatibilityError::TargetMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(payload_bytes: u64) -> ArtifactHeader {
        ArtifactHeader {
            magic: ARTIFACT_MAGIC,
            envelope_major: ARTIFACT_ABI_MAJOR,
            envelope_minor: 0,
            header_bytes: core::mem::size_of::<ArtifactHeader>() as u32,
            module_id: 1,
            target: TargetFingerprint {
                vendor_id: 0x8086,
                device_id: 0x643e,
                revision: 0,
                architecture: 40,
                firmware_api: [0; 8],
            },
            payload_kind: NativePayloadKind::INTEL_IVPU_ELF,
            binding_abi: BindingAbi {
                major: 1,
                minor: 0,
                input_count: 1,
                output_count: 1,
                scratch_count: 0,
                reserved: 0,
            },
            payload_bytes,
            resident_bytes: 4096,
            flags: 0,
        }
    }

    #[test]
    fn envelope_rejects_length_mismatch() {
        assert!(matches!(
            ArtifactEnvelope::new(header(2), &[0]),
            Err(CompatibilityError::PayloadLengthMismatch)
        ));
    }

    #[test]
    fn envelope_is_pointer_free_and_versioned() {
        let envelope = ArtifactEnvelope::new(header(1), &[7]).unwrap();
        assert_eq!(envelope.header().module_id, 1);
        assert_eq!(envelope.payload(), &[7]);
    }
}
