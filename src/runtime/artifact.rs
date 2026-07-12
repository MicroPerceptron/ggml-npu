//! Pointer-free artifact envelope shared by future compiler adapters and runtime backends.

#![allow(dead_code)]

use core::num::NonZeroU32;

pub const ARTIFACT_MAGIC: [u8; 8] = *b"GGNPUART";
pub const ARTIFACT_ABI_MAJOR: u16 = 1;
pub const MAX_ARTIFACT_BYTES: usize = 512 * 1024 * 1024;
pub const MAX_RESIDENT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Opaque target identity words. The artifact format's adapter owns their schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct TargetIdentity {
    words: [u32; 12],
}

impl TargetIdentity {
    pub const fn new(words: [u32; 12]) -> Self {
        Self { words }
    }

    pub const fn words(self) -> [u32; 12] {
        self.words
    }
}

/// Opaque nonzero artifact format identifier. Concrete constants live in compiler/vendor adapters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct ArtifactFormat(NonZeroU32);

impl ArtifactFormat {
    pub const fn new(value: u32) -> Option<Self> {
        match NonZeroU32::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u32 {
        self.0.get()
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
    pub target: TargetIdentity,
    pub format: ArtifactFormat,
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
        Ok(Self { header, payload })
    }

    pub const fn header(&self) -> &ArtifactHeader {
        &self.header
    }

    pub const fn payload(&self) -> &'a [u8] {
        self.payload
    }

    pub fn ensure_target(&self, target: &TargetIdentity) -> Result<(), CompatibilityError> {
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
            target: TargetIdentity::new([0; 12]),
            format: ArtifactFormat::new(1).unwrap(),
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

    #[test]
    fn opaque_format_and_target_preserve_wire_layout() {
        assert!(ArtifactFormat::new(0).is_none());
        assert_eq!(core::mem::size_of::<ArtifactFormat>(), 4);
        assert_eq!(core::mem::size_of::<TargetIdentity>(), 48);
        assert_eq!(core::mem::align_of::<TargetIdentity>(), 4);
        assert_eq!(core::mem::size_of::<ArtifactHeader>(), 112);
    }
}
