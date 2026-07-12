//! Domain-separated, length-framed semantic identities.

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

pub const IDENTITY_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SemanticDigest(pub [u8; 32]);

impl SemanticDigest {
    pub fn hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 0x0f) as usize] as char);
        }
        output
    }

    /// Non-authoritative short token for diagnostics only.
    pub const fn diagnostic_u64(self) -> u64 {
        u64::from_le_bytes([
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5], self.0[6], self.0[7],
        ])
    }
}

pub struct SemanticHasher {
    hasher: blake3::Hasher,
    metadata: [u8; 1024],
    metadata_len: usize,
}

#[derive(Clone, Debug)]
pub struct ExactSemanticKey {
    digest: SemanticDigest,
    framed: Vec<u8>,
}

impl ExactSemanticKey {
    pub const fn digest(&self) -> SemanticDigest {
        self.digest
    }
}

impl PartialEq for ExactSemanticKey {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest && self.framed == other.framed
    }
}

impl Eq for ExactSemanticKey {}

pub struct ExactKeyBuilder {
    hasher: SemanticHasher,
    framed: Vec<u8>,
}

impl ExactKeyBuilder {
    pub fn new(domain: &[u8]) -> Self {
        let mut framed = Vec::new();
        framed.extend_from_slice(b"ggml-npu-exact-key\0");
        framed.extend_from_slice(&IDENTITY_SCHEMA_VERSION.to_le_bytes());
        framed.extend_from_slice(&(domain.len() as u64).to_le_bytes());
        framed.extend_from_slice(domain);
        Self {
            hasher: SemanticHasher::new(domain),
            framed,
        }
    }

    pub fn field(&mut self, label: &[u8], value: &[u8]) -> &mut Self {
        self.hasher.field(label, value);
        self.framed.push(1);
        self.framed
            .extend_from_slice(&(label.len() as u64).to_le_bytes());
        self.framed.extend_from_slice(label);
        self.framed
            .extend_from_slice(&(value.len() as u64).to_le_bytes());
        self.framed.extend_from_slice(value);
        self
    }

    pub fn u64(&mut self, label: &[u8], value: u64) -> &mut Self {
        self.field(label, &value.to_le_bytes())
    }

    pub fn u8(&mut self, label: &[u8], value: u8) -> &mut Self {
        self.field(label, &[value])
    }

    pub fn u32(&mut self, label: &[u8], value: u32) -> &mut Self {
        self.field(label, &value.to_le_bytes())
    }

    pub fn i64(&mut self, label: &[u8], value: i64) -> &mut Self {
        self.field(label, &value.to_le_bytes())
    }

    pub fn finish(self) -> ExactSemanticKey {
        ExactSemanticKey {
            digest: self.hasher.finish(),
            framed: self.framed,
        }
    }
}

impl SemanticHasher {
    pub fn new(domain: &[u8]) -> Self {
        let mut this = Self {
            hasher: blake3::Hasher::new(),
            metadata: [0; 1024],
            metadata_len: 0,
        };
        this.append_small(b"ggml-npu-semantic-identity\0");
        this.append_small(&IDENTITY_SCHEMA_VERSION.to_le_bytes());
        this.append_small(&(domain.len() as u64).to_le_bytes());
        if domain.len() <= this.metadata.len() - this.metadata_len {
            this.append_small(domain);
        } else {
            this.flush();
            this.hasher.update(domain);
        }
        this
    }

    pub fn field(&mut self, label: &[u8], value: &[u8]) -> &mut Self {
        let framed_len = 1usize
            .saturating_add(8)
            .saturating_add(label.len())
            .saturating_add(8)
            .saturating_add(value.len());
        if framed_len <= self.metadata.len() - self.metadata_len {
            self.append_small(&[1]);
            self.append_small(&(label.len() as u64).to_le_bytes());
            self.append_small(label);
            self.append_small(&(value.len() as u64).to_le_bytes());
            self.append_small(value);
        } else {
            self.flush();
            let mut header = [0u8; 128];
            let header_len = 17usize.saturating_add(label.len());
            if header_len <= header.len() {
                header[0] = 1;
                header[1..9].copy_from_slice(&(label.len() as u64).to_le_bytes());
                header[9..9 + label.len()].copy_from_slice(label);
                header[9 + label.len()..17 + label.len()]
                    .copy_from_slice(&(value.len() as u64).to_le_bytes());
                self.hasher.update(&header[..header_len]);
            } else {
                self.hasher.update(&[1]);
                self.hasher.update(&(label.len() as u64).to_le_bytes());
                self.hasher.update(label);
                self.hasher.update(&(value.len() as u64).to_le_bytes());
            }
            // Large constant/requant payloads stream directly without a framing copy.
            self.hasher.update(value);
        }
        self
    }

    pub fn u8(&mut self, label: &[u8], value: u8) -> &mut Self {
        self.field(label, &[value])
    }

    pub fn u32(&mut self, label: &[u8], value: u32) -> &mut Self {
        self.field(label, &value.to_le_bytes())
    }

    pub fn u64(&mut self, label: &[u8], value: u64) -> &mut Self {
        self.field(label, &value.to_le_bytes())
    }

    pub fn i32(&mut self, label: &[u8], value: i32) -> &mut Self {
        self.field(label, &value.to_le_bytes())
    }

    pub fn i64(&mut self, label: &[u8], value: i64) -> &mut Self {
        self.field(label, &value.to_le_bytes())
    }

    pub fn finish(&self) -> SemanticDigest {
        let mut hasher = self.hasher.clone();
        hasher.update(&self.metadata[..self.metadata_len]);
        SemanticDigest(*hasher.finalize().as_bytes())
    }

    fn append_small(&mut self, bytes: &[u8]) {
        let end = self.metadata_len + bytes.len();
        self.metadata[self.metadata_len..end].copy_from_slice(bytes);
        self.metadata_len = end;
    }

    fn flush(&mut self) {
        if self.metadata_len != 0 {
            self.hasher.update(&self.metadata[..self.metadata_len]);
            self.metadata_len = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake3_known_empty_vector_matches_specification() {
        assert_eq!(
            blake3::hash(b"").to_hex().as_str(),
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        );
    }

    #[test]
    fn domains_and_length_frames_do_not_alias() {
        let mut first = SemanticHasher::new(b"domain-a");
        first.field(b"x", b"ab").field(b"y", b"c");
        let mut second = SemanticHasher::new(b"domain-a");
        second.field(b"x", b"a").field(b"y", b"bc");
        let mut other_domain = SemanticHasher::new(b"domain-b");
        other_domain.field(b"x", b"ab").field(b"y", b"c");
        assert_ne!(first.finish(), second.finish());
        assert_ne!(
            SemanticHasher::new(b"domain-a")
                .field(b"x", b"value")
                .finish(),
            SemanticHasher::new(b"domain-b")
                .field(b"x", b"value")
                .finish()
        );
        assert_ne!(
            other_domain.finish(),
            SemanticHasher::new(b"domain-a").finish()
        );
    }

    #[test]
    fn one_field_mutation_changes_full_digest() {
        let mut first = SemanticHasher::new(b"mutation");
        first.u32(b"op", 1).u64(b"shape", 32);
        let mut second = SemanticHasher::new(b"mutation");
        second.u32(b"op", 2).u64(b"shape", 32);
        assert_ne!(first.finish(), second.finish());
    }

    #[test]
    fn digest_prefilter_collision_still_checks_exact_bytes() {
        let mut first = ExactKeyBuilder::new(b"collision");
        first.field(b"value", b"first");
        let mut second = ExactKeyBuilder::new(b"collision");
        second.field(b"value", b"second");
        let first = first.finish();
        let mut second = second.finish();
        second.digest = first.digest;
        assert_ne!(first, second);
    }
}
