//! IVPU-owned interpretation of generic artifact format and target identity fields.

use crate::runtime::artifact::{ArtifactFormat, TargetIdentity};

pub(crate) const IVPU_ELF_FORMAT: ArtifactFormat = match ArtifactFormat::new(2) {
    Some(format) => format,
    None => panic!("IVPU ELF format identifier must be nonzero"),
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IvpuTargetIdentity {
    pub vendor_id: u32,
    pub device_id: u32,
    pub revision: u32,
    pub architecture: u32,
    pub firmware_api: [u32; 8],
}

impl IvpuTargetIdentity {
    pub(crate) const fn encode(self) -> TargetIdentity {
        let mut words = [0; 12];
        words[0] = self.vendor_id;
        words[1] = self.device_id;
        words[2] = self.revision;
        words[3] = self.architecture;
        let mut index = 0;
        while index < self.firmware_api.len() {
            words[4 + index] = self.firmware_api[index];
            index += 1;
        }
        TargetIdentity::new(words)
    }

    pub(crate) const fn decode(identity: TargetIdentity) -> Self {
        let words = identity.words();
        let mut firmware_api = [0; 8];
        let mut index = 0;
        while index < firmware_api.len() {
            firmware_api[index] = words[4 + index];
            index += 1;
        }
        Self {
            vendor_id: words[0],
            device_id: words[1],
            revision: words[2],
            architecture: words[3],
            firmware_api,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ivpu_target_identity_round_trips_through_opaque_words() {
        let target = IvpuTargetIdentity {
            vendor_id: 0x8086,
            device_id: 0x643e,
            revision: 4,
            architecture: 40,
            firmware_api: [1, 2, 3, 4, 5, 6, 7, 8],
        };
        assert_eq!(IvpuTargetIdentity::decode(target.encode()), target);
    }
}
