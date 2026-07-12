//! Public Intel Job Command ABI pinned to linux-npu-driver v1.32.0.
//!
//! Source: `firmware/include/api/vpu_jsm_job_cmd_api.h` (MIT), API 4.15. These
//! definitions are only parsed by the dry-run validator; no submit ioctl is exposed.

#![allow(dead_code)]

pub const JOB_CMD_API_MAJOR: u32 = 4;
pub const JOB_CMD_API_MINOR: u32 = 15;
pub const JOB_CMD_API_WORD: u32 = (JOB_CMD_API_MAJOR << 16) | JOB_CMD_API_MINOR;
pub const JOB_CMD_API_INDEX: usize = 3;
pub const NNRT_API_MAJOR: u32 = 11;
pub const NNRT_API_MINOR: u32 = 13;
pub const NNRT_API_WORD: u32 = (NNRT_API_MAJOR << 16) | NNRT_API_MINOR;
pub const NNRT_API_INDEX: usize = 7;
pub const ELF_LOADER_ABI_MAJOR: u32 = 1;
pub const ELF_LOADER_ABI_MINOR: u32 = 3;

pub const VPU_CMD_NOP: u16 = 0x0001;
pub const VPU_CMD_TIMESTAMP: u16 = 0x0100;
pub const VPU_CMD_FENCE_WAIT: u16 = 0x0101;
pub const VPU_CMD_FENCE_SIGNAL: u16 = 0x0102;
pub const VPU_CMD_BARRIER: u16 = 0x0103;
pub const VPU_CMD_METRIC_QUERY_BEGIN: u16 = 0x0104;
pub const VPU_CMD_METRIC_QUERY_END: u16 = 0x0105;
pub const VPU_CMD_MEMORY_FILL: u16 = 0x0202;
pub const VPU_CMD_COPY: u16 = 0x0302;
pub const VPU_CMD_INFERENCE_EXECUTE: u16 = 0x0306;
pub const MAX_COPY_DESCRIPTORS: u32 = 4096;
pub const MAX_COMMANDS: usize = 4096;
pub const MAX_MEMORY_FILL_BYTES: u64 = 16 * 1024 * 1024;
pub const COPY_DESCRIPTOR_40XX_BYTES: u64 = 192;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VpuResourceDescriptor {
    pub address: u64,
    pub width: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VpuCommandBufferHeader {
    pub command_buffer_bytes: u32,
    pub command_offset: u32,
    pub api_version: u32,
    pub reserved: u32,
    pub descriptor_heap_base: u64,
    pub submission_timestamp: u64,
    pub fence_heap_base: u64,
    pub context_save_area: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VpuCommandHeader {
    pub command_type: u16,
    pub bytes: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VpuInferenceExecute {
    pub header: VpuCommandHeader,
    pub reserved: u32,
    pub inference_id: u64,
    pub mapped_inference: VpuResourceDescriptor,
}

const _: [(); 16] = [(); core::mem::size_of::<VpuResourceDescriptor>()];
const _: [(); 8] = [(); core::mem::align_of::<VpuResourceDescriptor>()];
const _: [(); 48] = [(); core::mem::size_of::<VpuCommandBufferHeader>()];
const _: [(); 8] = [(); core::mem::align_of::<VpuCommandBufferHeader>()];
const _: [(); 4] = [(); core::mem::size_of::<VpuCommandHeader>()];
const _: [(); 2] = [(); core::mem::align_of::<VpuCommandHeader>()];
const _: [(); 32] = [(); core::mem::size_of::<VpuInferenceExecute>()];
const _: [(); 8] = [(); core::mem::align_of::<VpuInferenceExecute>()];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_job_command_layout_is_pinned() {
        assert_eq!(JOB_CMD_API_WORD, 0x0004_000f);
        assert_eq!(NNRT_API_WORD, 0x000b_000d);
        assert_eq!(
            core::mem::offset_of!(VpuCommandBufferHeader, context_save_area),
            40
        );
        assert_eq!(
            core::mem::offset_of!(VpuInferenceExecute, mapped_inference),
            16
        );
    }
}
