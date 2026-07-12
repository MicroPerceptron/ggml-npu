//! Checked subset of Linux `intel_vpu` UAPI.
//!
//! Source: `/usr/include/drm/ivpu_accel.h`, SPDX
//! `GPL-2.0-only WITH Linux-syscall-note`, copyright 2020-2025 Intel. Only GET_PARAM and GEM BO
//! ownership are represented here. Submission, command queues, and firmware packets are omitted.

#![allow(dead_code)]

use core::mem::{align_of, size_of};

const IOC_NRBITS: u32 = 8;
const IOC_TYPEBITS: u32 = 8;
const IOC_SIZEBITS: u32 = 14;
const IOC_NRSHIFT: u32 = 0;
const IOC_TYPESHIFT: u32 = IOC_NRSHIFT + IOC_NRBITS;
const IOC_SIZESHIFT: u32 = IOC_TYPESHIFT + IOC_TYPEBITS;
const IOC_DIRSHIFT: u32 = IOC_SIZESHIFT + IOC_SIZEBITS;
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;
const DRM_IOCTL_BASE: u32 = b'd' as u32;
const DRM_COMMAND_BASE: u32 = 0x40;

const fn ioc(direction: u32, ioctl_type: u32, number: u32, bytes: usize) -> u32 {
    (direction << IOC_DIRSHIFT)
        | (ioctl_type << IOC_TYPESHIFT)
        | (number << IOC_NRSHIFT)
        | ((bytes as u32) << IOC_SIZESHIFT)
}

const fn drm_iowr(number: u32, bytes: usize) -> u32 {
    ioc(IOC_READ | IOC_WRITE, DRM_IOCTL_BASE, number, bytes)
}

const fn drm_iow(number: u32, bytes: usize) -> u32 {
    ioc(IOC_WRITE, DRM_IOCTL_BASE, number, bytes)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrmIvpuParam {
    pub param: u32,
    pub index: u32,
    pub value: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrmIvpuBoCreate {
    pub size: u64,
    pub flags: u32,
    pub handle: u32,
    pub vpu_addr: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrmIvpuBoInfo {
    pub handle: u32,
    pub flags: u32,
    pub vpu_addr: u64,
    pub mmap_offset: u64,
    pub size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrmGemClose {
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrmIvpuBoWait {
    pub handle: u32,
    pub flags: u32,
    pub timeout_ns: i64,
    pub job_status: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrmIvpuCmdqCreate {
    pub cmdq_id: u32,
    pub priority: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrmIvpuCmdqDestroy {
    pub cmdq_id: u32,
}

/// Layout only. No direct submit wrapper exists until command bytes pass the dry-run validator.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrmIvpuCmdqSubmit {
    pub buffers_ptr: u64,
    pub buffer_count: u32,
    pub cmdq_id: u32,
    pub flags: u32,
    pub commands_offset: u32,
    pub preempt_buffer_index: u32,
    pub reserved: u32,
}

pub const DRM_IVPU_PARAM_DEVICE_ID: u32 = 0;
pub const DRM_IVPU_PARAM_DEVICE_REVISION: u32 = 1;
pub const DRM_IVPU_PARAM_PLATFORM_TYPE: u32 = 2;
pub const DRM_IVPU_PARAM_CORE_CLOCK_RATE: u32 = 3;
pub const DRM_IVPU_PARAM_NUM_CONTEXTS: u32 = 4;
pub const DRM_IVPU_PARAM_CONTEXT_BASE_ADDRESS: u32 = 5;
pub const DRM_IVPU_PARAM_CONTEXT_ID: u32 = 7;
pub const DRM_IVPU_PARAM_FW_API_VERSION: u32 = 8;
pub const DRM_IVPU_PARAM_UNIQUE_INFERENCE_ID: u32 = 10;
pub const DRM_IVPU_PARAM_TILE_CONFIG: u32 = 11;
pub const DRM_IVPU_PARAM_SKU: u32 = 12;
pub const DRM_IVPU_PARAM_CAPABILITIES: u32 = 13;
pub const DRM_IVPU_PARAM_PREEMPT_BUFFER_SIZE: u32 = 14;

pub const DRM_IVPU_CAP_METRIC_STREAMER: u32 = 1;
pub const DRM_IVPU_CAP_DMA_MEMORY_RANGE: u32 = 2;
pub const DRM_IVPU_CAP_MANAGE_CMDQ: u32 = 3;
pub const DRM_IVPU_CAP_BO_CREATE_FROM_USERPTR: u32 = 4;

pub const DRM_IVPU_JOB_PRIORITY_DEFAULT: u32 = 0;
pub const DRM_IVPU_JOB_PRIORITY_IDLE: u32 = 1;
pub const DRM_IVPU_JOB_PRIORITY_NORMAL: u32 = 2;
pub const DRM_IVPU_JOB_PRIORITY_FOCUS: u32 = 3;
pub const DRM_IVPU_JOB_PRIORITY_REALTIME: u32 = 4;
pub const DRM_IVPU_CMDQ_FLAG_TURBO: u32 = 1;
pub const DRM_IVPU_JOB_STATUS_SUCCESS: u32 = 0;
pub const DRM_IVPU_JOB_STATUS_ABORTED: u32 = 256;

pub const DRM_IVPU_BO_HIGH_MEM: u32 = 0x0000_0001;
pub const DRM_IVPU_BO_MAPPABLE: u32 = 0x0000_0002;
pub const DRM_IVPU_BO_DMA_MEM: u32 = 0x0000_0004;
pub const DRM_IVPU_BO_READ_ONLY: u32 = 0x0000_0008;
pub const DRM_IVPU_BO_WC: u32 = 0x0002_0000;
pub const DRM_IVPU_BO_FLAGS: u32 = DRM_IVPU_BO_HIGH_MEM
    | DRM_IVPU_BO_MAPPABLE
    | DRM_IVPU_BO_DMA_MEM
    | DRM_IVPU_BO_READ_ONLY
    | DRM_IVPU_BO_WC;

pub const DRM_IOCTL_IVPU_GET_PARAM: u32 = drm_iowr(DRM_COMMAND_BASE, size_of::<DrmIvpuParam>());
pub const DRM_IOCTL_IVPU_BO_CREATE: u32 =
    drm_iowr(DRM_COMMAND_BASE + 0x02, size_of::<DrmIvpuBoCreate>());
pub const DRM_IOCTL_IVPU_BO_INFO: u32 =
    drm_iowr(DRM_COMMAND_BASE + 0x03, size_of::<DrmIvpuBoInfo>());
pub const DRM_IOCTL_IVPU_BO_WAIT: u32 =
    drm_iowr(DRM_COMMAND_BASE + 0x06, size_of::<DrmIvpuBoWait>());
pub const DRM_IOCTL_IVPU_CMDQ_CREATE: u32 =
    drm_iowr(DRM_COMMAND_BASE + 0x0b, size_of::<DrmIvpuCmdqCreate>());
pub const DRM_IOCTL_IVPU_CMDQ_DESTROY: u32 =
    drm_iow(DRM_COMMAND_BASE + 0x0c, size_of::<DrmIvpuCmdqDestroy>());
pub const DRM_IOCTL_IVPU_CMDQ_SUBMIT: u32 =
    drm_iow(DRM_COMMAND_BASE + 0x0d, size_of::<DrmIvpuCmdqSubmit>());
pub const DRM_IOCTL_GEM_CLOSE: u32 = drm_iow(0x09, size_of::<DrmGemClose>());

const _: [(); 16] = [(); size_of::<DrmIvpuParam>()];
const _: [(); 8] = [(); align_of::<DrmIvpuParam>()];
const _: [(); 24] = [(); size_of::<DrmIvpuBoCreate>()];
const _: [(); 8] = [(); align_of::<DrmIvpuBoCreate>()];
const _: [(); 32] = [(); size_of::<DrmIvpuBoInfo>()];
const _: [(); 8] = [(); align_of::<DrmIvpuBoInfo>()];
const _: [(); 8] = [(); size_of::<DrmGemClose>()];
const _: [(); 4] = [(); align_of::<DrmGemClose>()];
const _: [(); 24] = [(); size_of::<DrmIvpuBoWait>()];
const _: [(); 8] = [(); align_of::<DrmIvpuBoWait>()];
const _: [(); 12] = [(); size_of::<DrmIvpuCmdqCreate>()];
const _: [(); 4] = [(); align_of::<DrmIvpuCmdqCreate>()];
const _: [(); 4] = [(); size_of::<DrmIvpuCmdqDestroy>()];
const _: [(); 4] = [(); align_of::<DrmIvpuCmdqDestroy>()];
const _: [(); 32] = [(); size_of::<DrmIvpuCmdqSubmit>()];
const _: [(); 8] = [(); align_of::<DrmIvpuCmdqSubmit>()];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ioctl_numbers_match_linux_ioc_encoding() {
        assert_eq!(DRM_IOCTL_IVPU_GET_PARAM, 0xc010_6440);
        assert_eq!(DRM_IOCTL_IVPU_BO_CREATE, 0xc018_6442);
        assert_eq!(DRM_IOCTL_IVPU_BO_INFO, 0xc020_6443);
        assert_eq!(DRM_IOCTL_IVPU_BO_WAIT, 0xc018_6446);
        assert_eq!(DRM_IOCTL_IVPU_CMDQ_CREATE, 0xc00c_644b);
        assert_eq!(DRM_IOCTL_IVPU_CMDQ_DESTROY, 0x4004_644c);
        assert_eq!(DRM_IOCTL_IVPU_CMDQ_SUBMIT, 0x4020_644d);
        assert_eq!(DRM_IOCTL_GEM_CLOSE, 0x4008_6409);
    }

    #[test]
    fn layouts_match_installed_uapi() {
        assert_eq!(
            (size_of::<DrmIvpuParam>(), align_of::<DrmIvpuParam>()),
            (16, 8)
        );
        assert_eq!(
            (size_of::<DrmIvpuBoCreate>(), align_of::<DrmIvpuBoCreate>()),
            (24, 8)
        );
        assert_eq!(
            (size_of::<DrmIvpuBoInfo>(), align_of::<DrmIvpuBoInfo>()),
            (32, 8)
        );
        assert_eq!(
            (size_of::<DrmGemClose>(), align_of::<DrmGemClose>()),
            (8, 4)
        );
        assert_eq!(core::mem::offset_of!(DrmIvpuParam, value), 8);
        assert_eq!(core::mem::offset_of!(DrmIvpuBoCreate, handle), 12);
        assert_eq!(core::mem::offset_of!(DrmIvpuBoCreate, vpu_addr), 16);
        assert_eq!(core::mem::offset_of!(DrmIvpuBoInfo, mmap_offset), 16);
        assert_eq!(core::mem::offset_of!(DrmIvpuBoInfo, size), 24);
        assert_eq!(core::mem::offset_of!(DrmIvpuBoWait, timeout_ns), 8);
        assert_eq!(core::mem::offset_of!(DrmIvpuBoWait, job_status), 16);
        assert_eq!(core::mem::offset_of!(DrmIvpuCmdqSubmit, cmdq_id), 12);
        assert_eq!(
            core::mem::offset_of!(DrmIvpuCmdqSubmit, commands_offset),
            20
        );
    }
}
