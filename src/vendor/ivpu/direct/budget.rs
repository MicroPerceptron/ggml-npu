//! Queried and user-configurable direct-runtime memory budgets.

use core::ffi::{c_char, CStr};

use rustix::fs::{open, Mode, OFlags};
use rustix::io::read;

use crate::runtime::RuntimeError;

use super::prepare::PreparationBudget;
use super::DeviceParameters;

const MIB: u64 = 1024 * 1024;
const DEFAULT_HOST_FLOOR_MIB: u64 = 4096;

extern "C" {
    fn getenv(name: *const c_char) -> *const c_char;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeBudget {
    pub host_available_bytes: u64,
    pub host_floor_bytes: u64,
    pub resident_program_bytes: u64,
    pub disk_artifact_bytes: Option<u64>,
    pub preemption_bytes: u64,
}

impl RuntimeBudget {
    pub fn query(parameters: &DeviceParameters) -> Result<Self, RuntimeError> {
        let host_available_bytes = host_available_bytes().ok_or(RuntimeError::Unsupported)?;
        let host_floor_bytes = env_mib(c"GGML_NPU_HOST_FLOOR_MB")
            .unwrap_or(DEFAULT_HOST_FLOOR_MIB)
            .checked_mul(MIB)
            .ok_or(RuntimeError::InvalidArgument)?;
        let available_for_runtime = host_available_bytes
            .checked_sub(host_floor_bytes)
            .ok_or(RuntimeError::OutOfMemory)?;
        let user_capacity = available_for_runtime
            .checked_sub(parameters.preempt_buffer_bytes)
            .ok_or(RuntimeError::OutOfMemory)?;
        let resident_program_bytes = env_mib(c"GGML_NPU_RESIDENT_MB")
            .and_then(|mib| mib.checked_mul(MIB))
            .unwrap_or(user_capacity)
            .min(user_capacity);
        if resident_program_bytes == 0 {
            return Err(RuntimeError::OutOfMemory);
        }
        let disk_artifact_bytes =
            env_mib(c"GGML_NPU_DISK_CACHE_MB").and_then(|mib| mib.checked_mul(MIB));
        Ok(Self {
            host_available_bytes,
            host_floor_bytes,
            resident_program_bytes,
            disk_artifact_bytes,
            preemption_bytes: parameters.preempt_buffer_bytes,
        })
    }

    pub fn preparation(self) -> PreparationBudget {
        PreparationBudget {
            max_user_bo_bytes: self.resident_program_bytes,
            max_total_accounted_bytes: self
                .resident_program_bytes
                .saturating_add(self.preemption_bytes),
        }
    }
}

fn env_mib(name: &CStr) -> Option<u64> {
    // SAFETY: getenv returns either null or a process-lifetime NUL-terminated environment value.
    let value = unsafe { getenv(name.as_ptr()) };
    if value.is_null() {
        return None;
    }
    let mut parsed = 0u64;
    let mut length = 0usize;
    loop {
        // SAFETY: scanning one byte at a time until the required terminating NUL.
        let byte = unsafe { *value.add(length) as u8 };
        if byte == 0 {
            break;
        }
        if !byte.is_ascii_digit() || length >= 20 {
            return None;
        }
        parsed = parsed
            .checked_mul(10)?
            .checked_add(u64::from(byte - b'0'))?;
        length += 1;
    }
    (length != 0).then_some(parsed)
}

fn host_available_bytes() -> Option<u64> {
    let file = open(
        c"/proc/meminfo",
        OFlags::RDONLY | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .ok()?;
    let mut buffer = [0u8; 4096];
    let length = read(&file, &mut buffer).ok()?;
    let text = core::str::from_utf8(&buffer[..length]).ok()?;
    let line = text
        .split('\n')
        .find(|line| line.starts_with("MemAvailable:"))?;
    let kib = line.split_ascii_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_available_memory_is_queried_without_fixed_ceiling() {
        assert!(host_available_bytes().is_some_and(|bytes| bytes > 64 * MIB));
    }
}
