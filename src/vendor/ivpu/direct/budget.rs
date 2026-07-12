//! Queried and user-configurable direct-runtime memory budgets.

use alloc::boxed::Box;
use core::ffi::CStr;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::host::{ConfigService, HostEnvironment, MemoryStatusService};
use crate::runtime::RuntimeError;

use super::prepare::PreparationBudget;
use super::DeviceParameters;

const MIB: u64 = 1024 * 1024;
const DEFAULT_HOST_FLOOR_MIB: u64 = 4096;
const DEFAULT_RESIDENT_MIB: u64 = 3072;
const LOW_WATER_PERCENT: u64 = 90;

static BUDGET: AtomicPtr<RuntimeBudget> = AtomicPtr::new(ptr::null_mut());

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeBudget {
    pub host_available_bytes: u64,
    pub host_floor_bytes: u64,
    pub resident_program_bytes: u64,
    pub resident_low_water_bytes: u64,
    pub disk_artifact_bytes: Option<u64>,
    pub preemption_bytes: u64,
}

impl RuntimeBudget {
    pub fn query(parameters: &DeviceParameters) -> Result<Self, RuntimeError> {
        let current = BUDGET.load(Ordering::Acquire);
        if !current.is_null() {
            // SAFETY: the process budget is immutable and retained after publication.
            return Ok(unsafe { *current });
        }
        let candidate = Self::build(parameters)?;
        let candidate = Box::into_raw(Box::new(candidate));
        match BUDGET.compare_exchange(
            ptr::null_mut(),
            candidate,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(unsafe { *candidate }),
            Err(winner) => {
                // SAFETY: this allocation was not published.
                drop(unsafe { Box::from_raw(candidate) });
                // SAFETY: the winner remains immutable and process-resident.
                Ok(unsafe { *winner })
            }
        }
    }

    fn build(parameters: &DeviceParameters) -> Result<Self, RuntimeError> {
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
            .unwrap_or(DEFAULT_RESIDENT_MIB.saturating_mul(MIB))
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
            resident_low_water_bytes: resident_program_bytes.saturating_mul(LOW_WATER_PERCENT)
                / 100,
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
    let value = crate::host::builtin().config().value(name)?;
    let mut parsed = 0u64;
    let bytes = value.to_bytes();
    if bytes.is_empty() || bytes.len() > 20 {
        return None;
    }
    for &byte in bytes {
        if !byte.is_ascii_digit() {
            return None;
        }
        parsed = parsed
            .checked_mul(10)?
            .checked_add(u64::from(byte - b'0'))?;
    }
    Some(parsed)
}

fn host_available_bytes() -> Option<u64> {
    crate::host::builtin()
        .memory()
        .constrained_available_memory_bytes()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_available_memory_is_cgroup_constrained_and_budget_is_stable() {
        assert!(host_available_bytes().is_some_and(|bytes| bytes > 64 * MIB));
    }
}
