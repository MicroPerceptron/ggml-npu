//! Direct session health: cancel admission, job-status classification, and reset recovery.
//!
//! Synchronous Direct still waits on the command BO; these boundaries do not invent a kernel
//! abort ioctl. They define fail-closed admission after cancel/device-loss and an explicit reset
//! path that re-validates firmware before clearing the latch.

#![allow(dead_code)]

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use crate::runtime::RuntimeError;

use super::uapi::{DRM_IVPU_JOB_STATUS_ABORTED, DRM_IVPU_JOB_STATUS_SUCCESS};
use super::validate::{validate_device, ValidationError};
use super::DeviceParameters;

/// Reason latched when the Direct session refuses further work without an explicit reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultReason {
    SubmitFailed,
    WaitIoctl,
    DeadlineExpired,
    JobFault { job_status: u32 },
    UnexpectedAbort,
    FirmwareMismatch,
    DeviceQuery,
}

/// Outcome of classifying a command-BO wait result under the current cancel request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitOutcome {
    Success,
    Cancelled,
    DeadlineExpired,
    DeviceLost { reason: FaultReason },
    Transport(RuntimeError),
}

/// Snapshot of Direct session health for tests and diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HealthState {
    Healthy,
    CancelRequested,
    DeviceLost {
        epoch: u64,
        reason: FaultReason,
    },
}

static CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);
static DEVICE_LOST: AtomicBool = AtomicBool::new(false);
static FAULT_REASON: AtomicU32 = AtomicU32::new(0);
static FAULT_JOB_STATUS: AtomicU32 = AtomicU32::new(0);
static FAULT_EPOCH: AtomicU64 = AtomicU64::new(0);

fn encode_reason(reason: FaultReason) -> u32 {
    match reason {
        FaultReason::SubmitFailed => 1,
        FaultReason::WaitIoctl => 2,
        FaultReason::DeadlineExpired => 3,
        FaultReason::JobFault { .. } => 4,
        FaultReason::UnexpectedAbort => 5,
        FaultReason::FirmwareMismatch => 6,
        FaultReason::DeviceQuery => 7,
    }
}

fn decode_reason(code: u32, job_status: u32) -> FaultReason {
    match code {
        1 => FaultReason::SubmitFailed,
        2 => FaultReason::WaitIoctl,
        3 => FaultReason::DeadlineExpired,
        4 => FaultReason::JobFault { job_status },
        5 => FaultReason::UnexpectedAbort,
        6 => FaultReason::FirmwareMismatch,
        7 => FaultReason::DeviceQuery,
        _ => FaultReason::WaitIoctl,
    }
}

/// Classify a firmware/command-BO job status word from `DRM_IOCTL_IVPU_BO_WAIT`.
pub fn classify_job_status(job_status: u32) -> WaitOutcome {
    if job_status == DRM_IVPU_JOB_STATUS_SUCCESS {
        return WaitOutcome::Success;
    }
    if job_status == DRM_IVPU_JOB_STATUS_ABORTED {
        if CANCEL_REQUESTED.load(Ordering::Acquire) {
            return WaitOutcome::Cancelled;
        }
        return WaitOutcome::DeviceLost {
            reason: FaultReason::UnexpectedAbort,
        };
    }
    WaitOutcome::DeviceLost {
        reason: FaultReason::JobFault { job_status },
    }
}

/// Classify a wait ioctl result, including cancel races and transport errors.
pub fn classify_wait_result(result: Result<u32, RuntimeError>) -> WaitOutcome {
    match result {
        Ok(status) => classify_job_status(status),
        Err(RuntimeError::DeadlineExpired) => WaitOutcome::DeadlineExpired,
        Err(RuntimeError::DeviceLost) => WaitOutcome::DeviceLost {
            reason: FaultReason::WaitIoctl,
        },
        Err(error) => WaitOutcome::Transport(error),
    }
}

/// Reject new Direct work when cancel is pending or the device is latched lost.
pub fn admit_dispatch() -> Result<(), WaitOutcome> {
    if DEVICE_LOST.load(Ordering::Acquire) {
        return Err(WaitOutcome::DeviceLost {
            reason: decode_reason(
                FAULT_REASON.load(Ordering::Acquire),
                FAULT_JOB_STATUS.load(Ordering::Acquire),
            ),
        });
    }
    if CANCEL_REQUESTED.load(Ordering::Acquire) {
        return Err(WaitOutcome::Cancelled);
    }
    Ok(())
}

pub fn request_cancel() -> bool {
    if DEVICE_LOST.load(Ordering::Acquire) {
        return false;
    }
    !CANCEL_REQUESTED.swap(true, Ordering::AcqRel)
}

pub fn clear_cancel() {
    CANCEL_REQUESTED.store(false, Ordering::Release);
}

pub fn cancel_requested() -> bool {
    CANCEL_REQUESTED.load(Ordering::Acquire)
}

pub fn latch_device_lost(reason: FaultReason) -> u64 {
    let job_status = match reason {
        FaultReason::JobFault { job_status } => job_status,
        _ => 0,
    };
    FAULT_REASON.store(encode_reason(reason), Ordering::Release);
    FAULT_JOB_STATUS.store(job_status, Ordering::Release);
    DEVICE_LOST.store(true, Ordering::Release);
    CANCEL_REQUESTED.store(false, Ordering::Release);
    FAULT_EPOCH.fetch_add(1, Ordering::AcqRel).wrapping_add(1)
}

pub fn apply_wait_outcome(outcome: WaitOutcome) -> WaitOutcome {
    match outcome {
        WaitOutcome::Success => WaitOutcome::Success,
        WaitOutcome::Cancelled => {
            clear_cancel();
            WaitOutcome::Cancelled
        }
        WaitOutcome::DeadlineExpired => {
            latch_device_lost(FaultReason::DeadlineExpired);
            WaitOutcome::DeadlineExpired
        }
        WaitOutcome::DeviceLost { reason } => {
            latch_device_lost(reason);
            WaitOutcome::DeviceLost { reason }
        }
        WaitOutcome::Transport(error) => {
            if matches!(
                error,
                RuntimeError::DeviceLost | RuntimeError::External { .. }
            ) {
                latch_device_lost(FaultReason::WaitIoctl);
            }
            WaitOutcome::Transport(error)
        }
    }
}

pub fn note_submit_failure(error: RuntimeError) -> WaitOutcome {
    let reason = match error {
        RuntimeError::DeviceLost => FaultReason::SubmitFailed,
        RuntimeError::Incompatible => FaultReason::FirmwareMismatch,
        _ => FaultReason::SubmitFailed,
    };
    latch_device_lost(reason);
    WaitOutcome::DeviceLost { reason }
}

pub fn health_state() -> HealthState {
    if DEVICE_LOST.load(Ordering::Acquire) {
        return HealthState::DeviceLost {
            epoch: FAULT_EPOCH.load(Ordering::Acquire),
            reason: decode_reason(
                FAULT_REASON.load(Ordering::Acquire),
                FAULT_JOB_STATUS.load(Ordering::Acquire),
            ),
        };
    }
    if CANCEL_REQUESTED.load(Ordering::Acquire) {
        return HealthState::CancelRequested;
    }
    HealthState::Healthy
}

pub fn fault_epoch() -> u64 {
    FAULT_EPOCH.load(Ordering::Acquire)
}

/// Re-validate firmware/device pins before clearing a latched fault.
///
/// Callers must drop resident programs before invoking this so a recovered session cannot reuse
/// buffers from a lost epoch.
pub fn try_reset(parameters: &DeviceParameters) -> Result<u64, ValidationError> {
    validate_device(parameters)?;
    CANCEL_REQUESTED.store(false, Ordering::Release);
    DEVICE_LOST.store(false, Ordering::Release);
    FAULT_REASON.store(0, Ordering::Release);
    FAULT_JOB_STATUS.store(0, Ordering::Release);
    Ok(FAULT_EPOCH.load(Ordering::Acquire))
}

/// Negotiate firmware/API pins at context open. Fail closed on mismatch.
pub fn negotiate_firmware(parameters: &DeviceParameters) -> Result<(), RuntimeError> {
    validate_device(parameters).map_err(|error| match error {
        ValidationError::FirmwareMismatch => RuntimeError::Incompatible,
        ValidationError::WrongDevice | ValidationError::WrongRevision => {
            RuntimeError::Unsupported
        }
        ValidationError::UnsupportedPlatform => RuntimeError::Unsupported,
        _ => RuntimeError::Incompatible,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::firmware::{
        JOB_CMD_API_INDEX, JOB_CMD_API_WORD, NNRT_API_INDEX, NNRT_API_WORD,
    };
    use super::super::validate::{LUNAR_LAKE_DEVICE_ID, LUNAR_LAKE_REVISION};

    fn reset_globals() {
        CANCEL_REQUESTED.store(false, Ordering::Release);
        DEVICE_LOST.store(false, Ordering::Release);
        FAULT_REASON.store(0, Ordering::Release);
        FAULT_JOB_STATUS.store(0, Ordering::Release);
        FAULT_EPOCH.store(0, Ordering::Release);
    }

    fn pinned_parameters() -> DeviceParameters {
        let mut firmware_api = [0u64; 8];
        firmware_api[JOB_CMD_API_INDEX] = JOB_CMD_API_WORD as u64;
        firmware_api[NNRT_API_INDEX] = NNRT_API_WORD as u64;
        DeviceParameters {
            device_id: LUNAR_LAKE_DEVICE_ID,
            revision: LUNAR_LAKE_REVISION,
            platform_type: 0,
            core_clock_rate: 0,
            context_id: 1,
            context_base_address: 0,
            firmware_api,
            tile_config: 0,
            sku: 0,
            capability_bits: 0,
            max_contexts: 1,
            preempt_buffer_bytes: 0,
        }
    }

    #[test]
    fn job_status_success_and_abort_classify() {
        reset_globals();
        assert_eq!(
            classify_job_status(DRM_IVPU_JOB_STATUS_SUCCESS),
            WaitOutcome::Success
        );
        assert_eq!(
            classify_job_status(DRM_IVPU_JOB_STATUS_ABORTED),
            WaitOutcome::DeviceLost {
                reason: FaultReason::UnexpectedAbort
            }
        );
        assert!(request_cancel());
        assert_eq!(
            classify_job_status(DRM_IVPU_JOB_STATUS_ABORTED),
            WaitOutcome::Cancelled
        );
        assert_eq!(
            classify_job_status(7),
            WaitOutcome::DeviceLost {
                reason: FaultReason::JobFault { job_status: 7 }
            }
        );
        reset_globals();
    }

    #[test]
    fn cancel_blocks_admission_without_device_loss() {
        reset_globals();
        assert!(request_cancel());
        assert!(!request_cancel());
        assert_eq!(admit_dispatch(), Err(WaitOutcome::Cancelled));
        assert_eq!(health_state(), HealthState::CancelRequested);
        clear_cancel();
        assert_eq!(admit_dispatch(), Ok(()));
        assert_eq!(health_state(), HealthState::Healthy);
        reset_globals();
    }

    #[test]
    fn fault_latches_and_reset_requires_firmware_match() {
        reset_globals();
        let epoch = latch_device_lost(FaultReason::JobFault { job_status: 42 });
        assert_eq!(epoch, 1);
        assert_eq!(
            admit_dispatch(),
            Err(WaitOutcome::DeviceLost {
                reason: FaultReason::JobFault { job_status: 42 }
            })
        );
        assert!(!request_cancel());

        let mut wrong = pinned_parameters();
        wrong.firmware_api[JOB_CMD_API_INDEX] = 0;
        assert_eq!(
            try_reset(&wrong),
            Err(ValidationError::FirmwareMismatch)
        );
        assert!(matches!(
            health_state(),
            HealthState::DeviceLost {
                epoch: 1,
                reason: FaultReason::JobFault { job_status: 42 }
            }
        ));

        assert_eq!(try_reset(&pinned_parameters()), Ok(1));
        assert_eq!(health_state(), HealthState::Healthy);
        assert_eq!(admit_dispatch(), Ok(()));
        reset_globals();
    }

    #[test]
    fn wait_outcomes_apply_cancel_and_deadline_rules() {
        reset_globals();
        assert!(request_cancel());
        assert_eq!(
            apply_wait_outcome(WaitOutcome::Cancelled),
            WaitOutcome::Cancelled
        );
        assert!(!cancel_requested());
        assert_eq!(
            apply_wait_outcome(WaitOutcome::DeadlineExpired),
            WaitOutcome::DeadlineExpired
        );
        assert!(matches!(
            health_state(),
            HealthState::DeviceLost {
                reason: FaultReason::DeadlineExpired,
                ..
            }
        ));
        reset_globals();
    }

    #[test]
    fn negotiate_firmware_rejects_mismatch() {
        let mut wrong = pinned_parameters();
        wrong.firmware_api[NNRT_API_INDEX] = 0;
        assert_eq!(
            negotiate_firmware(&wrong),
            Err(RuntimeError::Incompatible)
        );
        assert_eq!(negotiate_firmware(&pinned_parameters()), Ok(()));
    }
}
