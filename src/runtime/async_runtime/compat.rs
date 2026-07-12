use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};

use super::{
    Completion, CompletionControl, CompletionError, OwnedOperation, ResourceLease, SubmitError,
    SubmitFailure, Submitter,
};
use crate::runtime::{RuntimeError, RuntimeHal, SubmissionBindings};

const HAL_DOMAIN: u32 = u32::from_be_bytes(*b"HAL ");

/// Synchronous compatibility adapter for the existing [`RuntimeHal`].
///
/// It blocks only inside `submit`, matching the old provider behavior, and returns an already-ready
/// native future. It never blocks from `Future::poll` and does not claim transport overlap.
pub struct RuntimeHalAdapter<H, A, I> {
    hal: Arc<H>,
    next_sequence: AtomicU64,
    marker: PhantomData<fn() -> (A, I)>,
}

impl<H, A, I> RuntimeHalAdapter<H, A, I> {
    pub fn new(hal: Arc<H>) -> Self {
        Self {
            hal,
            next_sequence: AtomicU64::new(1),
            marker: PhantomData,
        }
    }

    pub fn hal(&self) -> &Arc<H> {
        &self.hal
    }
}

fn submit_error(error: RuntimeError) -> SubmitError {
    match error {
        RuntimeError::InvalidArgument | RuntimeError::Incompatible => SubmitError::InvalidOperation,
        RuntimeError::DeadlineExpired => SubmitError::AdmissionTimedOut,
        RuntimeError::OutOfMemory | RuntimeError::ResourceLimit => SubmitError::ResourceLimit,
        RuntimeError::DeviceLost | RuntimeError::Unsupported => SubmitError::DeviceUnavailable,
        RuntimeError::External { domain, code } => SubmitError::External { domain, code },
    }
}

fn completion_error(error: RuntimeError) -> CompletionError {
    match error {
        RuntimeError::DeviceLost => CompletionError::ExecutionFailed {
            domain: HAL_DOMAIN,
            code: -2,
        },
        RuntimeError::External { domain, code } => {
            CompletionError::ExecutionFailed { domain, code }
        }
        other => CompletionError::ExecutionFailed {
            domain: HAL_DOMAIN,
            code: match other {
                RuntimeError::Unsupported => -1,
                RuntimeError::Incompatible => -3,
                RuntimeError::InvalidArgument => -4,
                RuntimeError::DeadlineExpired => -5,
                RuntimeError::OutOfMemory => -6,
                RuntimeError::ResourceLimit => -7,
                RuntimeError::DeviceLost | RuntimeError::External { .. } => unreachable!(),
            },
        },
    }
}

impl<H, A, I> Submitter for RuntimeHalAdapter<H, A, I>
where
    H: RuntimeHal + Send + Sync + 'static,
    H::Program: ResourceLease,
    H::Queue: ResourceLease,
    A: ResourceLease,
    I: ResourceLease,
{
    type Operation = OwnedOperation<H::Program, H::Queue, A, I>;

    fn submit(
        &self,
        operation: Self::Operation,
    ) -> Result<Completion, SubmitFailure<Self::Operation>> {
        let descriptors = operation
            .bindings()
            .iter()
            .map(|binding| binding.descriptor())
            .collect::<Vec<_>>();
        let bindings = match SubmissionBindings::new(&descriptors) {
            Ok(bindings) => bindings,
            Err(error) => {
                return Err(SubmitFailure::Rejected {
                    error: submit_error(error),
                    operation,
                });
            }
        };
        let event = match self.hal.submit(
            operation.queue(),
            operation.program(),
            &bindings,
            operation.admission_deadline(),
        ) {
            Ok(event) => event,
            Err(error) => {
                return Err(SubmitFailure::Rejected {
                    error: submit_error(error),
                    operation,
                });
            }
        };

        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let (completion, control) = CompletionControl::pending(
            operation.id(),
            operation.epoch(),
            sequence,
            operation,
            false,
            None,
        );
        let wait_result = self.hal.wait(&event, crate::runtime::Deadline::Infinite);
        let destroy_result = self.hal.destroy_event(event);
        match wait_result.and(destroy_result) {
            Ok(()) => {
                control.succeed(true);
            }
            Err(RuntimeError::DeviceLost) => {
                control.mark_device_lost(0, true);
            }
            Err(error) => {
                control.fail(completion_error(error), true);
            }
        }
        Ok(completion)
    }
}
