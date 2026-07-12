//! Host-neutral asynchronous execution contracts.
//!
//! Rust [`Future`](core::future::Future) is the typed API. Providers complete operations through
//! [`CompletionControl`], while dynamic providers and foreign adapters use [`erased`]'s explicit
//! polling ABI. Nothing in this module creates threads or selects a process-global runtime.

mod compat;
pub(crate) mod completion;
mod erased;
mod executor;
pub mod mock;
mod operation;
mod transition;

pub use compat::RuntimeHalAdapter;
pub use completion::{
    wait_many, CancellationRequest, CancellationState, Completion, CompletionControl,
    CompletionError, CompletionInfo, CompletionResult, CompletionSource, DeviceEpochFanout,
    InFlightRegistry, Progress, WaitContext, WaitError, WaitManyOutcome, WaitMode, WaitStatus,
};
pub use erased::{
    CompletionAbiResult, CompletionAbiVTable, ErasedCompletion, ForeignWakeCallback, PollAbiStatus,
};
pub use executor::{
    BlockOnError, BlockingExecutorPort, BlockingRequest, CallerDrivenBlockingExecutor,
    CallerExecutor, RunReport, SpawnError, TaskHandle, TaskState,
};
pub use operation::{
    submit_and_wait, AsyncQueue, BatchSubmitResult, BindingLease, DeviceEpoch, ExecuteError,
    OperationId, OwnedOperation, ResourceLease, SubmitError, SubmitFailure, Submitter,
};
pub use transition::{
    OwnershipLease, OwnershipState, OwnershipTransition, QuarantinedLease, VisibilityDomain,
    VisibilityLease, VisibilityState, VisibilityTransition,
};

use alloc::boxed::Box;
use alloc::sync::Arc;
use core::sync::atomic::AtomicU32;

use crate::runtime::Deadline;

/// Result of one host atomic wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitChange {
    Changed,
    Spurious,
    TimedOut,
}

/// Focused host service shared by completion waits and cache single-flight coordination.
pub trait AtomicWait: Send + Sync + 'static {
    fn wait_u32(
        &self,
        state: &AtomicU32,
        expected: u32,
        deadline: Deadline,
    ) -> Result<WaitChange, WaitError>;

    fn wake_u32(&self, state: &AtomicU32, count: WakeCount) -> Result<(), WaitError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WakeCount {
    One,
    All,
}

/// Monotonic clock used to interpret absolute [`Deadline`] values.
pub trait MonotonicClock: Send + Sync + 'static {
    fn now_ns(&self) -> Result<u64, WaitError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgressBudget {
    pub max_work: usize,
}

impl ProgressBudget {
    pub const DEFAULT: Self = Self { max_work: 64 };
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProgressReport {
    pub work_done: usize,
    pub may_have_more: bool,
}

/// Caller-driven progress. Implementations must never block when called from `Future::poll`.
pub trait ProgressDriver: Send + Sync + 'static {
    fn progress(&self, budget: ProgressBudget) -> Result<ProgressReport, WaitError>;
}

#[derive(Default)]
pub struct NoProgress;

impl ProgressDriver for NoProgress {
    fn progress(&self, _budget: ProgressBudget) -> Result<ProgressReport, WaitError> {
        Ok(ProgressReport::default())
    }
}

/// Object-safe wake target used below platform reactors.
pub trait WakeSink: Send + Sync + 'static {
    fn wake(&self);
}

/// Host-owned registration contract. `Source` is adapter-specific and never enters a foreign ABI.
pub trait WakeSourceRegistrar: Send + Sync {
    type Source: ?Sized;
    type Registration;

    fn register(
        &self,
        source: &Self::Source,
        sink: Arc<dyn WakeSink>,
    ) -> Result<Self::Registration, WaitError>;

    fn unregister(&self, registration: Self::Registration) -> Result<(), WaitError>;
}

pub type BlockingJob = Box<dyn FnOnce() -> CompletionResult + Send + 'static>;
