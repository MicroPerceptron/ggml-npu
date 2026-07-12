use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::num::NonZeroU64;

use super::{Completion, CompletionError, CompletionInfo};
use crate::runtime::{Binding, Deadline};

macro_rules! nonzero_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(NonZeroU64);

        impl $name {
            pub const fn new(value: u64) -> Option<Self> {
                match NonZeroU64::new(value) {
                    Some(value) => Some(Self(value)),
                    None => None,
                }
            }

            pub const fn get(self) -> u64 {
                self.0.get()
            }
        }
    };
}

nonzero_id!(OperationId);
nonzero_id!(DeviceEpoch);

/// Strongly-owned resource held through safe operation reclamation.
pub trait ResourceLease: Send + Sync + 'static {}

impl<T: Send + Sync + 'static> ResourceLease for T {}

pub(crate) trait RetainedResources: Send + Sync {}
impl<T: Send + Sync> RetainedResources for T {}

/// One immutable descriptor paired with the allocation that makes it valid.
pub struct BindingLease<A: ResourceLease> {
    descriptor: Binding,
    allocation: Arc<A>,
}

impl<A: ResourceLease> BindingLease<A> {
    pub fn new(descriptor: Binding, allocation: Arc<A>) -> Self {
        Self {
            descriptor,
            allocation,
        }
    }

    pub const fn descriptor(&self) -> Binding {
        self.descriptor
    }

    pub fn allocation(&self) -> &Arc<A> {
        &self.allocation
    }
}

/// Complete ownership package transferred to a queue on submission.
///
/// `I` owns mutable per-invocation state such as command slots, scratch, or import bookkeeping.
/// It must not be shared merely because `program` is resident and reusable.
pub struct OwnedOperation<P, Q, A, I>
where
    P: ResourceLease,
    Q: ResourceLease,
    A: ResourceLease,
    I: ResourceLease,
{
    id: OperationId,
    epoch: DeviceEpoch,
    queue: Arc<Q>,
    program: Arc<P>,
    bindings: Box<[BindingLease<A>]>,
    invocation: I,
    admission_deadline: Deadline,
}

impl<P, Q, A, I> OwnedOperation<P, Q, A, I>
where
    P: ResourceLease,
    Q: ResourceLease,
    A: ResourceLease,
    I: ResourceLease,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: OperationId,
        epoch: DeviceEpoch,
        queue: Arc<Q>,
        program: Arc<P>,
        bindings: Box<[BindingLease<A>]>,
        invocation: I,
        admission_deadline: Deadline,
    ) -> Self {
        Self {
            id,
            epoch,
            queue,
            program,
            bindings,
            invocation,
            admission_deadline,
        }
    }

    pub const fn id(&self) -> OperationId {
        self.id
    }

    pub const fn epoch(&self) -> DeviceEpoch {
        self.epoch
    }

    pub fn queue(&self) -> &Arc<Q> {
        &self.queue
    }

    pub fn program(&self) -> &Arc<P> {
        &self.program
    }

    pub fn bindings(&self) -> &[BindingLease<A>] {
        &self.bindings
    }

    pub fn invocation(&self) -> &I {
        &self.invocation
    }

    pub const fn admission_deadline(&self) -> Deadline {
        self.admission_deadline
    }

    pub(crate) fn into_retained(self) -> Box<dyn RetainedResources> {
        Box::new(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitError {
    InvalidOperation,
    AdmissionTimedOut,
    ResourceLimit,
    DeviceUnavailable,
    RuntimeShuttingDown,
    External { domain: u32, code: i64 },
}

impl fmt::Display for SubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

/// Submission failure preserving the acceptance boundary.
pub enum SubmitFailure<O> {
    /// The provider guarantees that no execution was accepted.
    Rejected { error: SubmitError, operation: O },
    /// Acceptance could not be established. Resources remain in the completion's registry.
    Indeterminate {
        error: SubmitError,
        completion: Completion,
    },
}

impl<O> SubmitFailure<O> {
    pub const fn error(&self) -> SubmitError {
        match self {
            Self::Rejected { error, .. } | Self::Indeterminate { error, .. } => *error,
        }
    }
}

impl<O> fmt::Debug for SubmitFailure<O> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct(match self {
                Self::Rejected { .. } => "Rejected",
                Self::Indeterminate { .. } => "Indeterminate",
            })
            .field("error", &self.error())
            .finish_non_exhaustive()
    }
}

pub enum ExecuteError<O> {
    Submit(SubmitFailure<O>),
    Completion(CompletionError),
}

impl<O> fmt::Debug for ExecuteError<O> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Submit(error) => formatter.debug_tuple("Submit").field(error).finish(),
            Self::Completion(error) => formatter.debug_tuple("Completion").field(error).finish(),
        }
    }
}

/// Typed Rust submission boundary. The returned [`Completion`] is a native owned future.
pub trait Submitter {
    type Operation;

    fn submit(
        &self,
        operation: Self::Operation,
    ) -> Result<Completion, SubmitFailure<Self::Operation>>;
}

/// Queue marker retained for discoverable native async APIs.
pub trait AsyncQueue: Submitter {}
impl<T: Submitter + ?Sized> AsyncQueue for T {}

/// Native `async fn` convenience for statically dispatched queues.
pub async fn submit_and_wait<S>(
    submitter: &S,
    operation: S::Operation,
) -> Result<CompletionInfo, ExecuteError<S::Operation>>
where
    S: Submitter + ?Sized,
{
    submitter
        .submit(operation)
        .map_err(ExecuteError::Submit)?
        .await
        .map_err(ExecuteError::Completion)
}

pub struct BatchSubmitResult<O> {
    pub accepted: Vec<(usize, Completion)>,
    pub rejected: Vec<(usize, SubmitFailure<O>)>,
}

impl<O> BatchSubmitResult<O> {
    pub fn submit_all<S>(submitter: &S, operations: impl IntoIterator<Item = O>) -> Self
    where
        S: Submitter<Operation = O>,
    {
        let mut accepted = Vec::new();
        let mut rejected = Vec::new();
        for (index, operation) in operations.into_iter().enumerate() {
            match submitter.submit(operation) {
                Ok(completion) => accepted.push((index, completion)),
                Err(error) => rejected.push((index, error)),
            }
        }
        Self { accepted, rejected }
    }
}
