use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use super::{Completion, CompletionError};

/// Generic visibility domains only. Cache maintenance and provider fence rules stay adapter-owned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisibilityDomain {
    Host,
    Device,
    Coherent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisibilityState {
    Transitioning,
    Visible(VisibilityDomain),
    Quarantined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnershipState {
    Transitioning,
    Owned,
    Quarantined,
}

pub struct VisibilityLease<L> {
    lease: L,
    domain: VisibilityDomain,
}

impl<L> VisibilityLease<L> {
    pub fn new(lease: L, domain: VisibilityDomain) -> Self {
        Self { lease, domain }
    }

    pub const fn domain(&self) -> VisibilityDomain {
        self.domain
    }

    pub fn into_inner(self) -> L {
        self.lease
    }
}

/// Ownership marker `O` is adapter-defined; no foreign handle representation is generalized.
pub struct OwnershipLease<L, O> {
    lease: L,
    owner: O,
}

impl<L, O> OwnershipLease<L, O> {
    pub fn new(lease: L, owner: O) -> Self {
        Self { lease, owner }
    }

    pub fn owner(&self) -> &O {
        &self.owner
    }

    pub fn into_parts(self) -> (L, O) {
        (self.lease, self.owner)
    }
}

pub struct QuarantinedLease<L> {
    lease: L,
    cause: CompletionError,
}

impl<L> QuarantinedLease<L> {
    pub const fn cause(&self) -> CompletionError {
        self.cause
    }

    /// Recovery is deliberately explicit and adapter-owned.
    pub fn into_recovery_parts(self) -> (L, CompletionError) {
        (self.lease, self.cause)
    }
}

/// Future for a provider-supplied visibility operation.
pub struct VisibilityTransition<L> {
    completion: Completion,
    lease: Option<L>,
    target: VisibilityDomain,
}

impl<L> VisibilityTransition<L> {
    pub fn new(lease: L, target: VisibilityDomain, completion: Completion) -> Self {
        Self {
            completion,
            lease: Some(lease),
            target,
        }
    }

    pub const fn state(&self) -> VisibilityState {
        if self.lease.is_some() {
            VisibilityState::Transitioning
        } else {
            // The output owns the final state once this future resolves.
            VisibilityState::Quarantined
        }
    }
}

impl<L: Unpin> Future for VisibilityTransition<L> {
    type Output = Result<VisibilityLease<L>, QuarantinedLease<L>>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let result = match Pin::new(&mut self.completion).poll(context) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(result) => result,
        };
        let lease = self
            .lease
            .take()
            .expect("visibility transition polled after completion");
        Poll::Ready(match result {
            Ok(_) => Ok(VisibilityLease::new(lease, self.target)),
            Err(cause) => Err(QuarantinedLease { lease, cause }),
        })
    }
}

/// Future for an adapter-defined ownership handoff.
pub struct OwnershipTransition<L, O> {
    completion: Completion,
    lease: Option<L>,
    target_owner: Option<O>,
}

impl<L, O> OwnershipTransition<L, O> {
    pub fn new(lease: L, target_owner: O, completion: Completion) -> Self {
        Self {
            completion,
            lease: Some(lease),
            target_owner: Some(target_owner),
        }
    }

    pub const fn state(&self) -> OwnershipState {
        if self.lease.is_some() {
            OwnershipState::Transitioning
        } else {
            OwnershipState::Quarantined
        }
    }
}

impl<L: Unpin, O: Unpin> Future for OwnershipTransition<L, O> {
    type Output = Result<OwnershipLease<L, O>, QuarantinedLease<L>>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let result = match Pin::new(&mut self.completion).poll(context) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(result) => result,
        };
        let lease = self
            .lease
            .take()
            .expect("ownership transition polled after completion");
        Poll::Ready(match result {
            Ok(_) => Ok(OwnershipLease::new(
                lease,
                self.target_owner
                    .take()
                    .expect("ownership target consumed once"),
            )),
            Err(cause) => {
                self.target_owner.take();
                Err(QuarantinedLease { lease, cause })
            }
        })
    }
}
