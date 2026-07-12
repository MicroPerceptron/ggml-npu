//! Minimal shared-allocation semantics proven across runtime adapters.
//!
//! A shared handle proves allocation identity, not visibility or ordering. Handle types,
//! import/export rules, and synchronization protocols remain adapter-owned until their semantics
//! are proven across multiple implementations.

#![allow(dead_code)]

use core::ptr::NonNull;

use crate::runtime::AccessMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocationIdentity {
    /// Stable namespace for one allocator/provider implementation.
    pub provider: u64,
    /// Provider-local backing-object identity (not a process-local pointer or file descriptor).
    pub object: u64,
    /// Increments when an object identity can be recycled.
    pub generation: u64,
}

pub trait SharedMapping {
    fn address(&self) -> NonNull<u8>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn access(&self) -> AccessMode;
}
