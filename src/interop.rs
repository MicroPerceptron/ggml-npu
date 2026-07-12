//! Minimal shared-allocation semantics proven across runtime adapters.
//!
//! A shared handle proves allocation identity, not visibility or ordering. Handle types,
//! import/export rules, and synchronization protocols remain adapter-owned until their semantics
//! are proven across multiple implementations.

#![allow(dead_code)]

use core::ffi::{c_void, CStr};
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

impl AllocationIdentity {
    pub const fn external_id(self) -> u64 {
        self.provider.rotate_left(17) ^ self.object
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationOwnership {
    Backend,
    External,
    Uncertain,
    Released,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnershipError {
    WrongOwner,
    Uncertain,
    Released,
}

#[derive(Clone, Copy)]
pub struct OpaqueExternalHandle<'a> {
    pub type_name: &'static CStr,
    pub data: *const c_void,
    pub size: usize,
    pub _owner: core::marker::PhantomData<&'a ()>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationToken {
    pub provider: u64,
    pub owner: NonNull<c_void>,
}

pub trait SharedAllocationOwner {
    fn address(&self) -> NonNull<u8>;
    fn len(&self) -> usize;
    fn alignment(&self) -> usize;
    fn identity(&self) -> AllocationIdentity;
    fn token(&self) -> AllocationToken;
    fn external_handle(&self) -> OpaqueExternalHandle<'_>;
    fn ownership(&self) -> AllocationOwnership;
    fn acquire_backend(&self) -> Result<(), OwnershipError>;
    fn release_external(&self) -> Result<(), OwnershipError>;
    fn mark_uncertain(&self);
}

#[derive(Clone, Copy, Debug)]
pub struct SharedAllocationView {
    pub identity: AllocationIdentity,
    pub token: AllocationToken,
    pub allocation_bytes: usize,
    pub offset: usize,
    pub bytes: usize,
    pub alignment: usize,
    pub address: NonNull<u8>,
}

pub trait SharedMapping {
    fn address(&self) -> NonNull<u8>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn access(&self) -> AccessMode;
}
