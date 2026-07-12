//! Legacy five-verb compatibility façade.
//!
//! Compatibility adapters still reach this API. New runtimes should implement the explicit
//! ownership and lifecycle contracts in [`crate::runtime::RuntimeHal`]; this trait remains while
//! the current backend is adapted incrementally.
//!
//! The original model collapses target NPUs onto **five verbs**:
//! `load`, `alloc`, `bind`, `submit`, `sync`. The submission surface unifies even though
//! compiler and execution adapters do not. Implementations keep their native handles in associated
//! types, without boxing the hot path.
#![allow(dead_code)]

use alloc::vec::Vec;

/// Compatibility allocation hints.
#[derive(Clone, Copy, Debug, Default)]
pub struct BufFlags {
    /// Prefer an allocation visible to both host and accelerator where supported.
    pub host_shared: bool,
}

/// The compatibility submission surface. Associated types retain provider-native ownership.
pub trait NpuHal {
    type Device;
    type Queue;
    /// A loaded compiled artifact.
    type Binary;
    /// An accelerator-visible buffer.
    type Buffer;

    /// Enumerate NPUs of this vendor.
    fn probe() -> Vec<Self::Device>;
    fn queue_create(&self, dev: &Self::Device) -> Self::Queue;

    /// Load one compiled artifact off the hot path.
    fn load(&self, dev: &Self::Device, artifact: &[u8]) -> Self::Binary;

    /// ALLOC: prefer host-shared for zero-copy (see [`BufFlags::host_shared`]).
    fn alloc(&self, dev: &Self::Device, bytes: usize, flags: BufFlags) -> Self::Buffer;
    /// Map a device buffer into host address space (if mappable).
    fn map<'a>(&self, buf: &'a mut Self::Buffer) -> &'a mut [u8];
    fn sync_to_device(&self, buf: &Self::Buffer);
    fn sync_from_device(&self, buf: &Self::Buffer);

    /// Bind a buffer to the artifact's positional argument contract.
    fn bind(&self, bin: &mut Self::Binary, index: u32, buf: &Self::Buffer);

    /// Submit one execution.
    fn submit(&self, q: &Self::Queue, bin: &Self::Binary);

    /// Wait for prior submissions.
    fn sync(&self, q: &Self::Queue);
}
