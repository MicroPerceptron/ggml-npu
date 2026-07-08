//! The universal NPU submission seam.
//!
//! Half the project's value is that all three target NPUs collapse onto **five verbs**:
//! `load`, `alloc`, `bind`, `submit`, `sync`. The submission surface unifies even though
//! the compilers do not. One implementation per vendor under [`crate::vendor`], each
//! keeping its own native handles via the associated types (no boxing on the hot path).
#![allow(dead_code)]

use alloc::vec::Vec;

/// Allocation hints. On integrated NPUs the host↔device boundary copy is the main tax,
/// so a host-shared (zero-copy) allocation is the default preference.
#[derive(Clone, Copy, Debug, Default)]
pub struct BufFlags {
    /// Prefer a host-shared / mappable allocation where the device supports it
    /// (XRT `HOST_ONLY` BO, L0 host memory, Qualcomm rpcmem/ION).
    pub host_shared: bool,
}

/// The unified submission surface. Handles are associated types so each vendor keeps its
/// own native objects. Device buffers are allocator-parameterizable (`allocator_api`) so a
/// vendor can back them with a device allocator instead of the global one.
pub trait NpuHal {
    type Device;
    type Queue;
    /// A loaded compiled artifact (xclbin+insts | L0 graph | QNN context).
    type Binary;
    /// A device-visible buffer (xrt::bo | zeMemAlloc | rpcmem).
    type Buffer;

    /// Enumerate NPUs of this vendor.
    fn probe() -> Vec<Self::Device>;
    fn queue_create(&self, dev: &Self::Device) -> Self::Queue;

    /// LOAD (once, off the hot path): register xclbin / create L0 graph / QNN context.
    fn load(&self, dev: &Self::Device, artifact: &[u8]) -> Self::Binary;

    /// ALLOC: prefer host-shared for zero-copy (see [`BufFlags::host_shared`]).
    fn alloc(&self, dev: &Self::Device, bytes: usize, flags: BufFlags) -> Self::Buffer;
    /// Map a device buffer into host address space (if mappable).
    fn map<'a>(&self, buf: &'a mut Self::Buffer) -> &'a mut [u8];
    fn sync_to_device(&self, buf: &Self::Buffer);
    fn sync_from_device(&self, buf: &Self::Buffer);

    /// BIND argument `index`: positional kernel arg (XRT) | graph-arg index (L0) | named
    /// tensor (QNN). The lowering must emit the arg order/names the binder expects.
    fn bind(&self, bin: &mut Self::Binary, index: u32, buf: &Self::Buffer);

    /// SUBMIT: run.start(opcode=3) | AppendGraphExecute+Execute | qnnGraphExecute.
    fn submit(&self, q: &Self::Queue, bin: &Self::Binary);

    /// SYNC: run.wait() | zeCommandQueueSynchronize | FastRPC return.
    fn sync(&self, q: &Self::Queue);
}
