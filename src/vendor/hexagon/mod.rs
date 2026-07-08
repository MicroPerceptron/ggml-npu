//! Qualcomm Hexagon HTP (FastRPC / QNN) — skeleton; the aarch64 entry point.
//!
//! Runtime: **FastRPC** to the DSP, wrapped by QNN graph execute. Compile: `hexagon-mlir`
//! (our `qualcomm-npu-compiler` fork) → QNN context binary. Binding model: **NAMED** — QNN
//! binds by graph tensor *name*, not index. This is the asymmetry the pressure test exists
//! to surface: `bind(index, buf)` still fits, but only because the lowering commits to a
//! stable tensor order per artifact so the binder can resolve `index -> tensor_names[index]`.
//!
//! Also forces cross-ISA discipline: the host is aarch64, so this path must build cleanly on
//! aarch64 from Phase 0 (enforced by CI). Layout mirrors [`crate::vendor::ivpu`].

use alloc::string::String;
use alloc::vec::Vec;

use crate::hal::{BufFlags, NpuHal};

pub struct HexagonHal; // QnnInterface + FastRPC session
pub struct HexagonDevice; // HTP/DSP handle
pub struct HexagonQueue; // logical; QNN graph execute is synchronous over FastRPC

pub struct HexagonBinary {
    /// The lowering emits graph tensors in a fixed order; `bind(index)` resolves to
    /// `tensor_names[index]`. This is what lets a NAMED API sit behind an indexed verb.
    pub tensor_names: Vec<String>,
}

pub struct HexagonBuffer; // rpcmem/ION (dma-buf), host<->DSP shared

impl NpuHal for HexagonHal {
    type Device = HexagonDevice;
    type Queue = HexagonQueue;
    type Binary = HexagonBinary;
    type Buffer = HexagonBuffer;

    fn probe() -> Vec<HexagonDevice> {
        unimplemented!("QnnBackend_create + HTP device over FastRPC")
    }
    fn queue_create(&self, dev: &HexagonDevice) -> HexagonQueue {
        unimplemented!()
    }
    fn load(&self, dev: &HexagonDevice, artifact: &[u8]) -> HexagonBinary {
        unimplemented!("qnnContextCreateFromBinary / register DSP object over FastRPC")
    }
    fn alloc(&self, dev: &HexagonDevice, bytes: usize, flags: BufFlags) -> HexagonBuffer {
        unimplemented!("rpcmem_alloc (ION/dma-buf), host<->DSP shared")
    }
    fn map<'a>(&self, buf: &'a mut HexagonBuffer) -> &'a mut [u8] {
        unimplemented!("rpcmem is host-mapped")
    }
    fn sync_to_device(&self, buf: &HexagonBuffer) {
        unimplemented!()
    }
    fn sync_from_device(&self, buf: &HexagonBuffer) {
        unimplemented!()
    }
    fn bind(&self, bin: &mut HexagonBinary, index: u32, buf: &HexagonBuffer) {
        // NAMED: `index` resolves to a graph tensor name via the lowering's fixed order.
        let _name = &bin.tensor_names[index as usize];
        unimplemented!("set QNN graph tensor `_name` to the rpcmem buffer")
    }
    fn submit(&self, q: &HexagonQueue, bin: &HexagonBinary) {
        unimplemented!("qnnGraphExecute / FastRPC invoke")
    }
    fn sync(&self, q: &HexagonQueue) {
        unimplemented!("FastRPC returns (blocking)")
    }
}
