//! AMD Ryzen AI NPU (`amdxdna` driver) — skeleton until the SSH box is set up.
//!
//! Runtime: **XRT** (`xrt::device/hw_context/kernel/bo/run`). Compile: `mlir-aie` (IRON) →
//! `xclbin` + `insts.bin`. Binding model: **POSITIONAL** — kernel args are ordered and
//! coupled to the compiled kernel's arg order + `group_id`s; submit is
//! `kernel(opcode=3, bo_instr, len, …)`.
//!
//! Layout mirrors [`crate::vendor::ivpu`]: an `xrt/` FFI dir and a `lower` module join this
//! `NpuHal` impl when bring-up starts.

use alloc::vec::Vec;

use crate::hal::{BufFlags, NpuHal};

pub struct XdnaHal; // xrt::device + xrt::hw_context
pub struct XdnaDevice; // xrt::device
pub struct XdnaQueue; // logical; XRT runs are per-kernel
pub struct XdnaBinary; // xrt::xclbin + insts BO + xrt::kernel
pub struct XdnaBuffer; // xrt::bo (HOST_ONLY on integrated parts for zero-copy)

impl NpuHal for XdnaHal {
    type Device = XdnaDevice;
    type Queue = XdnaQueue;
    type Binary = XdnaBinary;
    type Buffer = XdnaBuffer;

    fn probe() -> Vec<XdnaDevice> {
        unimplemented!("enumerate xrt::device")
    }
    fn queue_create(&self, dev: &XdnaDevice) -> XdnaQueue {
        unimplemented!()
    }
    fn load(&self, dev: &XdnaDevice, artifact: &[u8]) -> XdnaBinary {
        unimplemented!("register_xclbin + read insts.bin into a cacheable BO")
    }
    fn alloc(&self, dev: &XdnaDevice, bytes: usize, flags: BufFlags) -> XdnaBuffer {
        unimplemented!("xrt::bo; HOST_ONLY when flags.host_shared")
    }
    fn map<'a>(&self, buf: &'a mut XdnaBuffer) -> &'a mut [u8] {
        unimplemented!("bo.map()")
    }
    fn sync_to_device(&self, buf: &XdnaBuffer) {
        unimplemented!("bo.sync(XCL_BO_SYNC_BO_TO_DEVICE)")
    }
    fn sync_from_device(&self, buf: &XdnaBuffer) {
        unimplemented!("bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE)")
    }
    fn bind(&self, bin: &mut XdnaBinary, index: u32, buf: &XdnaBuffer) {
        // POSITIONAL: `index` is the kernel arg slot (coupled to group_id ordering).
        unimplemented!("run.set_arg(index, bo)")
    }
    fn submit(&self, q: &XdnaQueue, bin: &XdnaBinary) {
        unimplemented!("kernel(3, bo_instr, len, …) -> run.start()")
    }
    fn sync(&self, q: &XdnaQueue) {
        unimplemented!("run.wait()")
    }
}
