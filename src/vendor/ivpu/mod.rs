//! Intel NPU (`intel_vpu` driver) — the active bring-up vendor.
//!
//! Runtime: **Level Zero** + the NPU graph extension (`ze_graph_ext.h`). Compile: OpenVINO
//! (`ov::Model` → native blob). Binding model: **INDEXED** — arguments are bound by graph
//! argument index queried from the blob (`pfnSetArgumentValue(graph, index, ptr)`).
//!
//! Layout (mirrors the other vendors):
//! - `level_zero/` — vendored Level Zero headers (`ze_api.h`; `ze_graph_ext.h` to come),
//!   bindgen'd into a `level_zero` bindings module during the L0 bring-up.
//! - `lower` (to come) — `ov::Model` construction / offline blob selection.
//!
//! Status: the OpenVINO offline-compile path and NPU-vs-CPU numeric parity are validated
//! (a fixed-shape MatMul runs on `AI Boost`). Next: bindgen `ze_api.h` + `ze_graph_ext.h`,
//! implement `probe` (zeInit + enumerate the NPU) and `load`/`submit` over the graph
//! extension, then wire into `graph_compute`.

use alloc::vec::Vec;

use crate::hal::{BufFlags, NpuHal};

pub struct IvpuHal; // ze driver/context + graph-ext dispatch table
pub struct IvpuDevice; // ze_device_handle_t (the NPU)
pub struct IvpuQueue; // ze_command_queue_handle_t (+ command list)
pub struct IvpuBinary; // ze_graph_handle_t created from a NATIVE blob
pub struct IvpuBuffer; // zeMemAllocHost / zeMemAllocDevice allocation

impl NpuHal for IvpuHal {
    type Device = IvpuDevice;
    type Queue = IvpuQueue;
    type Binary = IvpuBinary;
    type Buffer = IvpuBuffer;

    fn probe() -> Vec<IvpuDevice> {
        unimplemented!("zeInit -> enumerate drivers/devices, keep ZE_DEVICE_TYPE_VPU")
    }
    fn queue_create(&self, dev: &IvpuDevice) -> IvpuQueue {
        unimplemented!("zeCommandQueueCreate")
    }
    fn load(&self, dev: &IvpuDevice, artifact: &[u8]) -> IvpuBinary {
        unimplemented!("graph ext pfnCreate(blob, ZE_GRAPH_FORMAT_NATIVE)")
    }
    fn alloc(&self, dev: &IvpuDevice, bytes: usize, flags: BufFlags) -> IvpuBuffer {
        unimplemented!("zeMemAllocHost when flags.host_shared, else zeMemAllocDevice")
    }
    fn map<'a>(&self, buf: &'a mut IvpuBuffer) -> &'a mut [u8] {
        unimplemented!("host allocations are already mapped")
    }
    fn sync_to_device(&self, buf: &IvpuBuffer) {
        unimplemented!()
    }
    fn sync_from_device(&self, buf: &IvpuBuffer) {
        unimplemented!()
    }
    fn bind(&self, bin: &mut IvpuBinary, index: u32, buf: &IvpuBuffer) {
        // INDEXED: `index` is the graph argument index queried from the blob.
        unimplemented!("pfnSetArgumentValue(graph, index, ptr)")
    }
    fn submit(&self, q: &IvpuQueue, bin: &IvpuBinary) {
        unimplemented!("pfnAppendGraphExecute + zeCommandQueueExecuteCommandLists")
    }
    fn sync(&self, q: &IvpuQueue) {
        unimplemented!("zeCommandQueueSynchronize")
    }
}
