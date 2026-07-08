//! Intel NPU (`intel_vpu` driver) — the active bring-up vendor.
//!
//! Runtime: **Level Zero** ([`ze`]) + the NPU graph extension (`ze_graph_ext.h`, to come).
//! Compile: OpenVINO (`ov::Model` → native blob). Binding model: **INDEXED** — arguments
//! are bound by graph argument index queried from the blob.
//!
//! Status: `probe` is live (dlopen Level Zero, enumerate the VPU device). Next:
//! `ze_graph_ext.h` bindings, then `load`/`submit` over the graph extension and wiring the
//! first `MUL_MAT` into `graph_compute`. The other verbs are `unimplemented!()` until then.

mod graph;
mod ze;

use alloc::vec::Vec;

use crate::hal::{BufFlags, NpuHal};

pub use ze::Device as IvpuDevice;

/// Precompiled shape-generic MatMul blob for `[M=1, K=64, N=64]` (weights are a graph
/// input, so one blob serves any weights of this shape). Graph input args are ordered
/// `[activation, weight]`; see `ov_matmul_ggml.cpp`.
static MATMUL_1X64X64: &[u8] = include_bytes!("artifacts/matmul_1x64x64.blob");

/// Run a ggml `MUL_MAT` of the supported fixed shape on the NPU: `weight` is src0
/// (`[N,K]=[64,64]`), `activation` is src1 (`[K,M]=[64,1]`), result into `out`
/// (`[N,M]=[64,1]`). Returns true on success. Feeds `[activation, weight]` in arg order.
///
/// TODO: this re-probes and re-creates the graph every call; cache the device + graph
/// handle, and route weights through a persistent buffer once the JIT cache lands.
pub fn run_mul_mat(weight: &[f32], activation: &[f32], out: &mut [f32]) -> bool {
    let devices = <IvpuHal as NpuHal>::probe();
    let Some(dev) = devices.first() else {
        return false;
    };
    graph::run_native_blob(dev, MATMUL_1X64X64, &[activation, weight], out).is_ok()
}

/// Bring-up self-test: run a native (OpenVINO-compiled) blob on the first discovered NPU,
/// feeding `input` (f32) and writing the first output into `output` (f32). Returns the
/// number of outputs written, or a negative error. Scaffold for the submit path.
pub fn run_blob_selftest(blob: &[u8], input: &[f32], output: &mut [f32]) -> i32 {
    let devices = <IvpuHal as NpuHal>::probe();
    let Some(dev) = devices.first() else {
        return -100;
    };
    match graph::run_native_blob(dev, blob, &[input], output) {
        Ok(n) => n as i32,
        Err(code) if code > 0 => -code,
        Err(0) => -1,
        Err(code) => code,
    }
}

pub struct IvpuHal; // ze driver/context + graph-ext dispatch table
pub struct IvpuQueue; // ze_command_queue_handle_t (+ command list)
pub struct IvpuBinary; // ze_graph_handle_t created from a NATIVE blob
pub struct IvpuBuffer; // zeMemAllocHost / zeMemAllocDevice allocation

impl NpuHal for IvpuHal {
    type Device = IvpuDevice;
    type Queue = IvpuQueue;
    type Binary = IvpuBinary;
    type Buffer = IvpuBuffer;

    fn probe() -> Vec<IvpuDevice> {
        ze::probe()
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
