//! Intel NPU (`intel_vpu` driver) — the active bring-up vendor.
//!
//! Runtime: **Level Zero** ([`ze`]) + the NPU graph extension (`ze_graph_ext.h`, to come).
//! Compile: OpenVINO (`ov::Model` → native blob). Binding model: **INDEXED** — arguments
//! are bound by graph argument index queried from the blob.
//!
//! Status: `probe` is live (dlopen Level Zero, enumerate the VPU device). Next:
//! `ze_graph_ext.h` bindings, then `load`/`submit` over the graph extension and wiring the
//! first `MUL_MAT` into `graph_compute`. The other verbs are `unimplemented!()` until then.

mod compile;
mod graph;
mod ze;

use alloc::vec::Vec;

use crate::hal::{BufFlags, NpuHal};

pub use ze::Device as IvpuDevice;

/// Can we run a ggml `MUL_MAT` of shape `(M, K, N)` on the NPU? True iff a blob for it is
/// available (compiled on first sight, then cached). Called from `supports_op`.
pub fn can_matmul(m: i64, k: i64, n: i64) -> bool {
    compile::get_or_compile_matmul(m, k, n).is_some()
}

/// Run a ggml `MUL_MAT` of shape `(M, K, N)` on the NPU: `weight` is src0 (`[K,N]`),
/// `activation` is src1 (`[K,M]`), result into `out` (`[N,M]`). Fetches the shape's blob from
/// the JIT cache and feeds `[activation, weight]` in graph-arg order. Returns true on success.
///
/// TODO: re-probes the device and re-creates the graph every call; cache the device + a graph
/// handle per shape, and route weights through a persistent device buffer.
pub fn run_matmul(m: i64, k: i64, n: i64, weight: &[f32], activation: &[f32], out: &mut [f32]) -> bool {
    let Some(blob) = compile::get_or_compile_matmul(m, k, n) else {
        return false;
    };
    let devices = <IvpuHal as NpuHal>::probe();
    let Some(dev) = devices.first() else {
        return false;
    };
    graph::run_native_blob(dev, blob, &[activation, weight], out).is_ok()
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
