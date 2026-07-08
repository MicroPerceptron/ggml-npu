//! Intel NPU (`intel_vpu` driver) — the active vendor.
//!
//! Runtime: **Level Zero** ([`ze`], [`graph`]). Compile: OpenVINO via the shim, cached
//! per-shape ([`compile`]). Op lowerings live under [`lower`]. [`IvpuVendor`] is the
//! [`crate::vendor::Vendor`] impl the agnostic backend dispatches to; [`IvpuHal`] is the
//! (still-aspirational) 5-verb submission seam.

mod cache;
mod compile;
mod graph;
mod lower;
mod session;
mod ze;

use alloc::vec::Vec;
use core::ffi::c_void;

use crate::hal::{BufFlags, NpuHal};
use crate::ops::{Op, Operand};
use crate::vendor::{DeviceInfo, Vendor};

pub use ze::Device as IvpuDevice;

/// The Intel NPU vendor: probe via Level Zero, lower/run ops via the JIT cache + graph submit.
pub struct IvpuVendor;

impl Vendor for IvpuVendor {
    fn name(&self) -> &'static str {
        "ivpu"
    }

    fn probe(&self) -> Vec<DeviceInfo> {
        // The cached runtime (created once) holds the device; deriving the name from it means
        // supports_op no longer re-enumerates Level Zero on every call.
        let mut v = Vec::new();
        if let Some(r) = ze::runtime() {
            v.push(DeviceInfo { name: r.name.clone() });
        }
        v
    }

    fn can_lower(&self, op: Op) -> bool {
        match op {
            Op::MatMul { m, k, n } => compile::get_or_compile_matmul(m, k, n).is_some(),
        }
    }

    fn run(&self, op: Op, inputs: &[Operand], output: &Operand) -> bool {
        match op {
            Op::MatMul { m, k, n } => lower::matmul::run(m, k, n, inputs, output),
        }
    }

    unsafe fn alloc_host(&self, size: usize) -> *mut c_void {
        ze::alloc_host(size)
    }

    unsafe fn free_host(&self, ptr: *mut c_void) {
        ze::free_host(ptr)
    }
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
