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
mod profile;
mod session;
mod ze;

use alloc::vec::Vec;
use core::alloc::{AllocError, Allocator, Layout};
use core::ffi::c_void;
use core::ptr::NonNull;

use crate::hal::{BufFlags, NpuHal};
use crate::ops::{Op, Operand};
use crate::vendor::{DeviceInfo, Vendor};

pub use ze::Device as IvpuDevice;

/// [`Allocator`] over NPU-MMU-mapped host memory (`zeMemAllocHost`), so NPU-resident buffers are
/// first-class `Vec<u8, NpuHostAlloc>` instead of the raw-pointer + length bookkeeping the session
/// does today (issue #1). The driver page-aligns (4096), satisfying any alignment we bind. Returns
/// [`AllocError`] when the Level Zero runtime is absent (no NPU) or the request is over-aligned —
/// a caller then falls back to [`crate::HostAlloc`]. ZST: free to pass by value.
#[derive(Clone, Copy, Default)]
pub(super) struct NpuHostAlloc;

unsafe impl Allocator for NpuHostAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            let d = unsafe { NonNull::new_unchecked(layout.align() as *mut u8) };
            return Ok(NonNull::slice_from_raw_parts(d, 0));
        }
        if layout.align() > 4096 {
            return Err(AllocError);
        }
        let p = ze::alloc_host(layout.size()) as *mut u8;
        NonNull::new(p)
            .map(|p| NonNull::slice_from_raw_parts(p, layout.size()))
            .ok_or(AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            ze::free_host(ptr.as_ptr() as *mut c_void);
        }
    }
}

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

    fn can_lower(&self, op: Op, inputs: &[Operand]) -> bool {
        match op {
            Op::MatMul { m, k, n } => lower::matmul::can_lower(m, k, n, inputs),
        }
    }

    fn run(&self, op: Op, inputs: &[Operand], output: &Operand) -> bool {
        match op {
            Op::MatMul { m, k, n } => lower::matmul::run(m, k, n, inputs, output),
        }
    }

    fn run_subgraph(&self, sg: &crate::ops::subgraph::Subgraph) -> bool {
        lower::subgraph::run(sg)
    }

    fn can_lower_graph(&self, graph: &crate::graph_ir::GraphIr) -> bool {
        lower::graph::can_lower(graph)
    }

    fn run_graph(&self, graph: &crate::graph_ir::GraphIr) -> bool {
        lower::graph::run(graph)
    }

    fn can_lower_graph_node(&self, node: *const crate::ffi::ggml_tensor) -> bool {
        unsafe { compile::graph_node_feasible(node) }
    }

    fn synchronize(&self) -> bool {
        session::synchronize()
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
