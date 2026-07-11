//! The vendor axis. Each NPU family is a module named after its kernel driver, holding its
//! HAL ([`crate::hal::NpuHal`]) and its op lowerings:
//!
//! - [`ivpu`] — Intel NPU (`intel_vpu` / Level Zero + OpenVINO). **Active.**
//! - [`xdna`] — AMD Ryzen AI (`amdxdna` / XRT + mlir-aie). *(HAL skeleton.)*
//! - [`hexagon`] — Qualcomm Hexagon HTP (FastRPC/QNN + hexagon-mlir). *(HAL skeleton.)*
//! - [`stub`] — hardware-free CPU echo used to exercise the seam.
//!
//! The vendor-agnostic backend talks only to the [`Vendor`] trait + the registry here — never a
//! concrete vendor. A vendor becomes active by implementing `Vendor` and joining [`VENDORS`].
// Vendor modules carry aspirational HAL skeletons (unimplemented!() with named params).
#![allow(dead_code, unused_variables)]

pub mod hexagon;
pub mod ivpu;
pub mod stub;
pub mod xdna;

use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::c_void;

use crate::ffi::ggml_tensor;
use crate::graph_ir::GraphIr;
use crate::ops::subgraph::Subgraph;
use crate::ops::{Op, Operand};

/// A discovered NPU device — the agnostic view used for the ggml device description.
pub struct DeviceInfo {
    pub name: String,
}

/// A compiled-in NPU vendor: probe for devices, decide whether it can lower an op (compiling
/// its artifact on a cache miss), and run it. Implementations own their own artifact cache.
pub trait Vendor: Sync {
    /// Short driver-family name, e.g. `"ivpu"`.
    fn name(&self) -> &'static str;
    /// Devices of this vendor present on the host (empty if none / no runtime).
    fn probe(&self) -> Vec<DeviceInfo>;
    /// Can this op run here? (Compiles + caches the artifact on first sight.) `inputs` are the
    /// op's operands in arg order — a vendor whose artifact depends on operand contents (e.g. a
    /// weight baked as a compile-time Constant) needs them to compile; others may ignore them.
    fn can_lower(&self, op: Op, inputs: &[Operand]) -> bool;
    /// Run `op` on the device. `inputs` are the op's operands in arg order, `output` the dst;
    /// each carries whether it lives in NPU memory (bind directly) or must be staged.
    fn run(&self, op: Op, inputs: &[Operand], output: &Operand) -> bool;

    /// Run a recognized fused subgraph as **one** compiled program (weights baked, single submit).
    /// `false` if this vendor can't lower it (caller then dispatches the split node by node).
    fn run_subgraph(&self, sg: &Subgraph) -> bool;

    /// Whether this vendor can compile the complete captured scheduler split. The default keeps
    /// existing per-op vendors working while Graph IR v2 lowerings are brought up incrementally.
    fn can_lower_graph(&self, _graph: &GraphIr) -> bool {
        false
    }

    /// Compile/cache and execute a complete Graph IR split. Called only after
    /// [`Vendor::can_lower_graph`] succeeds.
    fn run_graph(&self, _graph: &GraphIr) -> bool {
        false
    }

    /// Cheap per-node scheduler probe used to form candidate Graph IR splits.
    fn can_lower_graph_node(&self, _node: *const ggml_tensor) -> bool {
        false
    }

    fn synchronize(&self) -> bool {
        true
    }

    fn debug_note(&self, _message: &str) {}

    /// Allocate `size` bytes of NPU-mapped host memory for the backend's buffer type — memory
    /// whose pointer can be bound as a graph argument with **no copy**. Null on failure (the
    /// buffer type then reports OOM rather than hand out unbindable memory).
    unsafe fn alloc_host(&self, size: usize) -> *mut c_void;
    /// Free memory returned by [`alloc_host`](Vendor::alloc_host).
    unsafe fn free_host(&self, ptr: *mut c_void);
}

static IVPU: ivpu::IvpuVendor = ivpu::IvpuVendor;

/// Vendors compiled in for this target. Intel today; AMD/Qualcomm join here (cfg-gated by arch)
/// once they implement [`Vendor`].
static VENDORS: [&'static dyn Vendor; 1] = [&IVPU];

/// All discovered NPU devices across compiled-in vendors (for the ggml device description).
pub fn devices() -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    for &v in VENDORS.iter() {
        out.extend(v.probe());
    }
    out
}

/// The vendor with a device present, if any (realistically one NPU per host).
pub fn present() -> Option<&'static dyn Vendor> {
    for &v in VENDORS.iter() {
        if !v.probe().is_empty() {
            return Some(v);
        }
    }
    None
}
