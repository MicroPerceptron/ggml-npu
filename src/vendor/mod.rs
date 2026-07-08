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

use crate::ops::Op;

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
    /// Can this op run here? (Compiles + caches the artifact on first sight.)
    fn can_lower(&self, op: Op) -> bool;
    /// Run `op` with operands already extracted (in the op's arg order) into `inputs`/`output`.
    fn run(&self, op: Op, inputs: &[&[f32]], output: &mut [f32]) -> bool;
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
