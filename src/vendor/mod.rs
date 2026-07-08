//! Per-vendor [`crate::hal::NpuHal`] implementations, one module per NPU family, named
//! after its kernel driver so the layout matches the platform vocabulary:
//!
//! - [`ivpu`] — Intel NPU (`intel_vpu` / Level Zero + OpenVINO). **Active bring-up.**
//! - [`xdna`] — AMD Ryzen AI (`amdxdna` / XRT + mlir-aie).
//! - [`hexagon`] — Qualcomm Hexagon HTP (FastRPC/QNN + hexagon-mlir).
//! - [`stub`] — hardware-free CPU echo used to exercise the seam.
//!
//! Every vendor module mirrors the same internal layout: a top-level `NpuHal` impl plus
//! submodules for its FFI bindings and its lowering.
#![allow(dead_code, unused_variables)]

pub mod hexagon;
pub mod ivpu;
pub mod stub;
pub mod xdna;

use alloc::string::String;

use crate::hal::NpuHal;

/// The NPU families this backend targets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Vendor {
    /// Intel NPU (Lunar Lake / Meteor Lake / Arrow Lake), x86_64 host.
    Ivpu,
    /// AMD Ryzen AI (XDNA / XDNA2), x86_64 host.
    Xdna,
    /// Qualcomm Snapdragon X (Hexagon HTP), aarch64 host.
    Hexagon,
}

/// Number of NPUs discovered across all supported vendors. Currently probes Intel only;
/// AMD/Qualcomm join here as they come online.
pub fn npu_count() -> usize {
    ivpu::IvpuHal::probe().len()
}

/// Human-readable name of the primary discovered NPU, if any (for the ggml device
/// description).
pub fn primary_npu_name() -> Option<String> {
    ivpu::IvpuHal::probe().into_iter().next().map(|d| d.name)
}
