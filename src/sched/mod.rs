//! Scheduler helpers living on the NPU side of the C ABI.
//!
//! Graph coloring / split cutting remain in llama.cpp's `ggml_backend_sched`.
//! Same-backend region analysis (bitmat/interval-inspired) and weighted
//! cut-cost gating live here in Rust and are exported for the scheduler via
//! `get_proc_address` — Rust-owned algorithm, thin C ABI only (no thick C++
//! cost model; free to take crates/`core::simd` without ggml C++ compat).

mod auto;
mod cut_cost;
mod region_coalesce;

pub(crate) unsafe fn proc_address(name: *const core::ffi::c_char) -> *mut core::ffi::c_void {
    let sched = unsafe { region_coalesce::proc_address(name) };
    if !sched.is_null() {
        return sched;
    }
    unsafe { auto::proc_address(name) }
}
