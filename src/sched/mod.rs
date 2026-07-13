//! Scheduler helpers living on the NPU side of the C ABI.
//!
//! Graph coloring / split cutting remain in llama.cpp's `ggml_backend_sched`.
//! Same-backend region analysis (bitmat/interval-inspired) and weighted
//! cut-cost gating live here in Rust and are exported for the scheduler via
//! `get_proc_address` — Rust-owned algorithm, thin C ABI only (no thick C++
//! cost model; free to take crates/`core::simd` without ggml C++ compat).

mod cut_cost;
mod region_coalesce;

pub(crate) use region_coalesce::proc_address;
