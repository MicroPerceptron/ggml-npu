//! Scheduler helpers living on the NPU side of the C ABI.
//!
//! Graph coloring / split cutting remain in llama.cpp's `ggml_backend_sched`.
//! Same-backend region analysis (bitmat/interval-inspired) lives here in Rust
//! and is exported for the scheduler to call via `get_proc_address`.

mod region_coalesce;

pub(crate) use region_coalesce::proc_address;
