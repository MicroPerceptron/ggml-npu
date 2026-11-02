//! ggml C ABI adapter to virtio-accel's transport-independent Rust providers.
#![deny(unsafe_op_in_unsafe_fn)]

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code
)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

mod backend;
mod graph;
mod runtime;

const _: () = assert!(ffi::GGML_BACKEND_API_VERSION == 3);

/// Called by ggml's dynamic loader. Registry and device handles live until unload.
#[unsafe(no_mangle)]
pub extern "C" fn ggml_backend_init() -> *mut ffi::ggml_backend_reg {
    backend::registry()
}

#[unsafe(no_mangle)]
pub extern "C" fn ggml_backend_score() -> std::ffi::c_int {
    backend::device_count() as std::ffi::c_int
}
