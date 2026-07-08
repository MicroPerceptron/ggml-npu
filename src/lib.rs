//! ggml-npu — a universal NPU backend for ggml. **Phase 0 skeleton, `no_std`.**
//!
//! Registers as a dynamically-loaded ggml `ACCEL` backend, claims **no** operations
//! (`supports_op == false`) so the scheduler runs the whole model on CPU, and stands up
//! the vendor-agnostic [`hal::NpuHal`] seam. No NPU is touched.
//!
//! `no_std` by design: a ggml backend is always `dlopen`'d into a full libc host, so the
//! allocator and `abort` resolve from the host at load time (`RTLD_NOW`). There is no
//! unwinding runtime — a panic in any vtable entry hits [`on_panic`] and aborts, which is
//! UB-free across the C ABI (the `no_std` replacement for a `catch_unwind` firewall).
//!
//! Load it with `GGML_BACKEND_PATH=/abs/path/libggml_npu.so`.
#![no_std]
#![feature(allocator_api)]

extern crate alloc;

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::{c_int, c_void};
use core::panic::PanicInfo;

/// Raw ggml C-ABI bindings, generated at build time by bindgen (see build.rs). `use_core`
/// keeps them `no_std` (`::core::ffi` rather than `::std::os::raw`).
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code
)]
pub(crate) mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

mod backend;
mod hal;
mod vendor;

// ---------------------------------------------------------------------------
// Host runtime glue. These symbols resolve from the host process at dlopen.
// ---------------------------------------------------------------------------
extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
    fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> c_int;
    fn abort() -> !;
}

/// Global allocator forwarding to the host libc. `malloc` is `max_align_t`-aligned
/// (16 on our targets); over-aligned requests go through `posix_memalign`.
struct HostAlloc;

unsafe impl GlobalAlloc for HostAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= 16 {
            malloc(layout.size()) as *mut u8
        } else {
            let mut p: *mut c_void = core::ptr::null_mut();
            if posix_memalign(&mut p, layout.align(), layout.size()) == 0 {
                p as *mut u8
            } else {
                core::ptr::null_mut()
            }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        free(ptr as *mut c_void);
    }
}

#[global_allocator]
static HOST_ALLOC: HostAlloc = HostAlloc;

#[panic_handler]
fn on_panic(_info: &PanicInfo) -> ! {
    // no_std: no unwinding. Abort rather than let a panic cross the C ABI.
    unsafe { abort() }
}

// ---------------------------------------------------------------------------
// DL entry points — the whole binary contract.
// ---------------------------------------------------------------------------

/// ggml `dlopen`s this cdylib, calls this, and checks
/// `reg.api_version == GGML_BACKEND_API_VERSION` (2).
#[no_mangle]
pub extern "C" fn ggml_backend_init() -> *mut ffi::ggml_backend_reg {
    backend::registry()
}

/// Optional support probe; non-zero = usable. The CPU-fallback skeleton is always up.
#[no_mangle]
pub extern "C" fn ggml_backend_score() -> c_int {
    1
}
