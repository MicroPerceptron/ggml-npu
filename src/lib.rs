//! ggml-npu — an experimental universal NPU backend for ggml, `no_std`.
//!
//! Registers as a dynamically-loaded ggml `ACCEL` backend. The active Intel implementation
//! recognizes constant-weight matmuls and fused subgraphs, compiles them through OpenVINO, and
//! submits native graphs through Level Zero. Unsupported work remains on other ggml backends.
//!
//! `no_std` by design: a ggml backend is always `dlopen`'d into a full libc host, so the
//! allocator and `abort` resolve from the host at load time (`RTLD_NOW`). There is no
//! unwinding runtime — a panic in any vtable entry hits [`on_panic`] and aborts, which is
//! UB-free across the C ABI (the `no_std` replacement for a `catch_unwind` firewall).
//!
//! Load it with `GGML_BACKEND_PATH=/abs/path/libggml_npu.so`.
#![no_std]
#![feature(allocator_api)]
#![feature(f16)]

extern crate alloc;

#[cfg(test)]
extern crate std;

use core::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
use core::ffi::{c_int, c_void};
#[cfg(not(test))]
use core::panic::PanicInfo;
use core::ptr::NonNull;

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
mod engine;
mod graph_ir;
mod hal;
mod ops;
mod policy;
mod trace;
mod vendor;

// ---------------------------------------------------------------------------
// Host runtime glue. These symbols resolve from the host process at dlopen.
// ---------------------------------------------------------------------------
extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
    fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> c_int;
    #[cfg(not(test))]
    fn abort() -> !;
}

/// The host libc allocator: `malloc`/`free` (`max_align_t`-aligned, 16 on our targets;
/// over-aligned requests go through `posix_memalign`). Serves as **both** the global hook
/// ([`GlobalAlloc`]) and a first-class allocator value ([`Allocator`]) so owning types can be
/// constructed with an explicit allocator — `Vec::new_in(HostAlloc)` — rather than the implicit
/// `Global` (issue #1). A zero-size ZST, so passing it by value is free.
#[derive(Clone, Copy, Default)]
pub(crate) struct HostAlloc;

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

// The `Allocator` (not `GlobalAlloc`) view of the same libc heap, for explicit `*_in` construction.
unsafe impl Allocator for HostAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            // A valid, aligned, non-null dangling pointer for a zero-size request (never read).
            let d = unsafe { NonNull::new_unchecked(layout.align() as *mut u8) };
            return Ok(NonNull::slice_from_raw_parts(d, 0));
        }
        let p = unsafe { <Self as GlobalAlloc>::alloc(self, layout) };
        NonNull::new(p)
            .map(|p| NonNull::slice_from_raw_parts(p, layout.size()))
            .ok_or(AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            <Self as GlobalAlloc>::dealloc(self, ptr.as_ptr(), layout);
        }
    }
}

#[global_allocator]
static HOST_ALLOC: HostAlloc = HostAlloc;

#[cfg(not(test))]
#[panic_handler]
fn on_panic(_info: &PanicInfo) -> ! {
    // no_std: no unwinding. Abort rather than let a panic cross the C ABI.
    unsafe { abort() }
}

// The precompiled `core`/`alloc` are built with unwinding and reference these symbols,
// even though `panic = "abort"` means we never unwind. Provide self-contained no-op shims
// so the cdylib does not depend on the host's libgcc; neither is ever actually called (a
// panic aborts in `on_panic` first). A leaner alternative is `-Zbuild-std` with
// `panic_immediate_abort`, but that would force an explicit `--target` on every build.
#[no_mangle]
#[cfg(not(test))]
extern "C" fn rust_eh_personality() {}

#[no_mangle]
#[cfg(not(test))]
extern "C" fn _Unwind_Resume() -> ! {
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

/// Optional support probe; non-zero = usable. Reports the number of NPUs discovered
/// (Level Zero VPU devices), so the loader gates on real hardware being present.
#[no_mangle]
pub extern "C" fn ggml_backend_score() -> c_int {
    vendor::devices().len() as c_int
}

/// Bring-up self-test (NOT part of the ggml ABI): run a native NPU blob through our Level
/// Zero HAL. Feeds `input` (f32) to the graph's first input, writes its first output into
/// `output` (f32). Returns the number of f32 outputs written, or negative on error. Used by
/// the Intel submit-path harness; will be removed once wired into `graph_compute`.
#[no_mangle]
pub extern "C" fn ggml_npu_ivpu_selftest(
    blob: *const u8,
    blob_len: usize,
    input: *const f32,
    input_len: usize,
    output: *mut f32,
    output_cap: usize,
) -> c_int {
    if blob.is_null() || output.is_null() || (input.is_null() && input_len != 0) {
        return -1;
    }
    let blob = unsafe { core::slice::from_raw_parts(blob, blob_len) };
    let input = if input_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(input, input_len) }
    };
    let output = unsafe { core::slice::from_raw_parts_mut(output, output_cap) };
    vendor::ivpu::run_blob_selftest(blob, input, output)
}

/// Multi-input variant used by graph-coverage tests. Input lengths are in 32-bit words, allowing
/// integer tensors to use the same byte-preserving binding path.
#[no_mangle]
pub extern "C" fn ggml_npu_ivpu_selftest_multi(
    blob: *const u8,
    blob_len: usize,
    input_ptrs: *const *const c_void,
    input_lens: *const usize,
    n_inputs: usize,
    output: *mut f32,
    output_cap: usize,
) -> c_int {
    if blob.is_null()
        || input_ptrs.is_null()
        || input_lens.is_null()
        || output.is_null()
        || n_inputs == 0
    {
        return -1;
    }
    let blob = unsafe { core::slice::from_raw_parts(blob, blob_len) };
    let ptrs = unsafe { core::slice::from_raw_parts(input_ptrs, n_inputs) };
    let lens = unsafe { core::slice::from_raw_parts(input_lens, n_inputs) };
    let mut inputs = alloc::vec::Vec::with_capacity(n_inputs);
    for (&ptr, &len) in ptrs.iter().zip(lens) {
        if ptr.is_null() && len != 0 {
            return -1;
        }
        inputs.push(if len == 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(ptr.cast::<f32>(), len) }
        });
    }
    let output = unsafe { core::slice::from_raw_parts_mut(output, output_cap) };
    vendor::ivpu::run_blob_selftest_multi(blob, &inputs, output)
}
