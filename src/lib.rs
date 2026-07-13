//! ggml-npu — an experimental universal NPU backend for ggml, `no_std`.
//!
//! Registers as a dynamically-loaded ggml `ACCEL` backend. The active Intel implementation
//! recognizes constant-weight matmuls and fused subgraphs, compiles them through OpenVINO, and
//! submits native graphs through Level Zero. Unsupported work remains on other ggml backends.
//!
//! `no_std` by design. The selected built-in host provider supplies process allocation,
//! diagnostics, dynamic loading, and termination. There is no unwinding runtime — a panic in any
//! vtable entry hits [`on_panic`] and aborts, which is UB-free across the C ABI.
//!
//! Load it with `GGML_BACKEND_PATH=/abs/path/libggml_npu.so`.
#![no_std]
#![feature(allocator_api)]
#![feature(f16)]
// `portable-simd` is a production default pinned to this repository's nightly toolchain. Builds
// which disable default features retain the scalar quant-transcode / requant helpers.
#![cfg_attr(feature = "portable-simd", feature(portable_simd))]

extern crate alloc;

#[cfg(any(test, feature = "bench", feature = "test-support"))]
extern crate std;

use core::alloc::{AllocError, Allocator, GlobalAlloc, Layout};
use core::ffi::{c_int, c_void};
#[cfg(not(any(test, feature = "bench", feature = "test-support")))]
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

pub(crate) mod address;
pub mod artifact_cache;
mod backend;
#[cfg(any(test, feature = "test-support"))]
mod coverage;
mod engine;
mod graph;
mod hal;
mod host;
pub mod identity;
pub mod interop;
pub mod metrics;
mod ops;
mod policy;
pub mod runtime;
mod sched;
mod timeline;
mod timeline_ring;
mod trace;
mod vendor;

#[cfg(feature = "direct-ivpu")]
pub use vendor::ivpu::direct as direct_ivpu;
#[cfg(feature = "bench-hardware")]
pub use vendor::ivpu::LevelZeroBenchmarkFixture;

use crate::host::{AllocationService, HostEnvironment};

/// The selected host allocator. Serves as **both** the global hook
/// ([`GlobalAlloc`]) and a first-class allocator value ([`Allocator`]) so owning types can be
/// constructed with an explicit allocator — `Vec::new_in(HostAlloc)` — rather than the implicit
/// `Global` (issue #1). A zero-size ZST, so passing it by value is free.
#[derive(Clone, Copy, Default)]
pub(crate) struct HostAlloc;

unsafe impl GlobalAlloc for HostAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { host::builtin().allocator().allocate(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { host::builtin().allocator().deallocate(ptr, layout) };
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

#[cfg(not(any(test, feature = "bench", feature = "test-support")))]
#[panic_handler]
fn on_panic(_info: &PanicInfo) -> ! {
    // no_std: no unwinding. Abort rather than let a panic cross the C ABI.
    host::builtin().allocator().abort()
}

// The precompiled `core`/`alloc` are built with unwinding and reference these symbols,
// even though `panic = "abort"` means we never unwind. Provide self-contained no-op shims
// so the cdylib does not depend on the host's libgcc; neither is ever actually called (a
// panic aborts in `on_panic` first). A leaner alternative is `-Zbuild-std` with
// `panic_immediate_abort`, but that would force an explicit `--target` on every build.
#[no_mangle]
#[cfg(not(any(test, feature = "bench", feature = "test-support")))]
extern "C" fn rust_eh_personality() {}

#[no_mangle]
#[cfg(not(any(test, feature = "bench", feature = "test-support")))]
extern "C" fn _Unwind_Resume() -> ! {
    host::builtin().allocator().abort()
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

#[cfg(feature = "bench-hardware")]
#[no_mangle]
pub extern "C" fn ggml_npu_hardware_fixture_create(
    kind: u32,
    blob: *const u8,
    blob_len: usize,
    input_ptrs: *const *const f32,
    input_lens: *const usize,
    input_count: usize,
) -> *mut c_void {
    if blob.is_null() || input_ptrs.is_null() || input_lens.is_null() || input_count == 0 {
        return core::ptr::null_mut();
    }
    // SAFETY: benchmark caller keeps these immutable buffers live for fixture construction.
    let blob = unsafe { core::slice::from_raw_parts(blob, blob_len) };
    let pointers = unsafe { core::slice::from_raw_parts(input_ptrs, input_count) };
    let lengths = unsafe { core::slice::from_raw_parts(input_lens, input_count) };
    let mut inputs = alloc::vec::Vec::with_capacity(input_count);
    for (&pointer, &length) in pointers.iter().zip(lengths) {
        if pointer.is_null() {
            return core::ptr::null_mut();
        }
        // SAFETY: lengths are supplied with live benchmark input pointers.
        inputs.push(unsafe { core::slice::from_raw_parts(pointer, length) });
    }
    match vendor::ivpu::HardwareBenchmarkFixture::new(kind, blob, &inputs) {
        Ok(fixture) => alloc::boxed::Box::into_raw(alloc::boxed::Box::new(fixture)).cast(),
        Err(_) => core::ptr::null_mut(),
    }
}

#[cfg(feature = "bench-hardware")]
#[no_mangle]
pub extern "C" fn ggml_npu_hardware_fixture_execute(fixture: *mut c_void) -> c_int {
    if fixture.is_null() {
        return -1;
    }
    // SAFETY: pointer was returned by create and remains uniquely owned by the caller.
    let fixture = unsafe { &mut *fixture.cast::<vendor::ivpu::HardwareBenchmarkFixture>() };
    fixture.execute().map_or_else(|code| code, |_| 0)
}

#[cfg(feature = "bench-hardware")]
#[no_mangle]
pub extern "C" fn ggml_npu_hardware_fixture_copy_output(
    fixture: *const c_void,
    output: *mut f32,
    output_capacity: usize,
) -> c_int {
    if fixture.is_null() || output.is_null() {
        return -1;
    }
    // SAFETY: pointers were created by this API and output capacity is caller-supplied.
    let fixture = unsafe { &*fixture.cast::<vendor::ivpu::HardwareBenchmarkFixture>() };
    let output = unsafe { core::slice::from_raw_parts_mut(output, output_capacity) };
    fixture
        .copy_output(output)
        .map_or_else(|code| code, |count| count as c_int)
}

#[cfg(feature = "bench-hardware")]
#[no_mangle]
pub extern "C" fn ggml_npu_hardware_fixture_destroy(fixture: *mut c_void) {
    if !fixture.is_null() {
        // SAFETY: pointer was returned by create and must be destroyed exactly once.
        drop(unsafe {
            alloc::boxed::Box::from_raw(fixture.cast::<vendor::ivpu::HardwareBenchmarkFixture>())
        });
    }
}
