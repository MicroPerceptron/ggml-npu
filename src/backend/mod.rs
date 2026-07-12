//! ggml backend shell: the four C vtables and the singleton they stand on.
//!
//! `reg -> device -> buffer_type`, plus one backend (stream) instance, all held in a
//! single heap allocation with self-referential interior pointers. Immutable after init
//! and never moved or freed, so the pointers we hand ggml stay valid for the process
//! lifetime. Init is a lock-free CAS (no global lock); the loader calls
//! `ggml_backend_init` once, so the race is a formality we nonetheless handle correctly.

mod buffer;
#[cfg(ggml_has_external_memory)]
mod external;
mod shell;

use core::ffi::{c_char, c_int};
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use alloc::boxed::Box;
use alloc::ffi::CString;

use crate::ffi::*;

const _: () = assert!(GGML_BACKEND_API_VERSION == 2);

/// One allocation holding the whole self-referential vtable graph, plus the owned device
/// description string (the probed NPU name, so `dev_get_description` can hand out a stable
/// pointer).
struct Handles {
    reg: ggml_backend_reg,
    device: ggml_backend_device,
    buft: ggml_backend_buffer_type,
    backend: ggml_backend,
    guid: ggml_guid,
    desc: CString,
}

// SAFETY: written once during init, only read afterwards, never moved.
unsafe impl Sync for Handles {}
unsafe impl Send for Handles {}

static HANDLES: AtomicPtr<Handles> = AtomicPtr::new(ptr::null_mut());

const NPU_GUID_BYTES: [u8; 16] = *b"ggml-npu-phase0\0";

fn get_or_init() -> *mut Handles {
    let existing = HANDLES.load(Ordering::Acquire);
    if !existing.is_null() {
        return existing;
    }
    let built = build();
    match HANDLES.compare_exchange(ptr::null_mut(), built, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => built,
        Err(winner) => {
            // Lost the (practically impossible) race: reclaim ours, use the winner's.
            drop(unsafe { Box::from_raw(built) });
            winner
        }
    }
}

/// Device description: the probed NPU name if present, else a generic fallback. The
/// literals contain no interior NUL, so `CString::new` cannot fail on them.
fn make_description() -> CString {
    match crate::vendor::devices().into_iter().next() {
        Some(dev) => CString::new(alloc::format!("NPU: {}", dev.name))
            .unwrap_or_else(|_| CString::new("NPU").unwrap()),
        None => CString::new("Universal NPU (no device; CPU fallback)").unwrap(),
    }
}

fn build() -> *mut Handles {
    // Allocate first for a stable address, then wire the self-referential pointers.
    let hp = Box::into_raw(Box::new(Handles {
        reg: ggml_backend_reg {
            api_version: GGML_BACKEND_API_VERSION as c_int,
            iface: shell::REG_IFACE,
            context: ptr::null_mut(),
        },
        device: ggml_backend_device {
            iface: shell::DEVICE_IFACE,
            reg: ptr::null_mut(),
            context: ptr::null_mut(),
        },
        buft: ggml_backend_buffer_type {
            iface: buffer::BUFT_IFACE,
            device: ptr::null_mut(),
            context: ptr::null_mut(),
        },
        backend: ggml_backend {
            guid: ptr::null_mut(),
            iface: shell::BACKEND_IFACE,
            device: ptr::null_mut(),
            context: ptr::null_mut(),
        },
        guid: NPU_GUID_BYTES,
        desc: make_description(),
    }));
    unsafe {
        (*hp).device.reg = &mut (*hp).reg;
        (*hp).buft.device = &mut (*hp).device;
        (*hp).backend.device = &mut (*hp).device;
        (*hp).backend.guid = &mut (*hp).guid as *mut ggml_guid;
    }
    hp
}

pub(crate) fn registry() -> *mut ggml_backend_reg {
    unsafe { &mut (*get_or_init()).reg }
}

fn device() -> *mut ggml_backend_device {
    unsafe { &mut (*get_or_init()).device }
}

fn buffer_type() -> *mut ggml_backend_buffer_type {
    unsafe { &mut (*get_or_init()).buft }
}

/// Whether `tensor` lives in **our** buffer type — i.e. its `data` is NPU-mapped memory
/// (shared-backend allocation under Direct IVPU; Level Zero may use `alloc_host`) that can be
/// bound as a graph argument with **no copy**. False for foreign buffers (a CPU-produced
/// activation, a weight the model kept on CPU), which the vendor must stage through a bounce
/// buffer. O(1): a pointer compare against our singleton buffer-type. Since our buffer type
/// only ever hands out NPU memory (it fails the allocation otherwise), "our buft" ⟺
/// "NPU-mapped", so this is exact.
pub(crate) unsafe fn owns_tensor(tensor: *const ggml_tensor) -> bool {
    if tensor.is_null() {
        return false;
    }
    let buf = (*tensor).buffer;
    !buf.is_null() && (*buf).buft == buffer_type()
}

pub(crate) unsafe fn shared_view(
    tensor: *const ggml_tensor,
) -> Option<crate::interop::SharedAllocationView> {
    buffer::shared_view(tensor)
}

/// Identity epoch for model-resident NPU buffers. Graph caches include this value because their
/// programs bake weights and therefore must not survive an unload/reload address reuse.
pub(crate) fn buffer_generation() -> u64 {
    buffer::generation()
}

pub(crate) fn allocated_bytes() -> usize {
    buffer::allocated_bytes()
}

fn backend_instance() -> *mut ggml_backend {
    unsafe { &mut (*get_or_init()).backend }
}

/// Stable pointer to the owned device-description C string.
fn description() -> *const c_char {
    unsafe { (*get_or_init()).desc.as_ptr() }
}
