//! Optional ggml external-memory registry extension.
//!
//! Generic ABI validation and ownership state live here. Platform handle construction remains in
//! the allocation owner supplied by the active vendor adapter.

use core::ffi::{c_char, c_void, CStr};
use core::ptr;

use crate::ffi::*;
use crate::interop::AllocationOwnership;

const VERSION: u32 = 1;
const ACCESS_READ: u32 = 1 << 0;
const ACCESS_WRITE: u32 = 1 << 1;
const CAP_OWNERSHIP_TRANSFER: u32 = 1 << 1;
const CAP_HOST_MAP: u32 = 1 << 4;
const CAP_SYNC_RELEASE: u32 = 1 << 5;
const CAP_EXPORT: u32 = 1 << 6;
const STATUS_ERROR: i32 = -1;
const STATUS_UNSUPPORTED: i32 = 0;
const STATUS_OWNERSHIP_TRANSFERRED: i32 = 2;
const STATUS_SYNCHRONIZED: i32 = 3;

fn status(value: i32) -> ggml_backend_external_memory_status {
    value
}

unsafe extern "C" fn get_capabilities(
    _dev: ggml_backend_dev_t,
    caps: *mut ggml_backend_external_memory_caps,
) -> bool {
    if caps.is_null() {
        return false;
    }
    (*caps).version = VERSION;
    (*caps).capabilities = CAP_EXPORT | CAP_OWNERSHIP_TRANSFER | CAP_HOST_MAP | CAP_SYNC_RELEASE;
    (*caps).min_alignment = 4096;
    true
}

unsafe extern "C" fn export_buffer(
    buffer: ggml_backend_buffer_t,
    offset: usize,
    size: usize,
    access: u32,
    desc: *mut ggml_backend_external_memory_desc,
) -> bool {
    if buffer.is_null()
        || desc.is_null()
        || access == 0
        || access & !(ACCESS_READ | ACCESS_WRITE) != 0
    {
        return false;
    }
    let Some(context) = super::buffer::context(buffer) else {
        return false;
    };
    let Some(owner) = context.owner() else {
        return false;
    };
    if owner.ownership() != AllocationOwnership::Backend
        || size == 0
        || offset.checked_add(size).is_none_or(|end| end > owner.len())
        || offset % owner.alignment() != 0
    {
        return false;
    }
    let identity = owner.identity();
    let handle = owner.external_handle();
    *desc = core::mem::zeroed();
    (*desc).version = VERSION;
    (*desc).access = access;
    (*desc).allocation_id = identity.external_id();
    (*desc).generation = identity.generation;
    (*desc).allocation_size = owner.len();
    (*desc).offset = offset;
    (*desc).size = size;
    (*desc).alignment = owner.alignment();
    (*desc).handle.type_ = handle.type_name.as_ptr();
    (*desc).handle.data = handle.data;
    (*desc).handle.size = handle.size;
    true
}

unsafe extern "C" fn release_memory_desc(
    _buffer: ggml_backend_buffer_t,
    desc: *mut ggml_backend_external_memory_desc,
) {
    if !desc.is_null() {
        *desc = core::mem::zeroed();
    }
}

unsafe extern "C" fn acquire(
    _backend: ggml_backend_t,
    buffer: ggml_backend_buffer_t,
    _access: u32,
    wait: *const ggml_backend_external_sync_desc,
) -> ggml_backend_external_memory_status {
    if !wait.is_null() {
        return status(STATUS_UNSUPPORTED);
    }
    match super::buffer::acquire_backend(buffer) {
        Ok(()) => status(STATUS_OWNERSHIP_TRANSFERRED),
        Err(_) => status(STATUS_ERROR),
    }
}

unsafe extern "C" fn release(
    _backend: ggml_backend_t,
    buffer: ggml_backend_buffer_t,
    _access: u32,
    signal: *mut ggml_backend_external_sync_desc,
) -> ggml_backend_external_memory_status {
    if !signal.is_null() {
        return status(STATUS_UNSUPPORTED);
    }
    if !crate::engine::synchronize() {
        if let Some(context) = super::buffer::context(buffer) {
            if let Some(owner) = context.owner() {
                owner.mark_uncertain();
            }
        }
        return status(STATUS_ERROR);
    }
    match super::buffer::release_external(buffer) {
        Ok(()) => status(STATUS_SYNCHRONIZED),
        Err(_) => status(STATUS_ERROR),
    }
}

static mut EXTERNAL_MEMORY_IFACE: ggml_backend_external_memory_i = ggml_backend_external_memory_i {
    version: VERSION,
    struct_size: core::mem::size_of::<ggml_backend_external_memory_i>() as u32,
    get_capabilities: Some(get_capabilities),
    export_buffer: Some(export_buffer),
    release_memory_desc: Some(release_memory_desc),
    import_buffer: None,
    acquire: Some(acquire),
    release: Some(release),
    release_sync_desc: None,
};

pub(super) unsafe fn proc_address(name: *const c_char) -> *mut c_void {
    if name.is_null() {
        return ptr::null_mut();
    }
    if CStr::from_ptr(name).to_bytes() == b"ggml_backend_external_memory_v1" {
        ptr::addr_of_mut!(EXTERNAL_MEMORY_IFACE).cast()
    } else {
        ptr::null_mut()
    }
}
