//! Buffer-type and buffer vtables: a minimal host allocation backed by the crate's
//! global allocator. Required to be present and valid, but never exercised while
//! `supports_op == false` (no tensor is ever placed on this device in Phase 0). The
//! set/get/memset paths become real weight-upload/repack code in Phase 2.

use alloc::alloc::{alloc, dealloc};
use core::alloc::Layout;
use core::ffi::{c_char, c_void};
use core::ptr;

use crate::ffi::*;

const BUFFER_ALIGN: usize = 32;

// ---------------------------------------------------------------------------
// Buffer type
// ---------------------------------------------------------------------------
pub(super) static BUFT_IFACE: ggml_backend_buffer_type_i = ggml_backend_buffer_type_i {
    get_name: Some(buft_get_name),
    alloc_buffer: Some(buft_alloc_buffer),
    get_alignment: Some(buft_get_alignment),
    get_max_size: None,
    get_alloc_size: None,
    is_host: Some(buft_is_host),
};

unsafe extern "C" fn buft_get_name(_buft: ggml_backend_buffer_type_t) -> *const c_char {
    c"NPU_HOST".as_ptr()
}

unsafe extern "C" fn buft_alloc_buffer(
    buft: ggml_backend_buffer_type_t,
    size: usize,
) -> ggml_backend_buffer_t {
    let layout = match Layout::from_size_align(size.max(1), BUFFER_ALIGN) {
        Ok(l) => l,
        Err(_) => return ptr::null_mut(),
    };
    let base = alloc(layout) as *mut c_void;
    if base.is_null() {
        return ptr::null_mut();
    }
    // ggml constructs the ggml_backend_buffer; its `context` is our base pointer,
    // which buffer_get_base returns.
    ggml_backend_buffer_init(buft, BUFFER_IFACE, base, size)
}

unsafe extern "C" fn buft_get_alignment(_buft: ggml_backend_buffer_type_t) -> usize {
    BUFFER_ALIGN
}

unsafe extern "C" fn buft_is_host(_buft: ggml_backend_buffer_type_t) -> bool {
    true
}

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------
pub(super) static BUFFER_IFACE: ggml_backend_buffer_i = ggml_backend_buffer_i {
    free_buffer: Some(buffer_free_buffer),
    get_base: Some(buffer_get_base),
    init_tensor: None,
    memset_tensor: Some(buffer_memset_tensor),
    set_tensor: Some(buffer_set_tensor),
    get_tensor: Some(buffer_get_tensor),
    set_tensor_2d: None,
    get_tensor_2d: None,
    cpy_tensor: None,
    clear: Some(buffer_clear),
    reset: None,
};

unsafe extern "C" fn buffer_free_buffer(buffer: ggml_backend_buffer_t) {
    if buffer.is_null() {
        return;
    }
    let base = (*buffer).context;
    let size = (*buffer).size.max(1);
    if base.is_null() {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(size, BUFFER_ALIGN) {
        dealloc(base as *mut u8, layout);
    }
}

unsafe extern "C" fn buffer_get_base(buffer: ggml_backend_buffer_t) -> *mut c_void {
    (*buffer).context
}

unsafe extern "C" fn buffer_memset_tensor(
    _buffer: ggml_backend_buffer_t,
    tensor: *mut ggml_tensor,
    value: u8,
    offset: usize,
    size: usize,
) {
    let dst = ((*tensor).data as *mut u8).add(offset);
    ptr::write_bytes(dst, value, size);
}

unsafe extern "C" fn buffer_set_tensor(
    _buffer: ggml_backend_buffer_t,
    tensor: *mut ggml_tensor,
    data: *const c_void,
    offset: usize,
    size: usize,
) {
    let dst = ((*tensor).data as *mut u8).add(offset);
    ptr::copy_nonoverlapping(data as *const u8, dst, size);
}

unsafe extern "C" fn buffer_get_tensor(
    _buffer: ggml_backend_buffer_t,
    tensor: *const ggml_tensor,
    data: *mut c_void,
    offset: usize,
    size: usize,
) {
    let src = ((*tensor).data as *const u8).add(offset);
    ptr::copy_nonoverlapping(src, data as *mut u8, size);
}

unsafe extern "C" fn buffer_clear(buffer: ggml_backend_buffer_t, value: u8) {
    let base = (*buffer).context as *mut u8;
    let size = (*buffer).size;
    if !base.is_null() {
        ptr::write_bytes(base, value, size);
    }
}
