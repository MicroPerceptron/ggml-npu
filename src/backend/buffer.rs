//! Buffer-type and buffer vtables — the **zero-copy** memory model.
//!
//! Allocations are NPU-mapped host memory (via the present vendor's `alloc_host`, e.g. Intel's
//! `zeMemAllocHost`): pinned + mapped into the NPU's address space so a tensor placed here can be
//! bound directly as a graph argument with no per-dispatch copy. It's still host-visible
//! (`is_host == true`), so the CPU reads/writes `tensor->data` directly and `set_tensor` (the
//! one-time weight upload at model load) is a plain memcpy into NPU memory. A weight offloaded to
//! us therefore costs one copy at load and **zero copies per token**.

use core::ffi::{c_char, c_void};
use core::ptr;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::ffi::*;

/// Page alignment for tensor offsets within a buffer. The base is page-aligned (the vendor
/// allocator returns page-aligned memory); keeping per-tensor offsets page-aligned too means
/// every bound argument pointer (`base + offset`) is page-aligned, which the NPU DMA path wants.
/// Weights are large, so the padding is negligible.
const BUFFER_ALIGN: usize = 4096;

/// Generation of model-resident NPU buffers. Compiled graphs bake model weights, so an address is
/// only a valid identity within the lifetime of the buffer allocation that owns it. Incrementing
/// this generation on buffer destruction prevents a later model allocation at the same virtual
/// address from reusing a graph with stale baked weights.
static BUFFER_GENERATION: AtomicU64 = AtomicU64::new(1);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

pub(super) fn generation() -> u64 {
    BUFFER_GENERATION.load(Ordering::Acquire)
}

pub(super) fn allocated_bytes() -> usize {
    ALLOCATED_BYTES.load(Ordering::Acquire)
}

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
    // NPU-mapped host memory so tensors placed here bind as graph args with no copy. If no
    // vendor is present (or the device allocation fails) we return null: ggml treats that as
    // OOM for this buffer type rather than silently handing back unbindable host memory.
    let base = match crate::vendor::present() {
        Some(v) => v.alloc_host(size.max(1)),
        None => ptr::null_mut(),
    };
    if base.is_null() {
        return ptr::null_mut();
    }
    ALLOCATED_BYTES.fetch_add(size, Ordering::AcqRel);
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
    if base.is_null() {
        return;
    }
    if let Some(v) = crate::vendor::present() {
        v.free_host(base, (*buffer).size);
    }
    ALLOCATED_BYTES.fetch_sub((*buffer).size, Ordering::AcqRel);
    BUFFER_GENERATION.fetch_add(1, Ordering::AcqRel);
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
