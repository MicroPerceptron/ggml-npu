//! Buffer-type and buffer vtables — the **zero-copy** memory model.
//!
//! Allocations are NPU-mapped host memory branded `NPU_HOST`: pinned + host-visible
//! (`is_host == true`) so the CPU can `set_tensor` / read `tensor->data`, and bindable as graph
//! args without per-dispatch copy when the runtime can import them.
//!
//! **Honest residency:** under Direct IVPU the only valid backing is a
//! [`SharedAllocationOwner`](crate::interop::SharedAllocationOwner) from
//! `alloc_shared_backend`. Falling back to bare `alloc_host` would still be named `NPU_HOST` but
//! would not be importable — that lie forced staging (#6, PR #22). Level Zero may still use
//! `alloc_host` because it can bind foreign host pointers.

use alloc::boxed::Box;
use core::ffi::{c_char, c_void};
use core::ptr;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::ffi::*;
use crate::interop::{
    AllocationOwnership, OwnershipError, SharedAllocationOwner, SharedAllocationView,
};

/// Page alignment for tensor offsets within a buffer. The base is page-aligned (the vendor
/// allocator returns page-aligned memory); keeping per-tensor offsets page-aligned too means
/// every bound argument pointer (`base + offset`) is page-aligned, which the NPU DMA path wants.
/// Weights are large, so the padding is negligible.
const BUFFER_ALIGN: usize = 4096;

/// Extra bytes reserved after each NPU_HOST tensor so OpenVINO ELF `allocation_bytes`
/// (often slightly above logical nbytes) still fit when the tensor sits at the end of its
/// reserved region (`FINDING/direct-tg128-decode-ffn-invalid-metadata`, PR #62 pad path).
const ELF_SECTION_PAD_HEADROOM: usize = 256;

/// Generation of model-resident NPU buffers. Compiled graphs bake model weights, so an address is
/// only a valid identity within the lifetime of the buffer allocation that owns it. Incrementing
/// this generation on buffer destruction prevents a later model allocation at the same virtual
/// address from reusing a graph with stale baked weights.
static BUFFER_GENERATION: AtomicU64 = AtomicU64::new(1);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

pub(super) struct BufferContext {
    owner: Option<Box<dyn SharedAllocationOwner>>,
    legacy_base: *mut c_void,
    bytes: usize,
}

impl BufferContext {
    fn base(&self) -> *mut c_void {
        self.owner
            .as_ref()
            .map_or(self.legacy_base, |owner| owner.address().as_ptr().cast())
    }

    pub(super) fn owner(&self) -> Option<&dyn SharedAllocationOwner> {
        self.owner.as_deref()
    }
}

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
    get_alloc_size: Some(buft_get_alloc_size),
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
    let Some(vendor) = crate::vendor::present() else {
        return ptr::null_mut();
    };
    let owner = vendor.alloc_shared_backend(size.max(1));
    // Direct IVPU can only import SharedAllocationOwner-backed buffers. Falling back to bare
    // alloc_host creates owned-but-unimportable tensors that always stage at dispatch (#6).
    #[cfg(all(feature = "direct-ivpu", not(feature = "level-zero-compat")))]
    let base: *mut c_void = match owner.as_ref() {
        Some(owner) => owner.address().as_ptr().cast(),
        None => return ptr::null_mut(),
    };
    #[cfg(any(not(feature = "direct-ivpu"), feature = "level-zero-compat"))]
    let base: *mut c_void = owner.as_ref().map_or_else(
        || vendor.alloc_host(size.max(1)),
        |owner| owner.address().as_ptr().cast(),
    );
    if base.is_null() {
        return ptr::null_mut();
    }
    if let Some(owner) = owner.as_ref() {
        vendor.note_shared_allocation(owner.identity(), owner.len());
    }
    let context = Box::into_raw(Box::new(BufferContext {
        owner,
        legacy_base: base,
        bytes: size.max(1),
    }));
    ALLOCATED_BYTES.fetch_add(size, Ordering::AcqRel);
    ggml_backend_buffer_init(buft, BUFFER_IFACE, context.cast(), size)
}

unsafe extern "C" fn buft_get_alignment(_buft: ggml_backend_buffer_type_t) -> usize {
    BUFFER_ALIGN
}

unsafe extern "C" fn buft_get_alloc_size(
    _buft: ggml_backend_buffer_type_t,
    tensor: *const ggml_tensor,
) -> usize {
    if tensor.is_null() {
        return 0;
    }
    // SAFETY: ggml passes a live tensor when querying alloc size during buffer planning.
    let nbytes = match tensor_logical_bytes(unsafe { &*tensor }) {
        Some(bytes) => bytes,
        None => return 0,
    };
    nbytes.saturating_add(ELF_SECTION_PAD_HEADROOM)
}

fn tensor_logical_bytes(tensor: &ggml_tensor) -> Option<usize> {
    // SAFETY: type traits pointer is a process-lifetime table from ggml.
    let traits = unsafe { crate::ffi::ggml_get_type_traits(tensor.type_) };
    if traits.is_null() {
        return None;
    }
    let traits = unsafe { &*traits };
    let elements = tensor.ne.iter().try_fold(1usize, |product, &dimension| {
        product.checked_mul(usize::try_from(dimension).ok()?)
    })?;
    let block = usize::try_from(traits.blck_size).ok()?;
    if block == 0 || elements % block != 0 {
        return None;
    }
    elements
        .checked_div(block)?
        .checked_mul(traits.type_size)
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
    let context = (*buffer).context.cast::<BufferContext>();
    if context.is_null() {
        return;
    }
    // SAFETY: context was allocated exactly once by buft_alloc_buffer and buffer owns it.
    let context = Box::from_raw(context);
    if let Some(owner) = context.owner() {
        if owner.ownership() != AllocationOwnership::Backend {
            crate::vendor::present().inspect(|vendor| {
                vendor.debug_note(
                    "[npu] refusing to destroy shared buffer without confirmed backend ownership\n",
                )
            });
            // Safety over reclamation: an external participant may still access this allocation.
            let _ = Box::into_raw(context);
            return;
        }
    } else if let Some(v) = crate::vendor::present() {
        v.free_host(context.legacy_base, context.bytes);
    }
    ALLOCATED_BYTES.fetch_sub((*buffer).size, Ordering::AcqRel);
    BUFFER_GENERATION.fetch_add(1, Ordering::AcqRel);
}

unsafe extern "C" fn buffer_get_base(buffer: ggml_backend_buffer_t) -> *mut c_void {
    let context = (*buffer).context.cast::<BufferContext>();
    if context.is_null() {
        ptr::null_mut()
    } else {
        (*context).base()
    }
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
    let base = buffer_get_base(buffer).cast::<u8>();
    let size = (*buffer).size;
    if !base.is_null() {
        ptr::write_bytes(base, value, size);
    }
}

pub(in crate::backend) unsafe fn context(buffer: ggml_backend_buffer_t) -> Option<&'static BufferContext> {
    if buffer.is_null() {
        return None;
    }
    (*buffer).context.cast::<BufferContext>().as_ref()
}

pub(super) unsafe fn shared_view(tensor: *const ggml_tensor) -> Option<SharedAllocationView> {
    if tensor.is_null() || (*tensor).buffer.is_null() || (*tensor).data.is_null() {
        return None;
    }
    if (*(*tensor).buffer).buft != super::buffer_type() {
        return None;
    }
    let context = context((*tensor).buffer)?;
    let owner = context.owner()?;
    let base = owner.address().as_ptr() as usize;
    let data = (*tensor).data as usize;
    let traits = crate::ffi::ggml_get_type_traits((*tensor).type_);
    if traits.is_null() {
        return None;
    }
    let elements = (*tensor)
        .ne
        .iter()
        .try_fold(1usize, |product, &dimension| {
            product.checked_mul(usize::try_from(dimension).ok()?)
        })?;
    let block = usize::try_from((*traits).blck_size).ok()?;
    let type_size = (*traits).type_size;
    if block == 0 || elements % block != 0 {
        return None;
    }
    let bytes = elements.checked_div(block)?.checked_mul(type_size)?;
    let offset = data.checked_sub(base)?;
    if bytes == 0 || offset.checked_add(bytes)? > owner.len() {
        return None;
    }
    Some(SharedAllocationView {
        identity: owner.identity(),
        token: owner.token(),
        allocation_bytes: owner.len(),
        offset,
        bytes,
        alignment: owner.alignment(),
        address: core::ptr::NonNull::new((*tensor).data.cast::<u8>())?,
    })
}

pub(super) fn acquire_backend(buffer: ggml_backend_buffer_t) -> Result<(), OwnershipError> {
    // SAFETY: extension callbacks receive a live buffer from ggml.
    let Some(context) = (unsafe { context(buffer) }) else {
        return Err(OwnershipError::Released);
    };
    context
        .owner()
        .ok_or(OwnershipError::Released)?
        .acquire_backend()
}

pub(super) fn release_external(buffer: ggml_backend_buffer_t) -> Result<(), OwnershipError> {
    // SAFETY: extension callbacks receive a live buffer from ggml.
    let Some(context) = (unsafe { context(buffer) }) else {
        return Err(OwnershipError::Released);
    };
    context
        .owner()
        .ok_or(OwnershipError::Released)?
        .release_external()
}

#[cfg(test)]
mod tests {
    use super::{ELF_SECTION_PAD_HEADROOM, BUFFER_ALIGN};

    #[test]
    fn elf_pad_headroom_is_multiple_of_common_section_align() {
        assert!(BUFFER_ALIGN.is_power_of_two());
        assert_eq!(
            ELF_SECTION_PAD_HEADROOM % 64,
            0,
            "pad headroom should cover typical ELF section alignment (64)"
        );
        assert!(ELF_SECTION_PAD_HEADROOM >= 64);
    }
}
