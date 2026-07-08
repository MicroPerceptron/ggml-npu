//! The ggml "shell" vtables: registry, device, and backend (stream). Phase 0 claims no
//! ops (`supports_op == false`) so the scheduler places everything on CPU. Bodies avoid
//! panicking paths (no unwrap/index) since a panic aborts the host (see crate docs).

use core::ffi::c_char;
use core::ptr;

use crate::ffi::*;

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
pub(super) static REG_IFACE: ggml_backend_reg_i = ggml_backend_reg_i {
    get_name: Some(reg_get_name),
    get_device_count: Some(reg_get_device_count),
    get_device: Some(reg_get_device),
    get_proc_address: None,
};

unsafe extern "C" fn reg_get_name(_reg: ggml_backend_reg_t) -> *const c_char {
    c"NPU".as_ptr()
}

unsafe extern "C" fn reg_get_device_count(_reg: ggml_backend_reg_t) -> usize {
    1
}

unsafe extern "C" fn reg_get_device(_reg: ggml_backend_reg_t, index: usize) -> ggml_backend_dev_t {
    if index != 0 {
        return ptr::null_mut();
    }
    super::device()
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------
pub(super) static DEVICE_IFACE: ggml_backend_device_i = ggml_backend_device_i {
    get_name: Some(dev_get_name),
    get_description: Some(dev_get_description),
    get_memory: Some(dev_get_memory),
    get_type: Some(dev_get_type),
    get_props: Some(dev_get_props),
    init_backend: Some(dev_init_backend),
    get_buffer_type: Some(dev_get_buffer_type),
    get_host_buffer_type: None,
    buffer_from_host_ptr: None,
    supports_op: Some(dev_supports_op),
    supports_buft: Some(dev_supports_buft),
    offload_op: None,
    event_new: None,
    event_free: None,
    event_synchronize: None,
};

unsafe extern "C" fn dev_get_name(_dev: ggml_backend_dev_t) -> *const c_char {
    c"NPU".as_ptr()
}

unsafe extern "C" fn dev_get_description(_dev: ggml_backend_dev_t) -> *const c_char {
    super::description()
}

unsafe extern "C" fn dev_get_memory(_dev: ggml_backend_dev_t, free: *mut usize, total: *mut usize) {
    if !free.is_null() {
        *free = 0;
    }
    if !total.is_null() {
        *total = 0;
    }
}

unsafe extern "C" fn dev_get_type(_dev: ggml_backend_dev_t) -> ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_ACCEL
}

unsafe extern "C" fn dev_get_props(dev: ggml_backend_dev_t, props: *mut ggml_backend_dev_props) {
    if props.is_null() {
        return;
    }
    (*props).name = dev_get_name(dev);
    (*props).description = dev_get_description(dev);
    (*props).type_ = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    (*props).memory_free = 0;
    (*props).memory_total = 0;
    (*props).device_id = ptr::null();
    (*props).caps = ggml_backend_dev_caps {
        async_: false,
        host_buffer: false,
        buffer_from_host_ptr: false,
        events: false,
    };
}

unsafe extern "C" fn dev_init_backend(
    _dev: ggml_backend_dev_t,
    _params: *const c_char,
) -> ggml_backend_t {
    super::backend_instance()
}

unsafe extern "C" fn dev_get_buffer_type(_dev: ggml_backend_dev_t) -> ggml_backend_buffer_type_t {
    super::buffer_type()
}

unsafe extern "C" fn dev_supports_op(_dev: ggml_backend_dev_t, _op: *const ggml_tensor) -> bool {
    // Phase 0: claim nothing. Recognizing fusible regions is Phase 3 work.
    false
}

unsafe extern "C" fn dev_supports_buft(
    _dev: ggml_backend_dev_t,
    _buft: ggml_backend_buffer_type_t,
) -> bool {
    false
}

// ---------------------------------------------------------------------------
// Backend (stream)
// ---------------------------------------------------------------------------
pub(super) static BACKEND_IFACE: ggml_backend_i = ggml_backend_i {
    get_name: Some(backend_get_name),
    free: Some(backend_free),
    set_tensor_async: None,
    get_tensor_async: None,
    set_tensor_2d_async: None,
    get_tensor_2d_async: None,
    cpy_tensor_async: None,
    synchronize: None,
    graph_plan_create: None,
    graph_plan_free: None,
    graph_plan_update: None,
    graph_plan_compute: None,
    graph_compute: Some(backend_graph_compute),
    event_record: None,
    event_wait: None,
    graph_optimize: None,
};

unsafe extern "C" fn backend_get_name(_backend: ggml_backend_t) -> *const c_char {
    c"NPU".as_ptr()
}

unsafe extern "C" fn backend_free(_backend: ggml_backend_t) {
    // Singletons live for the process lifetime; nothing to free.
}

unsafe extern "C" fn backend_graph_compute(
    _backend: ggml_backend_t,
    _cgraph: *mut ggml_cgraph,
) -> ggml_status {
    // Unreachable while supports_op == false; return SUCCESS for an empty assignment.
    GGML_STATUS_SUCCESS
}
