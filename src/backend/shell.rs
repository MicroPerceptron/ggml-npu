//! The ggml "shell" vtables: registry, device, and backend (stream). Phase 0 claims no
//! ops (`supports_op == false`) so the scheduler places everything on CPU. Bodies avoid
//! panicking paths (no unwrap/index) since a panic aborts the host (see crate docs).

use alloc::boxed::Box;
use core::ffi::{c_char, CStr};
use core::ptr;

use crate::ffi::*;

extern "C" {
    fn getenv(name: *const c_char) -> *const c_char;
}

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
    offload_op: Some(dev_offload_op),
    event_new: Some(dev_event_new),
    event_free: Some(dev_event_free),
    event_synchronize: Some(dev_event_synchronize),
};

unsafe extern "C" fn dev_get_name(_dev: ggml_backend_dev_t) -> *const c_char {
    c"NPU".as_ptr()
}

unsafe extern "C" fn dev_get_description(_dev: ggml_backend_dev_t) -> *const c_char {
    super::description()
}

unsafe extern "C" fn dev_get_memory(_dev: ggml_backend_dev_t, free: *mut usize, total: *mut usize) {
    let limit = memory_limit();
    let available = limit.saturating_sub(crate::backend::allocated_bytes());
    if !free.is_null() {
        *free = available;
    }
    if !total.is_null() {
        *total = limit;
    }
}

fn memory_limit() -> usize {
    let p = unsafe { getenv(c"GGML_NPU_MEMORY_LIMIT".as_ptr()) };
    if p.is_null() {
        return 0;
    }
    let mut value = 0usize;
    for &b in unsafe { CStr::from_ptr(p) }.to_bytes() {
        if !b.is_ascii_digit() {
            return 0;
        }
        value = match value
            .checked_mul(10)
            .and_then(|x| x.checked_add((b - b'0') as usize))
        {
            Some(v) => v,
            None => return 0,
        };
    }
    value
}

fn async_enabled() -> bool {
    !unsafe { getenv(c"GGML_NPU_ASYNC".as_ptr()) }.is_null()
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
    dev_get_memory(dev, &mut (*props).memory_free, &mut (*props).memory_total);
    (*props).device_id = ptr::null();
    (*props).caps = ggml_backend_dev_caps {
        async_: async_enabled(),
        host_buffer: false,
        buffer_from_host_ptr: false,
        events: true,
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

unsafe extern "C" fn dev_supports_op(_dev: ggml_backend_dev_t, op: *const ggml_tensor) -> bool {
    // Op-agnostic: the engine matches the node and asks the present vendor if it can run it.
    crate::policy::should_pull(op) && crate::engine::supports(op)
}

unsafe extern "C" fn dev_supports_buft(
    _dev: ggml_backend_dev_t,
    buft: ggml_backend_buffer_type_t,
) -> bool {
    // Integrated SoC: NPU and CPU share RAM, so we can read operands from any host buffer
    // (ours or the CPU backend's) without a copy. Accept host bufts.
    !buft.is_null() && ggml_backend_buft_is_host(buft)
}

unsafe extern "C" fn dev_offload_op(_dev: ggml_backend_dev_t, op: *const ggml_tensor) -> bool {
    // The graph compiler can bake weights directly from foreign host buffers, so expensive
    // supported supernodes remain candidates even when model placement did not pre-allocate them
    // in NPU memory. The capability check is still conservative and keeps the vocabulary head off.
    crate::engine::supports(op)
}

unsafe extern "C" fn dev_event_new(dev: ggml_backend_dev_t) -> ggml_backend_event_t {
    Box::into_raw(Box::new(ggml_backend_event {
        device: dev,
        context: ptr::null_mut(),
    }))
}

unsafe extern "C" fn dev_event_free(_dev: ggml_backend_dev_t, event: ggml_backend_event_t) {
    if !event.is_null() {
        drop(Box::from_raw(event));
    }
}

unsafe extern "C" fn dev_event_synchronize(_dev: ggml_backend_dev_t, _event: ggml_backend_event_t) {
    let _ = crate::engine::synchronize();
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
    synchronize: Some(backend_synchronize),
    graph_plan_create: Some(backend_graph_plan_create),
    graph_plan_free: Some(backend_graph_plan_free),
    graph_plan_update: Some(backend_graph_plan_update),
    graph_plan_compute: Some(backend_graph_plan_compute),
    graph_compute: Some(backend_graph_compute),
    event_record: Some(backend_event_record),
    event_wait: Some(backend_event_wait),
    graph_optimize: None,
};

unsafe extern "C" fn backend_get_name(_backend: ggml_backend_t) -> *const c_char {
    c"NPU".as_ptr()
}

unsafe extern "C" fn backend_free(_backend: ggml_backend_t) {
    // Singletons live for the process lifetime; nothing to free.
    let _ = crate::engine::synchronize();
}

unsafe extern "C" fn backend_synchronize(_backend: ggml_backend_t) {
    let _ = crate::engine::synchronize();
}

unsafe extern "C" fn backend_event_record(_backend: ggml_backend_t, _event: ggml_backend_event_t) {}

unsafe extern "C" fn backend_event_wait(_backend: ggml_backend_t, _event: ggml_backend_event_t) {
    let _ = crate::engine::synchronize();
}

struct GraphPlan {
    graph: *mut ggml_cgraph,
}

unsafe extern "C" fn backend_graph_plan_create(
    _backend: ggml_backend_t,
    graph: *const ggml_cgraph,
) -> ggml_backend_graph_plan_t {
    if graph.is_null() {
        return ptr::null_mut();
    }
    Box::into_raw(Box::new(GraphPlan {
        graph: graph as *mut ggml_cgraph,
    })) as ggml_backend_graph_plan_t
}

unsafe extern "C" fn backend_graph_plan_free(
    _backend: ggml_backend_t,
    plan: ggml_backend_graph_plan_t,
) {
    if !plan.is_null() {
        drop(Box::from_raw(plan as *mut GraphPlan));
    }
}

unsafe extern "C" fn backend_graph_plan_update(
    _backend: ggml_backend_t,
    plan: ggml_backend_graph_plan_t,
    graph: *const ggml_cgraph,
) {
    if !plan.is_null() && !graph.is_null() {
        (*(plan as *mut GraphPlan)).graph = graph as *mut ggml_cgraph;
    }
}

unsafe extern "C" fn backend_graph_plan_compute(
    _backend: ggml_backend_t,
    plan: ggml_backend_graph_plan_t,
) -> ggml_status {
    if plan.is_null() {
        return GGML_STATUS_FAILED;
    }
    if crate::engine::compute((*(plan as *mut GraphPlan)).graph) {
        GGML_STATUS_SUCCESS
    } else {
        GGML_STATUS_FAILED
    }
}

unsafe extern "C" fn backend_graph_compute(
    _backend: ggml_backend_t,
    cgraph: *mut ggml_cgraph,
) -> ggml_status {
    if crate::engine::compute(cgraph) {
        GGML_STATUS_SUCCESS
    } else {
        GGML_STATUS_FAILED
    }
}
