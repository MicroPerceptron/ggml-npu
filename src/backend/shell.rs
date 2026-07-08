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

unsafe extern "C" fn dev_supports_op(_dev: ggml_backend_dev_t, op: *const ggml_tensor) -> bool {
    // Claim any 2D f32 MUL_MAT we can compile a blob for (compile-on-miss, cached in
    // vendor::ivpu). src0 (weight) [K,N], src1 (activation) [K,M]. Everything else -> CPU.
    if op.is_null() {
        return false;
    }
    let t = &*op;
    if t.op != GGML_OP_MUL_MAT || t.type_ != GGML_TYPE_F32 {
        return false;
    }
    let (src0, src1) = (t.src[0], t.src[1]);
    if src0.is_null() || src1.is_null() {
        return false;
    }
    let (s0, s1) = (&*src0, &*src1);
    if s0.type_ != GGML_TYPE_F32 || s1.type_ != GGML_TYPE_F32 {
        return false;
    }
    // 2D only (no batch dims), contraction dim K must match, both contiguous.
    if s0.ne[2] != 1 || s0.ne[3] != 1 || s1.ne[2] != 1 || s1.ne[3] != 1 || s0.ne[0] != s1.ne[0] {
        return false;
    }
    if !ggml_is_contiguous(src0 as *const ggml_tensor)
        || !ggml_is_contiguous(src1 as *const ggml_tensor)
    {
        return false;
    }
    let (k, n, m) = (s0.ne[0], s0.ne[1], s1.ne[1]);
    if !(1..=65536).contains(&k) || !(1..=65536).contains(&n) || !(1..=65536).contains(&m) {
        return false;
    }
    crate::vendor::ivpu::can_matmul(m, k, n)
}

unsafe extern "C" fn dev_supports_buft(
    _dev: ggml_backend_dev_t,
    buft: ggml_backend_buffer_type_t,
) -> bool {
    // Integrated SoC: NPU and CPU share RAM, so we can read operands from any host buffer
    // (ours or the CPU backend's) without a copy. Accept host bufts.
    !buft.is_null() && ggml_backend_buft_is_host(buft)
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
    cgraph: *mut ggml_cgraph,
) -> ggml_status {
    if cgraph.is_null() {
        return GGML_STATUS_SUCCESS;
    }
    // The scheduler hands us a split of claimed ops; run each MUL_MAT on the NPU.
    let n = ggml_graph_n_nodes(cgraph);
    for i in 0..n {
        let node = ggml_graph_node(cgraph, i);
        if node.is_null() {
            continue;
        }
        let t = &*node;
        if t.op != GGML_OP_MUL_MAT {
            continue;
        }
        if !dispatch_mul_mat(t) {
            return GGML_STATUS_FAILED;
        }
    }
    GGML_STATUS_SUCCESS
}

fn nelements(t: &ggml_tensor) -> usize {
    (t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]) as usize
}

/// Read src0 (weight) and src1 (activation) host data, run the NPU MatMul, write dst.
unsafe fn dispatch_mul_mat(dst: &ggml_tensor) -> bool {
    let (src0, src1) = (dst.src[0], dst.src[1]);
    if src0.is_null() || src1.is_null() || dst.data.is_null() {
        return false;
    }
    let (s0, s1) = (&*src0, &*src1);
    if s0.data.is_null() || s1.data.is_null() {
        return false;
    }
    let (k, n, m) = (s0.ne[0], s0.ne[1], s1.ne[1]);
    let w = core::slice::from_raw_parts(s0.data as *const f32, nelements(s0));
    let x = core::slice::from_raw_parts(s1.data as *const f32, nelements(s1));
    let out = core::slice::from_raw_parts_mut(dst.data as *mut f32, nelements(dst));
    crate::vendor::ivpu::run_matmul(m, k, n, w, x, out)
}
