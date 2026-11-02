//! ggml registry/device/stream vtables. All execution is synchronous at the ggml boundary.
use crate::{
    ffi::*,
    graph,
    runtime::{self, Job, Provider, Worker},
};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::{CString, c_char, c_void};
use std::ptr;
use std::sync::OnceLock;
use virtio_accel_tosa::CapabilityDescriptor;

struct Device {
    handle: ggml_backend_device,
    buffer_type: ggml_backend_buffer_type,
    provider: Provider,
    cap: CapabilityDescriptor,
    name: CString,
    description: CString,
}
struct Registry {
    handle: ggml_backend_reg,
    // ggml retains pointers into each device; moving the vector must not move devices.
    #[allow(clippy::vec_box)]
    devices: Vec<Box<Device>>,
}
struct Stream {
    worker: Worker,
    cap: CapabilityDescriptor,
}
struct Buffer {
    base: *mut u8,
    layout: Layout,
}
static REGISTRY: OnceLock<usize> = OnceLock::new();
static GUID: [u8; 16] = *b"ggml-virtio-accl";

fn handles() -> *mut Registry {
    *REGISTRY.get_or_init(|| {
        let mut r = Box::new(Registry {
            handle: ggml_backend_reg {
                api_version: GGML_BACKEND_API_VERSION as i32,
                iface: ggml_backend_reg_i {
                    get_name: Some(reg_name),
                    get_device_count: Some(reg_count),
                    get_device: Some(reg_device),
                    get_proc_address: None,
                },
                context: ptr::null_mut(),
            },
            devices: Vec::new(),
        });
        r.handle.context = (&mut *r as *mut Registry).cast();
        for found in runtime::discover() {
            let Ok(name) = CString::new(found.name) else {
                continue;
            };
            let Ok(description) = CString::new(found.description) else {
                continue;
            };
            let mut d = Box::new(Device {
                handle: ggml_backend_device::default(),
                buffer_type: ggml_backend_buffer_type::default(),
                provider: found.provider,
                cap: found.capability,
                name,
                description,
            });
            d.handle = ggml_backend_device {
                iface: device_iface(),
                reg: &mut r.handle,
                context: (&mut *d as *mut Device).cast(),
            };
            d.buffer_type = ggml_backend_buffer_type {
                iface: buffer_type_iface(),
                device: &mut d.handle,
                context: ptr::null_mut(),
            };
            r.devices.push(d);
        }
        Box::into_raw(r) as usize
    }) as *mut Registry
}
pub fn registry() -> ggml_backend_reg_t {
    unsafe { &mut (*handles()).handle }
}
pub fn device_count() -> usize {
    unsafe { (*handles()).devices.len() }
}
unsafe fn device<'a>(d: ggml_backend_dev_t) -> &'a Device {
    unsafe { &*(*d).context.cast::<Device>() }
}
unsafe extern "C" fn reg_name(_: ggml_backend_reg_t) -> *const c_char {
    c"VIRTIO_ACCEL".as_ptr()
}
unsafe extern "C" fn reg_count(_: ggml_backend_reg_t) -> usize {
    device_count()
}
unsafe extern "C" fn reg_device(_: ggml_backend_reg_t, index: usize) -> ggml_backend_dev_t {
    unsafe {
        let registry = &*handles();
        registry
            .devices
            .get(index)
            .map_or(ptr::null_mut(), |d| ptr::from_ref(&d.handle).cast_mut())
    }
}
fn device_iface() -> ggml_backend_device_i {
    ggml_backend_device_i {
        get_name: Some(dev_name),
        get_description: Some(dev_description),
        get_memory: Some(dev_memory),
        get_type: Some(dev_type),
        get_props: Some(dev_props),
        init_backend: Some(dev_init),
        get_buffer_type: Some(dev_buffer),
        supports_op: Some(dev_supports),
        supports_buft: Some(dev_supports_buft),
        offload_op: Some(dev_offload),
        ..Default::default()
    }
}
unsafe extern "C" fn dev_name(d: ggml_backend_dev_t) -> *const c_char {
    unsafe { device(d).name.as_ptr() }
}
unsafe extern "C" fn dev_description(d: ggml_backend_dev_t) -> *const c_char {
    unsafe { device(d).description.as_ptr() }
}
unsafe extern "C" fn dev_memory(_: ggml_backend_dev_t, free: *mut usize, total: *mut usize) {
    unsafe {
        if !free.is_null() {
            *free = 0;
        }
        if !total.is_null() {
            *total = 0;
        }
    }
}
unsafe extern "C" fn dev_type(_: ggml_backend_dev_t) -> ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_ACCEL
}
unsafe extern "C" fn dev_props(d: ggml_backend_dev_t, props: *mut ggml_backend_dev_props) {
    unsafe {
        if let Some(p) = props.as_mut() {
            *p = ggml_backend_dev_props {
                name: dev_name(d),
                description: dev_description(d),
                type_: dev_type(d),
                ..Default::default()
            };
        }
    }
}
unsafe extern "C" fn dev_buffer(d: ggml_backend_dev_t) -> ggml_backend_buffer_type_t {
    unsafe { ptr::from_ref(&device(d).buffer_type).cast_mut() }
}
unsafe extern "C" fn dev_supports(d: ggml_backend_dev_t, t: *const ggml_tensor) -> bool {
    unsafe { graph::supports(t, device(d).cap) }
}
unsafe extern "C" fn dev_supports_buft(
    _: ggml_backend_dev_t,
    buft: ggml_backend_buffer_type_t,
) -> bool {
    unsafe { !buft.is_null() && ggml_backend_buft_is_host(buft) }
}
unsafe extern "C" fn dev_offload(d: ggml_backend_dev_t, t: *const ggml_tensor) -> bool {
    unsafe { !t.is_null() && (*t).op == GGML_OP_MUL_MAT && dev_supports(d, t) }
}
unsafe extern "C" fn dev_init(d: ggml_backend_dev_t, _: *const c_char) -> ggml_backend_t {
    unsafe {
        let dev = device(d);
        let worker = match Worker::new(dev.provider.clone(), dev.cap) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("ggml-npu: {e}");
                return ptr::null_mut();
            }
        };
        let stream = Box::into_raw(Box::new(Stream {
            worker,
            cap: dev.cap,
        }));
        Box::into_raw(Box::new(ggml_backend {
            guid: ptr::from_ref(&GUID).cast_mut(),
            iface: ggml_backend_i {
                get_name: Some(stream_name),
                free: Some(stream_free),
                graph_compute: Some(stream_compute),
                ..Default::default()
            },
            device: d,
            context: stream.cast(),
        }))
    }
}
unsafe extern "C" fn stream_name(b: ggml_backend_t) -> *const c_char {
    unsafe { dev_name((*b).device) }
}
unsafe extern "C" fn stream_free(b: ggml_backend_t) {
    unsafe {
        if !b.is_null() {
            let b = Box::from_raw(b);
            drop(Box::from_raw(b.context.cast::<Stream>()));
        }
    }
}
unsafe extern "C" fn stream_compute(b: ggml_backend_t, g: *mut ggml_cgraph) -> ggml_status {
    unsafe {
        let s = &*(*b).context.cast::<Stream>();
        let execute = || -> Result<(), String> {
            let captured = graph::capture(g, s.cap)?;
            if captured.outputs.is_empty() {
                return Ok(());
            }
            let inputs = captured
                .inputs
                .iter()
                .map(|&t| graph::read(t))
                .collect::<Result<_, _>>()?;
            let output_sizes = captured
                .outputs
                .iter()
                .map(|&t| graph::byte_len(t))
                .collect::<Result<_, _>>()?;
            let results = s.worker.execute(Job {
                artifact: captured.artifact,
                inputs,
                output_sizes,
            })?;
            if results.len() != captured.outputs.len() {
                return Err("output count mismatch".into());
            }
            for (&t, bytes) in captured.outputs.iter().zip(results) {
                graph::write(t, &bytes)?;
            }
            Ok(())
        };
        match execute() {
            Ok(()) => GGML_STATUS_SUCCESS,
            Err(e) => {
                eprintln!("ggml-npu: {e}");
                GGML_STATUS_FAILED
            }
        }
    }
}

fn buffer_type_iface() -> ggml_backend_buffer_type_i {
    ggml_backend_buffer_type_i {
        get_name: Some(buft_name),
        alloc_buffer: Some(buft_alloc),
        get_alignment: Some(buft_alignment),
        is_host: Some(buft_host),
        ..Default::default()
    }
}
unsafe extern "C" fn buft_name(_: ggml_backend_buffer_type_t) -> *const c_char {
    c"VIRTIO_HOST".as_ptr()
}
unsafe extern "C" fn buft_alignment(_: ggml_backend_buffer_type_t) -> usize {
    64
}
unsafe extern "C" fn buft_host(_: ggml_backend_buffer_type_t) -> bool {
    true
}
unsafe extern "C" fn buft_alloc(
    buft: ggml_backend_buffer_type_t,
    size: usize,
) -> ggml_backend_buffer_t {
    unsafe {
        let Ok(layout) = Layout::from_size_align(size.max(1), 64) else {
            return ptr::null_mut();
        };
        let base = alloc_zeroed(layout);
        if base.is_null() {
            return ptr::null_mut();
        }
        let context = Box::into_raw(Box::new(Buffer { base, layout }));
        ggml_backend_buffer_init(
            buft,
            ggml_backend_buffer_i {
                free_buffer: Some(buffer_free),
                get_base: Some(buffer_base),
                memset_tensor: Some(buffer_memset),
                set_tensor: Some(buffer_set),
                get_tensor: Some(buffer_get),
                clear: Some(buffer_clear),
                ..Default::default()
            },
            context.cast(),
            size,
        )
    }
}
unsafe extern "C" fn buffer_free(b: ggml_backend_buffer_t) {
    unsafe {
        let c = Box::from_raw((*b).context.cast::<Buffer>());
        dealloc(c.base, c.layout);
    }
}
unsafe extern "C" fn buffer_base(b: ggml_backend_buffer_t) -> *mut c_void {
    unsafe { (*(*b).context.cast::<Buffer>()).base.cast() }
}
unsafe fn range(
    b: ggml_backend_buffer_t,
    t: *const ggml_tensor,
    offset: usize,
    size: usize,
) -> *mut u8 {
    unsafe {
        let base = buffer_base(b) as usize;
        let start = ((*t).data as usize)
            .checked_add(offset)
            .expect("tensor offset overflow");
        assert!(
            start >= base
                && start
                    .checked_add(size)
                    .is_some_and(|end| end - base <= (*b).size)
        );
        start as *mut u8
    }
}
unsafe extern "C" fn buffer_memset(
    b: ggml_backend_buffer_t,
    t: *mut ggml_tensor,
    v: u8,
    offset: usize,
    size: usize,
) {
    unsafe {
        ptr::write_bytes(range(b, t, offset, size), v, size);
    }
}
unsafe extern "C" fn buffer_set(
    b: ggml_backend_buffer_t,
    t: *mut ggml_tensor,
    data: *const c_void,
    offset: usize,
    size: usize,
) {
    unsafe {
        ptr::copy(data.cast::<u8>(), range(b, t, offset, size), size);
    }
}
unsafe extern "C" fn buffer_get(
    b: ggml_backend_buffer_t,
    t: *const ggml_tensor,
    data: *mut c_void,
    offset: usize,
    size: usize,
) {
    unsafe {
        ptr::copy(range(b, t, offset, size), data.cast::<u8>(), size);
    }
}
unsafe extern "C" fn buffer_clear(b: ggml_backend_buffer_t, v: u8) {
    unsafe {
        ptr::write_bytes(buffer_base(b).cast::<u8>(), v, (*b).size);
    }
}
