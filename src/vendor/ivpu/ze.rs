//! Level Zero runtime integration for the Intel NPU.
//!
//! The selected host dynamic-loader service resolves the loader at runtime, so the cdylib has no
//! hard dependency on Level Zero — on a machine without it, [`probe`] simply returns empty. The
//! loaded function table is a process-lifetime singleton published with a lock-free `AtomicPtr`.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::{c_char, c_void, CStr};
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::host::{DynamicLibraryService, HostEnvironment, LibraryVisibility};

/// bindgen'd Level Zero types + constants (functions are dlsym'd, not linked).
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unused_imports
)]
pub(super) mod raw {
    include!(concat!(env!("OUT_DIR"), "/ze_bindings.rs"));
}
use raw::*;

type PfnInit = unsafe extern "C" fn(ze_init_flags_t) -> ze_result_t;
type PfnDriverGet = unsafe extern "C" fn(*mut u32, *mut ze_driver_handle_t) -> ze_result_t;
type PfnDeviceGet =
    unsafe extern "C" fn(ze_driver_handle_t, *mut u32, *mut ze_device_handle_t) -> ze_result_t;
type PfnDeviceGetProperties =
    unsafe extern "C" fn(ze_device_handle_t, *mut ze_device_properties_t) -> ze_result_t;
type PfnGetExtFnAddr =
    unsafe extern "C" fn(ze_driver_handle_t, *const c_char, *mut *mut c_void) -> ze_result_t;
type PfnContextCreate = unsafe extern "C" fn(
    ze_driver_handle_t,
    *const ze_context_desc_t,
    *mut ze_context_handle_t,
) -> ze_result_t;
type PfnContextDestroy = unsafe extern "C" fn(ze_context_handle_t) -> ze_result_t;
type PfnCmdQueueCreate = unsafe extern "C" fn(
    ze_context_handle_t,
    ze_device_handle_t,
    *const ze_command_queue_desc_t,
    *mut ze_command_queue_handle_t,
) -> ze_result_t;
type PfnCmdQueueExec = unsafe extern "C" fn(
    ze_command_queue_handle_t,
    u32,
    *mut ze_command_list_handle_t,
    ze_fence_handle_t,
) -> ze_result_t;
type PfnCmdQueueSync = unsafe extern "C" fn(ze_command_queue_handle_t, u64) -> ze_result_t;
type PfnCmdQueueDestroy = unsafe extern "C" fn(ze_command_queue_handle_t) -> ze_result_t;
type PfnCmdListCreate = unsafe extern "C" fn(
    ze_context_handle_t,
    ze_device_handle_t,
    *const ze_command_list_desc_t,
    *mut ze_command_list_handle_t,
) -> ze_result_t;
type PfnCmdListClose = unsafe extern "C" fn(ze_command_list_handle_t) -> ze_result_t;
type PfnCmdListDestroy = unsafe extern "C" fn(ze_command_list_handle_t) -> ze_result_t;
type PfnMemAllocHost = unsafe extern "C" fn(
    ze_context_handle_t,
    *const ze_host_mem_alloc_desc_t,
    usize,
    usize,
    *mut *mut c_void,
) -> ze_result_t;
type PfnMemFree = unsafe extern "C" fn(ze_context_handle_t, *mut c_void) -> ze_result_t;

/// Resolved Level Zero entry points. All fields are function pointers (`Send + Sync`).
pub(super) struct ZeLib {
    _library: crate::host::BuiltinLibrary,
    pub(super) init: PfnInit,
    pub(super) driver_get: PfnDriverGet,
    pub(super) device_get: PfnDeviceGet,
    pub(super) device_get_properties: PfnDeviceGetProperties,
    pub(super) get_ext_fn_addr: PfnGetExtFnAddr,
    pub(super) context_create: PfnContextCreate,
    pub(super) context_destroy: PfnContextDestroy,
    pub(super) cmd_queue_create: PfnCmdQueueCreate,
    pub(super) cmd_queue_exec: PfnCmdQueueExec,
    pub(super) cmd_queue_sync: PfnCmdQueueSync,
    pub(super) cmd_queue_destroy: PfnCmdQueueDestroy,
    pub(super) cmd_list_create: PfnCmdListCreate,
    pub(super) cmd_list_close: PfnCmdListClose,
    pub(super) cmd_list_destroy: PfnCmdListDestroy,
    pub(super) mem_alloc_host: PfnMemAllocHost,
    pub(super) mem_free: PfnMemFree,
}

/// `null` = not yet tried, `FAILED` = tried and unavailable, else a leaked `*mut ZeLib`.
static ZE_LIB: AtomicPtr<ZeLib> = AtomicPtr::new(ptr::null_mut());
const FAILED: *mut ZeLib = 1usize as *mut ZeLib;

/// Resolve one symbol and reinterpret it as a function-pointer type `T`.
unsafe fn sym<T>(library: &crate::host::BuiltinLibrary, name: &CStr) -> Option<T> {
    let p = unsafe {
        crate::host::builtin()
            .libraries()
            .symbol(library, name)
            .ok()?
    };
    Some(mem::transmute_copy::<*mut c_void, T>(&p))
}

unsafe fn load() -> Option<ZeLib> {
    let libraries = crate::host::builtin().libraries();
    let library = libraries
        .open(c"libze_loader.so.1", LibraryVisibility::Local)
        .or_else(|_| libraries.open(c"libze_loader.so", LibraryVisibility::Local))
        .ok()?;
    Some(ZeLib {
        init: sym(&library, c"zeInit")?,
        driver_get: sym(&library, c"zeDriverGet")?,
        device_get: sym(&library, c"zeDeviceGet")?,
        device_get_properties: sym(&library, c"zeDeviceGetProperties")?,
        get_ext_fn_addr: sym(&library, c"zeDriverGetExtensionFunctionAddress")?,
        context_create: sym(&library, c"zeContextCreate")?,
        context_destroy: sym(&library, c"zeContextDestroy")?,
        cmd_queue_create: sym(&library, c"zeCommandQueueCreate")?,
        cmd_queue_exec: sym(&library, c"zeCommandQueueExecuteCommandLists")?,
        cmd_queue_sync: sym(&library, c"zeCommandQueueSynchronize")?,
        cmd_queue_destroy: sym(&library, c"zeCommandQueueDestroy")?,
        cmd_list_create: sym(&library, c"zeCommandListCreate")?,
        cmd_list_close: sym(&library, c"zeCommandListClose")?,
        cmd_list_destroy: sym(&library, c"zeCommandListDestroy")?,
        mem_alloc_host: sym(&library, c"zeMemAllocHost")?,
        mem_free: sym(&library, c"zeMemFree")?,
        _library: library,
    })
}

pub(super) fn lib() -> Option<&'static ZeLib> {
    let cur = ZE_LIB.load(Ordering::Acquire);
    if cur == FAILED {
        return None;
    }
    if !cur.is_null() {
        return Some(unsafe { &*cur });
    }
    match unsafe { load() } {
        Some(l) => {
            let boxed = Box::into_raw(Box::new(l));
            match ZE_LIB.compare_exchange(
                ptr::null_mut(),
                boxed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => Some(unsafe { &*boxed }),
                Err(winner) => {
                    drop(unsafe { Box::from_raw(boxed) });
                    if winner == FAILED {
                        None
                    } else {
                        Some(unsafe { &*winner })
                    }
                }
            }
        }
        None => {
            let _ = ZE_LIB.compare_exchange(
                ptr::null_mut(),
                FAILED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            None
        }
    }
}

/// Allocate `size` bytes of NPU-accessible host memory in the shared runtime context. Such
/// memory is pinned + mapped into the NPU's address space, so its pointer can be bound as a
/// graph argument with **no per-dispatch copy** (unlike a plain `malloc` pointer, which the NPU
/// cannot DMA). Page-aligned (matches the arg allocations in [`super::session`]). Returns null
/// if Level Zero / the NPU is absent or the allocation fails — the caller then falls back to
/// ordinary host memory (correct, just not zero-copy).
pub(super) fn alloc_host(size: usize) -> *mut c_void {
    let (Some(l), Some(rt)) = (lib(), runtime()) else {
        return ptr::null_mut();
    };
    unsafe {
        let mut hd: ze_host_mem_alloc_desc_t = mem::zeroed();
        hd.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        let mut p: *mut c_void = ptr::null_mut();
        if (l.mem_alloc_host)(rt.context, &hd, size.max(1), 4096, &mut p) != ZE_RESULT_SUCCESS {
            return ptr::null_mut();
        }
        p
    }
}

/// Free memory returned by [`alloc_host`]. No-op on a null pointer or absent runtime.
pub(super) fn free_host(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    if let (Some(l), Some(rt)) = (lib(), runtime()) {
        unsafe {
            (l.mem_free)(rt.context, p);
        }
    }
}

/// A discovered Level Zero NPU device (and the driver that owns it, needed to create a
/// context and fetch the graph dispatch table).
pub struct Device {
    pub driver: ze_driver_handle_t,
    pub handle: ze_device_handle_t,
    pub name: String,
}

fn name_from(buf: &[c_char]) -> String {
    let bytes = unsafe { core::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len()) };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Enumerate Level Zero devices and keep those of type VPU (the NPU). Cheap (the loader is
/// cached); returns empty if Level Zero or the NPU is absent. Never panics.
pub fn probe() -> Vec<Device> {
    let mut out = Vec::new();
    let Some(l) = lib() else {
        return out;
    };
    unsafe {
        if (l.init)(0) != ZE_RESULT_SUCCESS {
            return out;
        }
        let mut n_drv: u32 = 0;
        if (l.driver_get)(&mut n_drv, ptr::null_mut()) != ZE_RESULT_SUCCESS || n_drv == 0 {
            return out;
        }
        let mut drivers: Vec<ze_driver_handle_t> = Vec::new();
        drivers.resize(n_drv as usize, ptr::null_mut());
        if (l.driver_get)(&mut n_drv, drivers.as_mut_ptr()) != ZE_RESULT_SUCCESS {
            return out;
        }
        for &drv in drivers.iter().take(n_drv as usize) {
            let mut n_dev: u32 = 0;
            if (l.device_get)(drv, &mut n_dev, ptr::null_mut()) != ZE_RESULT_SUCCESS || n_dev == 0 {
                continue;
            }
            let mut devs: Vec<ze_device_handle_t> = Vec::new();
            devs.resize(n_dev as usize, ptr::null_mut());
            if (l.device_get)(drv, &mut n_dev, devs.as_mut_ptr()) != ZE_RESULT_SUCCESS {
                continue;
            }
            for &dev in devs.iter().take(n_dev as usize) {
                let mut props: ze_device_properties_t = mem::zeroed();
                props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                props.pNext = ptr::null_mut();
                if (l.device_get_properties)(dev, &mut props) != ZE_RESULT_SUCCESS {
                    continue;
                }
                if props.type_ == ZE_DEVICE_TYPE_VPU {
                    out.push(Device {
                        driver: drv,
                        handle: dev,
                        name: name_from(&props.name),
                    });
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Cached device runtime — context, graph dispatch table, and command queue, built once.
// ---------------------------------------------------------------------------

/// Process-lifetime Level Zero runtime for the NPU: driver/device plus a context, the graph
/// dispatch table, and a command queue — all created once and reused by every session.
pub(super) struct Runtime {
    pub(super) driver: ze_driver_handle_t,
    pub(super) device: ze_device_handle_t,
    pub(super) context: ze_context_handle_t,
    pub(super) ddi: *mut ze_graph_dditable_ext_t,
    pub(super) queue: ze_command_queue_handle_t,
    pub(super) name: String,
}

// SAFETY: immutable after init; the Level Zero handles are shared read-only across threads.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

static RUNTIME: AtomicPtr<Runtime> = AtomicPtr::new(ptr::null_mut());
const RT_FAILED: *mut Runtime = 1usize as *mut Runtime;

unsafe fn build_runtime() -> Option<Runtime> {
    let l = lib()?;
    let dev = probe().into_iter().next()?; // zeInit + enumerate; keeps the VPU device

    let mut ctx_desc: ze_context_desc_t = mem::zeroed();
    ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    let mut context: ze_context_handle_t = ptr::null_mut();
    if (l.context_create)(dev.driver, &ctx_desc, &mut context) != ZE_RESULT_SUCCESS {
        return None;
    }

    let mut table: *mut c_void = ptr::null_mut();
    if (l.get_ext_fn_addr)(dev.driver, c"ZE_extension_graph".as_ptr(), &mut table)
        != ZE_RESULT_SUCCESS
        || table.is_null()
    {
        return None;
    }

    let mut qd: ze_command_queue_desc_t = mem::zeroed();
    qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qd.ordinal = 0;
    qd.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    let mut queue: ze_command_queue_handle_t = ptr::null_mut();
    if (l.cmd_queue_create)(context, dev.handle, &qd, &mut queue) != ZE_RESULT_SUCCESS {
        return None;
    }

    Some(Runtime {
        driver: dev.driver,
        device: dev.handle,
        context,
        ddi: table as *mut ze_graph_dditable_ext_t,
        queue,
        name: dev.name,
    })
}

/// The cached device runtime, initialized on first use (and cached even on failure).
pub(super) fn runtime() -> Option<&'static Runtime> {
    let cur = RUNTIME.load(Ordering::Acquire);
    if cur == RT_FAILED {
        return None;
    }
    if !cur.is_null() {
        return Some(unsafe { &*cur });
    }
    match unsafe { build_runtime() } {
        Some(rt) => {
            let boxed = Box::into_raw(Box::new(rt));
            match RUNTIME.compare_exchange(
                ptr::null_mut(),
                boxed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => Some(unsafe { &*boxed }),
                Err(w) => {
                    drop(unsafe { Box::from_raw(boxed) });
                    if w == RT_FAILED {
                        None
                    } else {
                        Some(unsafe { &*w })
                    }
                }
            }
        }
        None => {
            let _ = RUNTIME.compare_exchange(
                ptr::null_mut(),
                RT_FAILED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            None
        }
    }
}
