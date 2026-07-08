//! Level Zero runtime integration for the Intel NPU.
//!
//! The loader (`libze_loader.so.1`) is `dlopen`'d at runtime and its functions resolved by
//! `dlsym`, so the cdylib has no hard dependency on Level Zero — on a machine without it,
//! [`probe`] simply returns empty. The loaded function table is a process-lifetime
//! singleton published with a lock-free `AtomicPtr` (no global lock). `no_std`: `dlopen`,
//! `dlsym`, and the ze functions all resolve from the host process.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_void};
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

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

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;

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

/// Resolve one symbol and reinterpret it as a function-pointer type `T`. `transmute_copy`
/// reads the (pointer-sized) address bits as `T`; POSIX guarantees dlsym results are
/// castable to function pointers.
unsafe fn sym<T>(h: *mut c_void, name: *const c_char) -> Option<T> {
    let p = dlsym(h, name);
    if p.is_null() {
        return None;
    }
    Some(mem::transmute_copy::<*mut c_void, T>(&p))
}

unsafe fn load() -> Option<ZeLib> {
    let mut h = dlopen(c"libze_loader.so.1".as_ptr(), RTLD_NOW);
    if h.is_null() {
        h = dlopen(c"libze_loader.so".as_ptr(), RTLD_NOW);
    }
    if h.is_null() {
        return None;
    }
    Some(ZeLib {
        init: sym(h, c"zeInit".as_ptr())?,
        driver_get: sym(h, c"zeDriverGet".as_ptr())?,
        device_get: sym(h, c"zeDeviceGet".as_ptr())?,
        device_get_properties: sym(h, c"zeDeviceGetProperties".as_ptr())?,
        get_ext_fn_addr: sym(h, c"zeDriverGetExtensionFunctionAddress".as_ptr())?,
        context_create: sym(h, c"zeContextCreate".as_ptr())?,
        context_destroy: sym(h, c"zeContextDestroy".as_ptr())?,
        cmd_queue_create: sym(h, c"zeCommandQueueCreate".as_ptr())?,
        cmd_queue_exec: sym(h, c"zeCommandQueueExecuteCommandLists".as_ptr())?,
        cmd_queue_sync: sym(h, c"zeCommandQueueSynchronize".as_ptr())?,
        cmd_queue_destroy: sym(h, c"zeCommandQueueDestroy".as_ptr())?,
        cmd_list_create: sym(h, c"zeCommandListCreate".as_ptr())?,
        cmd_list_close: sym(h, c"zeCommandListClose".as_ptr())?,
        cmd_list_destroy: sym(h, c"zeCommandListDestroy".as_ptr())?,
        mem_alloc_host: sym(h, c"zeMemAllocHost".as_ptr())?,
        mem_free: sym(h, c"zeMemFree".as_ptr())?,
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
