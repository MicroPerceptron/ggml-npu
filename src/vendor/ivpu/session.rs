//! Per-shape Level Zero "session" cache — the bandwidth/latency optimization.
//!
//! The naive path re-created the graph (`pfnCreate` re-parses the blob), re-allocated buffers,
//! re-copied every operand (including the whole weight), built a command list, and tore it all
//! down — on **every** dispatch (~41 ms for a 1Kx1K matmul). Instead we build a [`Session`] once
//! per shape — graph created, host buffers allocated + bound, graph initialized, execute command
//! list recorded — and cache it lock-free. A repeat dispatch then only copies the small
//! activation in, executes the pre-recorded list, syncs, and copies the output out; the weight
//! copy is skipped when the weight pointer is unchanged (weights are constant across tokens).
//!
//! Assumes single-threaded compute per backend (ggml calls `graph_compute` serially) and FP32
//! device I/O (true for the OpenVINO f32 MatMul).

use alloc::boxed::Box;
use core::ffi::c_void;
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use super::cache::ShapeCache;
use super::ze;
use super::ze::raw::*;

/// A prepared, reusable graph for one MatMul shape. Immutable except `last_weight`. Buffers are
/// bound to the graph once; dispatches update their *contents*, not the bindings.
struct Session {
    act_buf: *mut c_void, // activation input (arg 0)
    act_elems: usize,
    wgt_buf: *mut c_void, // weight input (arg 1)
    wgt_elems: usize,
    out_buf: *mut c_void, // output
    out_elems: usize,
    exec_list: ze_command_list_handle_t, // pre-recorded [pfnAppendGraphExecute]
    last_weight: AtomicPtr<f32>,          // skip the weight copy when unchanged
}

// SAFETY: immutable after build except `last_weight` (atomic); compute is single-threaded.
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

static SESSIONS: ShapeCache<Option<&'static Session>> = ShapeCache::new();

fn session(m: i64, k: i64, n: i64) -> Option<&'static Session> {
    SESSIONS.get_or_insert_with((m, k, n), || {
        let blob = super::compile::get_or_compile_matmul(m, k, n)?;
        unsafe { build_session(blob) }
    })
}

unsafe fn build_session(blob: &[u8]) -> Option<&'static Session> {
    let l = ze::lib()?;
    let rt = ze::runtime()?;
    let g = rt.ddi;
    let pfn_create = (*g).pfnCreate?;
    let pfn_props = (*g).pfnGetProperties?;
    let pfn_arg = (*g).pfnGetArgumentProperties?;
    let pfn_set = (*g).pfnSetArgumentValue?;
    let pfn_init = (*g).pfnAppendGraphInitialize?;
    let pfn_exec = (*g).pfnAppendGraphExecute?;

    let mut gd: ze_graph_desc_t = mem::zeroed();
    gd.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC;
    gd.format = ZE_GRAPH_FORMAT_NATIVE;
    gd.inputSize = blob.len();
    gd.pInput = blob.as_ptr();
    let mut graph: ze_graph_handle_t = ptr::null_mut();
    if pfn_create(rt.context, rt.device, &gd, &mut graph) != ZE_RESULT_SUCCESS {
        return None;
    }

    let mut gp: ze_graph_properties_t = mem::zeroed();
    gp.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    if pfn_props(graph, &mut gp) != ZE_RESULT_SUCCESS {
        return None;
    }

    // A host buffer per arg, bound once. Inputs in arg order: [activation (0), weight (1)].
    let mut host_desc: ze_host_mem_alloc_desc_t = mem::zeroed();
    host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    let (mut act_buf, mut act_elems) = (ptr::null_mut(), 0usize);
    let (mut wgt_buf, mut wgt_elems) = (ptr::null_mut(), 0usize);
    let (mut out_buf, mut out_elems) = (ptr::null_mut(), 0usize);
    let mut input_i = 0usize;

    for i in 0..gp.numGraphArgs {
        let mut ap: ze_graph_argument_properties_t = mem::zeroed();
        ap.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        if pfn_arg(graph, i, &mut ap) != ZE_RESULT_SUCCESS {
            return None;
        }
        let mut elems = 1usize;
        for d in 0..(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE as usize) {
            if ap.dims[d] > 0 {
                elems *= ap.dims[d] as usize;
            }
        }
        let psz = match ap.devicePrecision {
            ZE_GRAPH_ARGUMENT_PRECISION_FP32 => 4,
            ZE_GRAPH_ARGUMENT_PRECISION_FP16 => 2,
            _ => 0,
        };
        let mut p: *mut c_void = ptr::null_mut();
        if (l.mem_alloc_host)(rt.context, &host_desc, elems * psz, 4096, &mut p) != ZE_RESULT_SUCCESS {
            return None;
        }
        ptr::write_bytes(p as *mut u8, 0, elems * psz);
        if pfn_set(graph, i, p) != ZE_RESULT_SUCCESS {
            return None;
        }
        if ap.type_ == ZE_GRAPH_ARGUMENT_TYPE_INPUT {
            if input_i == 0 {
                (act_buf, act_elems) = (p, elems);
            } else {
                (wgt_buf, wgt_elems) = (p, elems);
            }
            input_i += 1;
        } else {
            (out_buf, out_elems) = (p, elems);
        }
    }

    // Initialize the graph once.
    let mut ld: ze_command_list_desc_t = mem::zeroed();
    ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    ld.commandQueueGroupOrdinal = 0;
    let mut init_list: ze_command_list_handle_t = ptr::null_mut();
    if (l.cmd_list_create)(rt.context, rt.device, &ld, &mut init_list) != ZE_RESULT_SUCCESS {
        return None;
    }
    let ok = pfn_init(init_list, graph, ptr::null_mut(), 0, ptr::null_mut()) == ZE_RESULT_SUCCESS
        && (l.cmd_list_close)(init_list) == ZE_RESULT_SUCCESS
        && (l.cmd_queue_exec)(rt.queue, 1, &mut init_list, ptr::null_mut()) == ZE_RESULT_SUCCESS
        && (l.cmd_queue_sync)(rt.queue, u64::MAX) == ZE_RESULT_SUCCESS;
    (l.cmd_list_destroy)(init_list);
    if !ok {
        return None;
    }

    // Record a reusable execute command list.
    let mut exec_list: ze_command_list_handle_t = ptr::null_mut();
    if (l.cmd_list_create)(rt.context, rt.device, &ld, &mut exec_list) != ZE_RESULT_SUCCESS {
        return None;
    }
    if pfn_exec(exec_list, graph, ptr::null_mut(), ptr::null_mut(), 0, ptr::null_mut()) != ZE_RESULT_SUCCESS
        || (l.cmd_list_close)(exec_list) != ZE_RESULT_SUCCESS
    {
        return None;
    }

    Some(Box::leak(Box::new(Session {
        act_buf,
        act_elems,
        wgt_buf,
        wgt_elems,
        out_buf,
        out_elems,
        exec_list,
        last_weight: AtomicPtr::new(ptr::null_mut()),
    })))
}

/// Run a MatMul of shape `(M, K, N)` on the NPU, reusing (or building) its session. `activation`
/// is always copied in; `weight` is copied only when its pointer differs from last time.
pub(super) fn run_matmul(m: i64, k: i64, n: i64, activation: &[f32], weight: &[f32], out: &mut [f32]) -> bool {
    let Some(s) = session(m, k, n) else {
        return false;
    };
    let (Some(l), Some(rt)) = (ze::lib(), ze::runtime()) else {
        return false;
    };
    unsafe {
        if activation.len() >= s.act_elems && !s.act_buf.is_null() {
            ptr::copy_nonoverlapping(activation.as_ptr(), s.act_buf as *mut f32, s.act_elems);
        }
        let wptr = weight.as_ptr() as *mut f32;
        if s.last_weight.load(Ordering::Acquire) != wptr {
            if weight.len() >= s.wgt_elems && !s.wgt_buf.is_null() {
                ptr::copy_nonoverlapping(weight.as_ptr(), s.wgt_buf as *mut f32, s.wgt_elems);
            }
            s.last_weight.store(wptr, Ordering::Release);
        }
        let mut list = s.exec_list;
        if (l.cmd_queue_exec)(rt.queue, 1, &mut list, ptr::null_mut()) != ZE_RESULT_SUCCESS
            || (l.cmd_queue_sync)(rt.queue, u64::MAX) != ZE_RESULT_SUCCESS
        {
            return false;
        }
        if !s.out_buf.is_null() {
            let nout = core::cmp::min(out.len(), s.out_elems);
            ptr::copy_nonoverlapping(s.out_buf as *const f32, out.as_mut_ptr(), nout);
        }
    }
    true
}
